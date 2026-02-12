#include "optimizer/admm_kernels.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

#include "utils/cuda_utils.h"

namespace cpo {

// ── GPU kernel: R-U objective evaluation ────────────────────────────
//
// Each thread handles one scenario:
//   1. Compute loss_i = -r_i'w (dot product via shared-memory weights)
//   2. If loss_i > zeta, accumulate excess and gradient contribution
//   3. Block-level reduction, then atomicAdd to global accumulators
//
// This is the same pattern as k_compute_portfolio_loss but extended
// to also accumulate the gradient.

/// Kernel: compute R-U objective excess and gradient across scenarios.
///
/// Layout: scenario matrix is column-major —
///   r(scenario, asset) = d_scenarios[asset * n_scenarios + scenario]
///
/// @param d_scenarios Column-major scenario matrix (float).
/// @param d_weights Portfolio weights (float, in shared memory).
/// @param n_scenarios Number of scenarios.
/// @param n_assets Number of assets.
/// @param zeta VaR estimate (threshold for tail).
/// @param d_sum_excess Output: sum of max(0, loss_i - zeta).
/// @param d_grad_w Output: accumulated gradient (double, n_assets).
/// @param d_count Output: number of tail scenarios.
__global__ void k_evaluate_ru_objective(
    const float* __restrict__ d_scenarios,
    const float* __restrict__ d_weights,
    int n_scenarios, int n_assets, float zeta,
    double* __restrict__ d_sum_excess,
    double* __restrict__ d_grad_w,
    int* __restrict__ d_count) {
    // Shared memory: weights (n_assets floats).
    extern __shared__ float s_weights[];
    for (int j = threadIdx.x; j < n_assets; j += blockDim.x) {
        s_weights[j] = d_weights[j];
    }
    __syncthreads();

    int sid = blockIdx.x * blockDim.x + threadIdx.x;
    if (sid >= n_scenarios) return;

    // Compute loss_i = -r_i' w.
    float dot = 0.0f;
    for (int j = 0; j < n_assets; ++j) {
        dot += s_weights[j] * d_scenarios[j * n_scenarios + sid];
    }
    float loss = -dot;

    // Check if this scenario is in the tail.
    float excess = loss - zeta;
    if (excess > 0.0f) {
        // Accumulate excess (double precision for accuracy).
        atomicAdd(d_sum_excess, static_cast<double>(excess));
        atomicAdd(d_count, 1);

        // Accumulate gradient: -r_i for each tail scenario.
        for (int j = 0; j < n_assets; ++j) {
            double r_ij = static_cast<double>(
                d_scenarios[j * n_scenarios + sid]);
            atomicAdd(&d_grad_w[j], -r_ij);
        }
    }
}

// ── Host wrapper ────────────────────────────────────────────────────

GpuObjectiveResult evaluate_objective_gpu(const ScenarioMatrix& scenarios,
                                           const VectorXs& w,
                                           Scalar zeta,
                                           int threads_per_block) {
    const int n_scenarios = scenarios.n_scenarios();
    const int n_assets = scenarios.n_assets();

    if (w.size() != n_assets) {
        throw std::runtime_error(
            "evaluate_objective_gpu: w size (" +
            std::to_string(w.size()) + ") != n_assets (" +
            std::to_string(n_assets) + ")");
    }

    // Upload weights to device.
    float* d_weights = nullptr;
    size_t w_bytes = static_cast<size_t>(n_assets) * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_weights, w_bytes));
    CUDA_CHECK(cudaMemcpy(d_weights, w.data(), w_bytes,
                          cudaMemcpyHostToDevice));

    // Allocate output accumulators on device.
    double* d_sum_excess = nullptr;
    double* d_grad_w = nullptr;
    int* d_count = nullptr;

    CUDA_CHECK(cudaMalloc(&d_sum_excess, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_grad_w,
                          static_cast<size_t>(n_assets) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));

    CUDA_CHECK(cudaMemset(d_sum_excess, 0, sizeof(double)));
    CUDA_CHECK(cudaMemset(d_grad_w, 0,
                          static_cast<size_t>(n_assets) * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));

    // Launch kernel.
    int blocks = (n_scenarios + threads_per_block - 1) / threads_per_block;
    size_t smem = static_cast<size_t>(n_assets) * sizeof(float);

    k_evaluate_ru_objective<<<blocks, threads_per_block, smem>>>(
        scenarios.device_ptr(), d_weights,
        n_scenarios, n_assets, zeta,
        d_sum_excess, d_grad_w, d_count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results to host.
    double h_sum_excess = 0.0;
    int h_count = 0;
    std::vector<double> h_grad_w(n_assets, 0.0);

    CUDA_CHECK(cudaMemcpy(&h_sum_excess, d_sum_excess, sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_grad_w.data(), d_grad_w,
                          static_cast<size_t>(n_assets) * sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_count, d_count, sizeof(int),
                          cudaMemcpyDeviceToHost));

    // Cleanup.
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_sum_excess));
    CUDA_CHECK(cudaFree(d_grad_w));
    CUDA_CHECK(cudaFree(d_count));

    // Assemble result.
    GpuObjectiveResult result;
    result.value = h_sum_excess;
    result.grad_w = Eigen::Map<VectorXd>(h_grad_w.data(), n_assets);
    result.count_exceeding = h_count;

    return result;
}

}  // namespace cpo
