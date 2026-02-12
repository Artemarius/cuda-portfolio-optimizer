#include "risk/portfolio_loss.h"

#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

#include "utils/cuda_utils.h"

namespace cpo {

// ── GPU kernel ──────────────────────────────────────────────────────

/// Compute portfolio loss for each scenario.
///
/// One thread per scenario. Weights loaded into shared memory for reuse
/// across all threads in a block.
///
/// Layout: scenario matrix is column-major — element (scenario, asset)
///   at d_scenarios[asset * n_scenarios + scenario].
/// This gives coalesced reads: threads in a warp access consecutive
/// elements within the same column.
///
/// loss_i = -sum_j(w_j * r_{i,j})
///
/// @param d_scenarios Column-major scenario matrix (n_scenarios x n_assets).
/// @param d_weights Portfolio weights (n_assets).
/// @param d_losses Output loss vector (n_scenarios).
/// @param n_scenarios Number of scenarios.
/// @param n_assets Number of assets.
__global__ void k_compute_portfolio_loss(
    const float* __restrict__ d_scenarios,
    const float* __restrict__ d_weights,
    float* __restrict__ d_losses,
    int n_scenarios, int n_assets) {
    // Load weights into shared memory — 2KB for 500 assets (float).
    extern __shared__ float s_weights[];
    for (int j = threadIdx.x; j < n_assets; j += blockDim.x) {
        s_weights[j] = d_weights[j];
    }
    __syncthreads();

    int sid = blockIdx.x * blockDim.x + threadIdx.x;
    if (sid >= n_scenarios) return;

    float dot = 0.0f;
    for (int j = 0; j < n_assets; ++j) {
        // Column-major: r(sid, j) = d_scenarios[j * n_scenarios + sid]
        dot += s_weights[j] * d_scenarios[j * n_scenarios + sid];
    }
    d_losses[sid] = -dot;
}

// ── Host wrapper ────────────────────────────────────────────────────

DeviceVector<Scalar> compute_portfolio_loss_gpu(
    const ScenarioMatrix& scenarios, const VectorXs& weights,
    const RiskConfig& config) {
    const Index n_scenarios = scenarios.n_scenarios();
    const Index n_assets = scenarios.n_assets();

    if (weights.size() != n_assets) {
        throw std::runtime_error(
            "compute_portfolio_loss_gpu: weights size (" +
            std::to_string(weights.size()) + ") != n_assets (" +
            std::to_string(n_assets) + ")");
    }

    spdlog::debug("Portfolio loss GPU: {} scenarios x {} assets",
                  n_scenarios, n_assets);

    // Upload weights to device.
    Scalar* d_weights = nullptr;
    size_t w_bytes = static_cast<size_t>(n_assets) * sizeof(Scalar);
    CUDA_CHECK(cudaMalloc(&d_weights, w_bytes));
    CUDA_CHECK(cudaMemcpy(d_weights, weights.data(), w_bytes,
                          cudaMemcpyHostToDevice));

    // Allocate output loss vector.
    DeviceVector<Scalar> d_losses(n_scenarios);

    // Launch kernel with shared memory for weights.
    int threads = config.threads_per_block;
    int blocks = (n_scenarios + threads - 1) / threads;
    size_t smem = static_cast<size_t>(n_assets) * sizeof(Scalar);

    k_compute_portfolio_loss<<<blocks, threads, smem>>>(
        scenarios.device_ptr(), d_weights, d_losses.device_ptr(),
        n_scenarios, n_assets);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Cleanup device weight vector.
    CUDA_CHECK(cudaFree(d_weights));

    return d_losses;
}

// ── CPU reference implementation ────────────────────────────────────

VectorXd compute_portfolio_loss_cpu(const MatrixXd& scenarios_host,
                                    const VectorXd& weights) {
    if (weights.size() != scenarios_host.cols()) {
        throw std::runtime_error(
            "compute_portfolio_loss_cpu: weights size (" +
            std::to_string(weights.size()) + ") != n_assets (" +
            std::to_string(scenarios_host.cols()) + ")");
    }

    // loss = -(scenarios * weights)
    return -(scenarios_host * weights);
}

}  // namespace cpo
