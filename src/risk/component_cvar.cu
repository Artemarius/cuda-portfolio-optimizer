#include "risk/component_cvar.h"

#include <cmath>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

#include "risk/cvar.h"
#include "risk/portfolio_loss.h"
#include "utils/cuda_utils.h"

namespace cpo {

// ── GPU kernel ──────────────────────────────────────────────────────

/// Compute per-asset CVaR contributions for tail scenarios.
///
/// One thread per scenario. For scenarios where loss >= var_threshold,
/// accumulate w_j * (-r_{i,j}) to the component CVaR accumulators
/// via atomicAdd (double precision).
///
/// Weights are loaded into shared memory for reuse across all threads.
///
/// Layout: scenario matrix is column-major — element (scenario, asset)
///   at d_scenarios[asset * n_scenarios + scenario].
///
/// @param d_scenarios Column-major scenario matrix (n_scenarios x n_assets).
/// @param d_weights Portfolio weights (n_assets, float).
/// @param d_losses Portfolio loss vector (n_scenarios, float).
/// @param d_component Output per-asset CVaR accumulators (n_assets, double).
/// @param d_tail_count Output tail scenario count (int).
/// @param var_threshold VaR threshold (float).
/// @param n_scenarios Number of scenarios.
/// @param n_assets Number of assets.
__global__ void k_compute_component_cvar(
    const float* __restrict__ d_scenarios,
    const float* __restrict__ d_weights,
    const float* __restrict__ d_losses,
    double* __restrict__ d_component,
    int* __restrict__ d_tail_count,
    float var_threshold,
    int n_scenarios, int n_assets) {
    // Load weights into shared memory.
    extern __shared__ float s_weights[];
    for (int j = threadIdx.x; j < n_assets; j += blockDim.x) {
        s_weights[j] = d_weights[j];
    }
    __syncthreads();

    int sid = blockIdx.x * blockDim.x + threadIdx.x;
    if (sid >= n_scenarios) return;

    float loss = d_losses[sid];
    if (loss >= var_threshold) {
        atomicAdd(d_tail_count, 1);
        for (int j = 0; j < n_assets; ++j) {
            // r_{i,j} = d_scenarios[j * n_scenarios + sid]
            // Component contribution: w_j * (-r_{i,j}) = w_j * loss_j
            double r_ij = static_cast<double>(
                d_scenarios[j * n_scenarios + sid]);
            double contrib = static_cast<double>(s_weights[j]) * (-r_ij);
            atomicAdd(&d_component[j], contrib);
        }
    }
}

// ── GPU host wrapper ────────────────────────────────────────────────

VectorXd compute_component_cvar_gpu(const ScenarioMatrix& scenarios,
                                     const VectorXs& weights,
                                     const DeviceVector<Scalar>& d_losses,
                                     ScalarCPU var,
                                     const RiskConfig& config) {
    const Index n_scenarios = scenarios.n_scenarios();
    const Index n_assets = scenarios.n_assets();

    if (weights.size() != n_assets) {
        throw std::runtime_error(
            "compute_component_cvar_gpu: weights size (" +
            std::to_string(weights.size()) + ") != n_assets (" +
            std::to_string(n_assets) + ")");
    }
    if (d_losses.size() != n_scenarios) {
        throw std::runtime_error(
            "compute_component_cvar_gpu: d_losses size (" +
            std::to_string(d_losses.size()) + ") != n_scenarios (" +
            std::to_string(n_scenarios) + ")");
    }

    spdlog::debug("Component CVaR GPU: {} scenarios x {} assets, VaR={:.6f}",
                  n_scenarios, n_assets, var);

    // Upload weights to device.
    Scalar* d_weights = nullptr;
    size_t w_bytes = static_cast<size_t>(n_assets) * sizeof(Scalar);
    CUDA_CHECK(cudaMalloc(&d_weights, w_bytes));
    CUDA_CHECK(cudaMemcpy(d_weights, weights.data(), w_bytes,
                          cudaMemcpyHostToDevice));

    // Allocate per-asset component accumulators (double) and tail count.
    double* d_component = nullptr;
    int* d_tail_count = nullptr;
    size_t comp_bytes = static_cast<size_t>(n_assets) * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_component, comp_bytes));
    CUDA_CHECK(cudaMalloc(&d_tail_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_component, 0, comp_bytes));
    CUDA_CHECK(cudaMemset(d_tail_count, 0, sizeof(int)));

    // Launch kernel.
    int threads = config.threads_per_block;
    int blocks = (n_scenarios + threads - 1) / threads;
    size_t smem = static_cast<size_t>(n_assets) * sizeof(Scalar);

    float var_f = static_cast<float>(var);
    k_compute_component_cvar<<<blocks, threads, smem>>>(
        scenarios.device_ptr(), d_weights, d_losses.device_ptr(),
        d_component, d_tail_count, var_f, n_scenarios, n_assets);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Download results.
    std::vector<double> h_component(n_assets);
    int h_tail_count = 0;
    CUDA_CHECK(cudaMemcpy(h_component.data(), d_component, comp_bytes,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_tail_count, d_tail_count, sizeof(int),
                          cudaMemcpyDeviceToHost));

    // Cleanup.
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_component));
    CUDA_CHECK(cudaFree(d_tail_count));

    // Normalize by tail count.
    VectorXd result(n_assets);
    if (h_tail_count > 0) {
        for (Index j = 0; j < n_assets; ++j) {
            result(j) = h_component[j] / h_tail_count;
        }
    } else {
        result.setZero();
    }

    spdlog::debug("Component CVaR GPU: tail_count={}, sum={:.6f}",
                  h_tail_count, result.sum());

    return result;
}

// ── CPU reference implementation ────────────────────────────────────

VectorXd compute_component_cvar_cpu(const MatrixXd& scenarios_host,
                                     const VectorXd& weights,
                                     const VectorXd& losses,
                                     ScalarCPU var,
                                     const RiskConfig& config) {
    const Index n_scenarios = static_cast<Index>(scenarios_host.rows());
    const Index n_assets = static_cast<Index>(scenarios_host.cols());

    if (weights.size() != n_assets) {
        throw std::runtime_error(
            "compute_component_cvar_cpu: weights size (" +
            std::to_string(weights.size()) + ") != n_assets (" +
            std::to_string(n_assets) + ")");
    }
    if (losses.size() != n_scenarios) {
        throw std::runtime_error(
            "compute_component_cvar_cpu: losses size (" +
            std::to_string(losses.size()) + ") != n_scenarios (" +
            std::to_string(n_scenarios) + ")");
    }

    VectorXd component = VectorXd::Zero(n_assets);
    int tail_count = 0;

    for (Index i = 0; i < n_scenarios; ++i) {
        if (losses(i) >= var) {
            ++tail_count;
            for (Index j = 0; j < n_assets; ++j) {
                // Component contribution: w_j * (-r_{i,j})
                component(j) += weights(j) * (-scenarios_host(i, j));
            }
        }
    }

    if (tail_count > 0) {
        component /= tail_count;
    }

    return component;
}

// ── Convenience functions ───────────────────────────────────────────

std::pair<RiskResult, VectorXd> compute_portfolio_risk_decomp_gpu(
    const ScenarioMatrix& scenarios,
    const VectorXs& weights,
    const RiskConfig& config) {
    auto d_losses = compute_portfolio_loss_gpu(scenarios, weights, config);
    auto risk = compute_risk_gpu(d_losses, config);
    auto component = compute_component_cvar_gpu(
        scenarios, weights, d_losses, risk.var, config);
    return {risk, std::move(component)};
}

std::pair<RiskResult, VectorXd> compute_portfolio_risk_decomp_cpu(
    const MatrixXd& scenarios_host,
    const VectorXd& weights,
    const RiskConfig& config) {
    VectorXd losses = compute_portfolio_loss_cpu(scenarios_host, weights);
    auto risk = compute_risk_cpu(losses, config);
    auto component = compute_component_cvar_cpu(
        scenarios_host, weights, losses, risk.var, config);
    return {risk, std::move(component)};
}

}  // namespace cpo
