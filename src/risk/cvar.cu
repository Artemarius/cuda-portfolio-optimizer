#include "risk/cvar.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

#include "risk/portfolio_loss.h"
#include "utils/cuda_utils.h"

namespace cpo {

// ── Statistics reduction kernel ─────────────────────────────────────

/// Single-pass reduction over the loss vector computing three accumulators:
///   sum(loss), sum(loss^2), sum(max(-loss, 0)^2)
///
/// -loss = portfolio return, so max(-loss, 0)^2 accumulates the squared
/// positive returns, which is the downside deviation denominator
/// (using 0 as the threshold — semi-deviation convention).
///
/// Block-level shared-memory reduction + atomicAdd to global accumulators.
///
/// @param d_losses Input loss vector (n elements).
/// @param n Number of elements.
/// @param d_sum Output: sum of losses.
/// @param d_sum_sq Output: sum of squared losses.
/// @param d_sum_downside_sq Output: sum of max(-loss, 0)^2.
__global__ void k_compute_loss_statistics(
    const float* __restrict__ d_losses, int n,
    double* __restrict__ d_sum,
    double* __restrict__ d_sum_sq,
    double* __restrict__ d_sum_downside_sq) {
    // Shared memory for block-level reduction (3 accumulators per thread).
    extern __shared__ double s_data[];
    double* s_sum = s_data;
    double* s_sum_sq = s_data + blockDim.x;
    double* s_down_sq = s_data + 2 * blockDim.x;

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Grid-stride loop: each thread accumulates multiple elements.
    double local_sum = 0.0;
    double local_sum_sq = 0.0;
    double local_down_sq = 0.0;

    for (int i = gid; i < n; i += stride) {
        double val = static_cast<double>(d_losses[i]);
        local_sum += val;
        local_sum_sq += val * val;
        // -loss = return; if return > 0, it's not downside.
        // Downside: when return < 0, i.e., loss > 0.
        // Semi-deviation uses max(-return, 0)^2 = max(loss, 0)^2
        // But for Sortino, downside = returns below target (0).
        // return = -loss, downside_sq = max(-return, 0)^2 = max(loss, 0)^2
        double down = (val > 0.0) ? val : 0.0;
        local_down_sq += down * down;
    }

    s_sum[tid] = local_sum;
    s_sum_sq[tid] = local_sum_sq;
    s_down_sq[tid] = local_down_sq;
    __syncthreads();

    // Block-level reduction (power-of-2 stride).
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_sum_sq[tid] += s_sum_sq[tid + s];
            s_down_sq[tid] += s_down_sq[tid + s];
        }
        __syncthreads();
    }

    // Thread 0 writes block result to global via atomicAdd.
    if (tid == 0) {
        atomicAdd(d_sum, s_sum[0]);
        atomicAdd(d_sum_sq, s_sum_sq[0]);
        atomicAdd(d_sum_downside_sq, s_down_sq[0]);
    }
}

// ── GPU CVaR computation ────────────────────────────────────────────

RiskResult compute_risk_gpu(const DeviceVector<Scalar>& d_losses,
                            const RiskConfig& config) {
    const Index n = d_losses.size();
    const ScalarCPU alpha = config.confidence_level;

    if (n <= 0) {
        throw std::runtime_error("compute_risk_gpu: empty loss vector");
    }
    if (alpha <= 0.0 || alpha >= 1.0) {
        throw std::runtime_error(
            "compute_risk_gpu: confidence_level must be in (0, 1), got " +
            std::to_string(alpha));
    }

    spdlog::debug("CVaR GPU: {} scenarios, alpha={}", n, alpha);

    // ── Step 1: Sort losses ascending using CUB ─────────────────────
    DeviceVector<Scalar> d_sorted(n);

    // Determine temporary storage requirement.
    size_t temp_bytes = 0;
    cub::DeviceRadixSort::SortKeys(
        nullptr, temp_bytes,
        d_losses.device_ptr(), d_sorted.device_ptr(), n);

    // Allocate temp storage and run sort.
    void* d_temp = nullptr;
    CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));
    cub::DeviceRadixSort::SortKeys(
        d_temp, temp_bytes,
        d_losses.device_ptr(), d_sorted.device_ptr(), n);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_temp));

    // ── Step 2: VaR and CVaR from sorted losses ─────────────────────
    // VaR index: floor(alpha * N). Elements at indices [var_index, N-1]
    // are the worst (1-alpha) fraction.
    const Index var_index = static_cast<Index>(std::floor(alpha * n));
    const Index n_tail = n - var_index;

    // Compute tail sum using CUB DeviceReduce::Sum on the tail portion.
    Scalar* d_tail_sum_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_tail_sum_out, sizeof(Scalar)));

    // Determine temp storage for reduction.
    size_t reduce_temp_bytes = 0;
    cub::DeviceReduce::Sum(
        nullptr, reduce_temp_bytes,
        d_sorted.device_ptr() + var_index, d_tail_sum_out, n_tail);

    void* d_reduce_temp = nullptr;
    CUDA_CHECK(cudaMalloc(&d_reduce_temp, reduce_temp_bytes));
    cub::DeviceReduce::Sum(
        d_reduce_temp, reduce_temp_bytes,
        d_sorted.device_ptr() + var_index, d_tail_sum_out, n_tail);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_reduce_temp));

    // Copy VaR value and tail sum to host.
    Scalar h_var_value = 0.0f;
    Scalar h_tail_sum = 0.0f;
    CUDA_CHECK(cudaMemcpy(&h_var_value, d_sorted.device_ptr() + var_index,
                          sizeof(Scalar), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_tail_sum, d_tail_sum_out, sizeof(Scalar),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_tail_sum_out));

    ScalarCPU var = static_cast<ScalarCPU>(h_var_value);
    ScalarCPU cvar = static_cast<ScalarCPU>(h_tail_sum) / n_tail;

    // ── Step 3: Loss statistics (on original unsorted losses) ───────
    double* d_sum = nullptr;
    double* d_sum_sq = nullptr;
    double* d_sum_downside_sq = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sum, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sum_sq, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sum_downside_sq, sizeof(double)));
    CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(double)));
    CUDA_CHECK(cudaMemset(d_sum_sq, 0, sizeof(double)));
    CUDA_CHECK(cudaMemset(d_sum_downside_sq, 0, sizeof(double)));

    int threads = config.threads_per_block;
    int blocks = std::min((n + threads - 1) / threads, 1024);
    size_t smem = 3 * static_cast<size_t>(threads) * sizeof(double);

    k_compute_loss_statistics<<<blocks, threads, smem>>>(
        d_losses.device_ptr(), n, d_sum, d_sum_sq, d_sum_downside_sq);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy statistics to host.
    double h_sum = 0.0, h_sum_sq = 0.0, h_sum_downside_sq = 0.0;
    CUDA_CHECK(cudaMemcpy(&h_sum, d_sum, sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_sum_sq, d_sum_sq, sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_sum_downside_sq, d_sum_downside_sq,
                          sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_sum));
    CUDA_CHECK(cudaFree(d_sum_sq));
    CUDA_CHECK(cudaFree(d_sum_downside_sq));

    // ── Step 4: Derive statistics ───────────────────────────────────
    // loss = -return, so expected_return = -mean(loss)
    double mean_loss = h_sum / n;
    double expected_return = -mean_loss;

    // Var[loss] = E[loss^2] - E[loss]^2
    double variance = h_sum_sq / n - mean_loss * mean_loss;
    double volatility = std::sqrt(std::max(variance, 0.0));

    // Sharpe = expected_return / volatility (risk-free rate assumed 0)
    double sharpe = (volatility > 1e-15) ? expected_return / volatility : 0.0;

    // Sortino = expected_return / downside_deviation
    // downside_deviation = sqrt(mean(max(loss, 0)^2))
    // (loss > 0 means negative return — the downside)
    double downside_var = h_sum_downside_sq / n;
    double downside_dev = std::sqrt(std::max(downside_var, 0.0));
    double sortino = (downside_dev > 1e-15) ? expected_return / downside_dev
                                            : 0.0;

    RiskResult result;
    result.var = var;
    result.cvar = cvar;
    result.expected_return = expected_return;
    result.volatility = volatility;
    result.sharpe_ratio = sharpe;
    result.sortino_ratio = sortino;
    result.confidence_level = alpha;
    result.n_scenarios = n;

    spdlog::debug("CVaR GPU result: VaR={:.6f} CVaR={:.6f} E[r]={:.6f} "
                  "vol={:.6f} Sharpe={:.4f} Sortino={:.4f}",
                  var, cvar, expected_return, volatility, sharpe, sortino);

    return result;
}

// ── CPU reference implementation ────────────────────────────────────

RiskResult compute_risk_cpu(const VectorXd& losses,
                            const RiskConfig& config) {
    const Index n = static_cast<Index>(losses.size());
    const ScalarCPU alpha = config.confidence_level;

    if (n <= 0) {
        throw std::runtime_error("compute_risk_cpu: empty loss vector");
    }
    if (alpha <= 0.0 || alpha >= 1.0) {
        throw std::runtime_error(
            "compute_risk_cpu: confidence_level must be in (0, 1), got " +
            std::to_string(alpha));
    }

    // Sort a copy of losses ascending.
    std::vector<double> sorted(losses.data(), losses.data() + n);
    std::sort(sorted.begin(), sorted.end());

    // VaR = sorted[floor(alpha * N)]
    const Index var_index = static_cast<Index>(std::floor(alpha * n));
    const Index n_tail = n - var_index;
    double var = sorted[var_index];

    // CVaR = mean of sorted[var_index .. N-1]
    double tail_sum = 0.0;
    for (Index i = var_index; i < n; ++i) {
        tail_sum += sorted[i];
    }
    double cvar = tail_sum / n_tail;

    // Statistics from the original (unsorted) losses.
    double sum_loss = 0.0;
    double sum_loss_sq = 0.0;
    double sum_downside_sq = 0.0;

    for (Index i = 0; i < n; ++i) {
        double l = losses(i);
        sum_loss += l;
        sum_loss_sq += l * l;
        if (l > 0.0) {
            sum_downside_sq += l * l;
        }
    }

    double mean_loss = sum_loss / n;
    double expected_return = -mean_loss;
    double variance = sum_loss_sq / n - mean_loss * mean_loss;
    double volatility = std::sqrt(std::max(variance, 0.0));
    double sharpe = (volatility > 1e-15) ? expected_return / volatility : 0.0;

    double downside_var = sum_downside_sq / n;
    double downside_dev = std::sqrt(std::max(downside_var, 0.0));
    double sortino = (downside_dev > 1e-15) ? expected_return / downside_dev
                                            : 0.0;

    RiskResult result;
    result.var = var;
    result.cvar = cvar;
    result.expected_return = expected_return;
    result.volatility = volatility;
    result.sharpe_ratio = sharpe;
    result.sortino_ratio = sortino;
    result.confidence_level = alpha;
    result.n_scenarios = n;

    return result;
}

// ── Convenience functions ───────────────────────────────────────────

RiskResult compute_portfolio_risk_gpu(const ScenarioMatrix& scenarios,
                                      const VectorXs& weights,
                                      const RiskConfig& config) {
    auto d_losses = compute_portfolio_loss_gpu(scenarios, weights, config);
    return compute_risk_gpu(d_losses, config);
}

RiskResult compute_portfolio_risk_cpu(const MatrixXd& scenarios_host,
                                      const VectorXd& weights,
                                      const RiskConfig& config) {
    VectorXd losses = compute_portfolio_loss_cpu(scenarios_host, weights);
    return compute_risk_cpu(losses, config);
}

}  // namespace cpo
