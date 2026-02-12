#include "backtest/strategy.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <random>
#include <stdexcept>

#include <Eigen/Cholesky>
#include <spdlog/spdlog.h>

#include "models/factor_monte_carlo.h"
#include "simulation/cholesky_utils.h"

namespace cpo {

// ════════════════════════════════════════════════════════════════════
// Helpers (file-local)
// ════════════════════════════════════════════════════════════════════

namespace {

/// Compute sample mean of each column (asset).
VectorXd compute_sample_mean(const MatrixXd& returns) {
    const Index T = static_cast<Index>(returns.rows());
    return returns.colwise().mean();
}

/// Compute sample covariance matrix with optional shrinkage.
///
/// Shrinkage target: (trace(S)/n) * I  (Ledoit & Wolf 2004 simplified).
/// Shrunk covariance: (1 - d) * S + d * (trace(S)/n) * I
///
/// @param returns Return matrix (T x N).
/// @param shrinkage Shrinkage intensity in [0, 1].
/// @return Sample (or shrunk) covariance matrix (N x N).
MatrixXd compute_sample_covariance(const MatrixXd& returns,
                                    ScalarCPU shrinkage = 0.0) {
    const Index T = static_cast<Index>(returns.rows());
    const Index N = static_cast<Index>(returns.cols());

    // Center returns.
    VectorXd mu = returns.colwise().mean();
    MatrixXd centered = returns.rowwise() - mu.transpose();

    // Sample covariance: S = (1/(T-1)) * X' * X.
    MatrixXd S = (centered.transpose() * centered) / static_cast<double>(T - 1);

    // Apply shrinkage if requested.
    if (shrinkage > 0.0 && shrinkage <= 1.0) {
        ScalarCPU trace_over_n = S.trace() / static_cast<double>(N);
        S = (1.0 - shrinkage) * S
            + shrinkage * trace_over_n * MatrixXd::Identity(N, N);
    }

    return S;
}

/// Clamp negative weights to zero and renormalize to sum to 1.
/// Long-only approximation for unconstrained solutions.
void clamp_and_renormalize(VectorXd& w) {
    for (Index i = 0; i < static_cast<Index>(w.size()); ++i) {
        if (w(i) < 0.0) w(i) = 0.0;
    }
    ScalarCPU s = w.sum();
    if (s > 0.0) {
        w /= s;
    }
}

}  // anonymous namespace

// ════════════════════════════════════════════════════════════════════
// EqualWeightStrategy
// ════════════════════════════════════════════════════════════════════

AllocationResult EqualWeightStrategy::allocate(const MatrixXd& returns,
                                                const VectorXd& /*w_prev*/) {
    const Index n = static_cast<Index>(returns.cols());
    VectorXd w = VectorXd::Constant(n, 1.0 / static_cast<double>(n));

    // Compute expected return from sample mean.
    VectorXd mu = compute_sample_mean(returns);
    ScalarCPU exp_ret = mu.dot(w);

    // Compute portfolio volatility.
    MatrixXd cov = compute_sample_covariance(returns);
    ScalarCPU vol = std::sqrt(w.dot(cov * w));

    return AllocationResult{std::move(w), exp_ret, vol, true};
}

// ════════════════════════════════════════════════════════════════════
// RiskParityStrategy
// ════════════════════════════════════════════════════════════════════

AllocationResult RiskParityStrategy::allocate(const MatrixXd& returns,
                                              const VectorXd& /*w_prev*/) {
    const Index n = static_cast<Index>(returns.cols());

    // Compute per-asset standard deviation from the window.
    VectorXd mu = compute_sample_mean(returns);
    MatrixXd centered = returns.rowwise() - mu.transpose();
    const Index T = static_cast<Index>(returns.rows());

    VectorXd w(n);
    ScalarCPU inv_sum = 0.0;
    for (Index i = 0; i < n; ++i) {
        ScalarCPU var_i = centered.col(i).squaredNorm()
                          / static_cast<double>(T - 1);
        ScalarCPU sigma_i = std::sqrt(var_i);
        // Guard against zero vol: assign large inverse weight.
        ScalarCPU inv_vol = (sigma_i > 1e-12) ? (1.0 / sigma_i) : 1e8;
        w(i) = inv_vol;
        inv_sum += inv_vol;
    }
    w /= inv_sum;

    // Compute expected return and portfolio vol.
    ScalarCPU exp_ret = mu.dot(w);
    MatrixXd cov = compute_sample_covariance(returns);
    ScalarCPU vol = std::sqrt(w.dot(cov * w));

    return AllocationResult{std::move(w), exp_ret, vol, true};
}

// ════════════════════════════════════════════════════════════════════
// MeanVarianceStrategy
// ════════════════════════════════════════════════════════════════════

MeanVarianceStrategy::MeanVarianceStrategy(const MeanVarianceConfig& config)
    : config_(config) {}

AllocationResult MeanVarianceStrategy::allocate(const MatrixXd& returns,
                                                 const VectorXd& /*w_prev*/) {
    const Index n = static_cast<Index>(returns.cols());

    VectorXd mu = compute_sample_mean(returns);
    MatrixXd cov = compute_sample_covariance(returns, config_.shrinkage_intensity);

    // LDLT decomposition for solving linear systems.
    Eigen::LDLT<MatrixXd> ldlt(cov);
    if (ldlt.info() != Eigen::Success) {
        spdlog::warn("MeanVarianceStrategy: LDLT decomposition failed");
        // Fallback to equal weight.
        VectorXd w = VectorXd::Constant(n, 1.0 / static_cast<double>(n));
        return AllocationResult{std::move(w), mu.dot(w), 0.0, false};
    }

    VectorXd ones = VectorXd::Ones(n);
    VectorXd w;

    if (!config_.has_target_return) {
        // Global minimum variance: w = Sigma^{-1} * 1 / (1' * Sigma^{-1} * 1).
        VectorXd Sinv_1 = ldlt.solve(ones);
        ScalarCPU denom = ones.dot(Sinv_1);
        if (std::abs(denom) < 1e-14) {
            VectorXd wf = VectorXd::Constant(n, 1.0 / static_cast<double>(n));
            return AllocationResult{std::move(wf), mu.dot(wf), 0.0, false};
        }
        w = Sinv_1 / denom;
    } else {
        // Merton 1972: two-fund separation.
        // w = a * Sigma^{-1} * 1  +  b * Sigma^{-1} * mu
        // where a and b solve the linear system from the two constraints:
        //   1' w = 1  and  mu' w = target.
        VectorXd Sinv_1 = ldlt.solve(ones);
        VectorXd Sinv_mu = ldlt.solve(mu);

        ScalarCPU A = ones.dot(Sinv_1);   // 1' Sigma^{-1} 1
        ScalarCPU B = ones.dot(Sinv_mu);  // 1' Sigma^{-1} mu
        ScalarCPU C = mu.dot(Sinv_mu);    // mu' Sigma^{-1} mu
        ScalarCPU D = A * C - B * B;

        if (std::abs(D) < 1e-14) {
            // Degenerate: fall back to min-variance.
            w = Sinv_1 / A;
        } else {
            ScalarCPU target = config_.target_return;
            ScalarCPU lambda1 = (C - B * target) / D;
            ScalarCPU lambda2 = (A * target - B) / D;
            w = lambda1 * Sinv_1 + lambda2 * Sinv_mu;
        }
    }

    // Long-only approximation: clamp negatives, renormalize.
    clamp_and_renormalize(w);

    ScalarCPU exp_ret = mu.dot(w);
    ScalarCPU vol = std::sqrt(w.dot(cov * w));

    return AllocationResult{std::move(w), exp_ret, vol, true};
}

// ════════════════════════════════════════════════════════════════════
// MeanCVaRStrategy
// ════════════════════════════════════════════════════════════════════

MeanCVaRStrategy::MeanCVaRStrategy(const MeanCVaRConfig& config)
    : config_(config) {}

AllocationResult MeanCVaRStrategy::allocate(const MatrixXd& returns,
                                             const VectorXd& w_prev) {
    const Index n = static_cast<Index>(returns.cols());

    // 1. Estimate mu and Sigma from the return window.
    VectorXd mu;
    MatrixXd cov;
    std::optional<FactorModelResult> factor_result;

    if (config_.use_factor_model) {
        // Factor model covariance estimation.
        try {
            factor_result = fit_factor_model(returns, config_.factor_config);
        } catch (const std::exception& e) {
            spdlog::warn("MeanCVaRStrategy: factor model failed ({}), "
                         "falling back to sample covariance", e.what());
        }
    }

    if (factor_result.has_value()) {
        mu = factor_result->mu;
        cov = reconstruct_covariance(*factor_result);
    } else {
        mu = compute_sample_mean(returns);
        cov = compute_sample_covariance(returns);
    }

    // 2. Generate scenarios.
    MonteCarloConfig mc = config_.mc_config;
    mc.n_assets = n;

    // 3. Configure ADMM.
    AdmmConfig admm = config_.admm_config;
    if (admm.constraints.has_turnover && w_prev.size() == n) {
        admm.constraints.turnover.w_prev = w_prev;
    }

    // 4. Generate scenarios and solve.
    //    GPU path: keep ScenarioMatrix alive and use GPU admm_solve.
    //    CPU path: generate MatrixXd and use CPU admm_solve.
    AdmmResult result;
    if (config_.use_gpu) {
        // GPU path: generate scenarios on GPU, solve with GPU ADMM.
        std::optional<ScenarioMatrix> scenarios_gpu;
        if (config_.use_factor_mc && factor_result.has_value()) {
            scenarios_gpu.emplace(generate_scenarios_factor_gpu(
                mu, *factor_result, mc, config_.curand_states));
        } else {
            CholeskyResult cholesky;
            try {
                cholesky = compute_cholesky(cov);
            } catch (const std::runtime_error& e) {
                spdlog::warn("MeanCVaRStrategy: Cholesky failed ({}), "
                             "falling back to equal weight", e.what());
                VectorXd w = VectorXd::Constant(n, 1.0 / static_cast<double>(n));
                return AllocationResult{std::move(w), mu.dot(w), 0.0, false};
            }
            scenarios_gpu.emplace(generate_scenarios_gpu(
                mu, cholesky, mc, config_.curand_states));
        }
        result = admm_solve(*scenarios_gpu, mu, admm, w_prev);
    } else {
        // CPU path: generate scenarios on CPU, solve with CPU ADMM.
        MatrixXd scenarios_cpu;
        if (config_.use_factor_mc && factor_result.has_value()) {
            scenarios_cpu = generate_scenarios_factor_cpu(
                mu, *factor_result, mc);
        } else {
            CholeskyResult cholesky;
            try {
                cholesky = compute_cholesky(cov);
            } catch (const std::runtime_error& e) {
                spdlog::warn("MeanCVaRStrategy: Cholesky failed ({}), "
                             "falling back to equal weight", e.what());
                VectorXd w = VectorXd::Constant(n, 1.0 / static_cast<double>(n));
                return AllocationResult{std::move(w), mu.dot(w), 0.0, false};
            }
            scenarios_cpu = generate_scenarios_cpu(mu, cholesky, mc);
        }
        result = admm_solve(scenarios_cpu, mu, admm, w_prev);
    }

    if (!result.converged) {
        spdlog::warn("MeanCVaRStrategy: ADMM did not converge after {} iterations",
                     result.iterations);
    }

    return AllocationResult{
        std::move(result.weights),
        result.expected_return,
        result.cvar,
        result.converged
    };
}

// ════════════════════════════════════════════════════════════════════
// Factory
// ════════════════════════════════════════════════════════════════════

std::unique_ptr<Strategy> create_strategy(const std::string& name) {
    if (name == "EqualWeight") {
        return std::make_unique<EqualWeightStrategy>();
    } else if (name == "RiskParity") {
        return std::make_unique<RiskParityStrategy>();
    } else if (name == "MeanVariance") {
        return std::make_unique<MeanVarianceStrategy>();
    } else if (name == "MeanCVaR") {
        MeanCVaRConfig cfg;
        cfg.mc_config.n_scenarios = 10000;
        cfg.use_gpu = false;
        return std::make_unique<MeanCVaRStrategy>(cfg);
    }
    throw std::runtime_error("Unknown strategy: " + name);
}

}  // namespace cpo
