#pragma once

/// @file strategy.h
/// @brief Portfolio allocation strategy interface and implementations.
///
/// Strategies receive a sub-window of historical returns and the previous
/// portfolio weights, and produce a new allocation. The backtest engine
/// handles slicing the return data and calling strategies at each rebalance.
///
/// Implementations:
///   - EqualWeightStrategy:    w_i = 1/N
///   - RiskParityStrategy:     w_i = (1/sigma_i) / sum(1/sigma_j)
///   - MeanVarianceStrategy:   Min-variance or target-return via sample cov + LDLT
///   - MeanCVaRStrategy:       Monte Carlo + ADMM solver (full pipeline)

#include <memory>
#include <string>

#include "core/types.h"
#include "models/factor_model.h"
#include "optimizer/admm_solver.h"
#include "simulation/monte_carlo.h"

namespace cpo {

/// Result of a strategy allocation.
struct AllocationResult {
    VectorXd weights;            ///< Portfolio weights (sum = 1, non-negative).
    ScalarCPU expected_return;   ///< Estimated expected return (annualized or per-period).
    ScalarCPU risk_metric;       ///< Strategy-specific risk metric (vol, CVaR, etc.).
    bool success;                ///< Whether the allocation succeeded.
};

/// Abstract base class for portfolio allocation strategies.
class Strategy {
public:
    virtual ~Strategy() = default;

    /// Compute portfolio allocation from a return window.
    ///
    /// @param returns Sub-window of historical returns (T x N, double).
    /// @param w_prev  Previous portfolio weights (for warm-start / turnover).
    /// @return AllocationResult with new weights.
    virtual AllocationResult allocate(const MatrixXd& returns,
                                      const VectorXd& w_prev) = 0;

    /// Human-readable strategy name.
    virtual std::string name() const = 0;
};

// ── Equal Weight ─────────────────────────────────────────────────────

/// Equal-weight (1/N) strategy. Always succeeds.
class EqualWeightStrategy : public Strategy {
public:
    AllocationResult allocate(const MatrixXd& returns,
                              const VectorXd& w_prev) override;
    std::string name() const override { return "EqualWeight"; }
};

// ── Risk Parity (inverse volatility) ────────────────────────────────

/// Inverse-volatility risk parity: w_i = (1/sigma_i) / sum(1/sigma_j).
///
/// Estimates per-asset volatility from the return window. If an asset has
/// zero volatility, assigns a large inverse (1e8) to avoid division by zero.
class RiskParityStrategy : public Strategy {
public:
    AllocationResult allocate(const MatrixXd& returns,
                              const VectorXd& w_prev) override;
    std::string name() const override { return "RiskParity"; }
};

// ── Mean-Variance ───────────────────────────────────────────────────

/// Configuration for the mean-variance strategy.
struct MeanVarianceConfig {
    ScalarCPU target_return = 0.0;       ///< Target return constraint.
    bool has_target_return = false;       ///< false = global min-variance.
    ScalarCPU shrinkage_intensity = 0.0;  ///< 0 = pure sample cov; (0,1) = shrink toward identity.
};

/// Mean-variance optimization using sample covariance + Eigen LDLT.
///
/// Global min-variance: w = Sigma^{-1} * 1 / (1' * Sigma^{-1} * 1).
/// With target return: Merton 1972 closed-form 2-constraint solution.
/// Long-only approximation: clamp negatives to 0, renormalize.
///
/// Reference: Merton, "An Analytic Derivation of the Efficient Portfolio
/// Frontier", Journal of Financial and Quantitative Analysis, 1972.
class MeanVarianceStrategy : public Strategy {
public:
    explicit MeanVarianceStrategy(const MeanVarianceConfig& config = {});

    AllocationResult allocate(const MatrixXd& returns,
                              const VectorXd& w_prev) override;
    std::string name() const override { return "MeanVariance"; }

private:
    MeanVarianceConfig config_;
};

// ── Mean-CVaR (ADMM) ───────────────────────────────────────────────

/// Configuration for the Mean-CVaR strategy.
struct MeanCVaRConfig {
    AdmmConfig admm_config;            ///< ADMM solver settings.
    MonteCarloConfig mc_config;        ///< Monte Carlo scenario generation settings.
    bool use_gpu = false;              ///< GPU or CPU path for scenario generation.
    CurandStates* curand_states = nullptr;  ///< Shared cuRAND states (engine-owned).
    bool use_factor_model = false;     ///< Use factor model for covariance estimation.
    FactorModelConfig factor_config;   ///< Factor model settings.
    bool use_factor_mc = false;        ///< Use factor MC kernel (vs full Cholesky MC).
};

/// Mean-CVaR optimization: Monte Carlo scenario generation + ADMM solver.
///
/// Pipeline per rebalance:
///   1. Estimate mu, Sigma from return window
///   2. Cholesky decomposition: Sigma = L * L'
///   3. Generate scenarios: r = mu + L * z (GPU or CPU)
///   4. ADMM solve: min CVaR(w) s.t. constraints
///
/// Warm-starts from w_prev. Sets turnover constraint w_prev if enabled.
class MeanCVaRStrategy : public Strategy {
public:
    explicit MeanCVaRStrategy(const MeanCVaRConfig& config);

    AllocationResult allocate(const MatrixXd& returns,
                              const VectorXd& w_prev) override;
    std::string name() const override { return "MeanCVaR"; }

private:
    MeanCVaRConfig config_;
};

/// Factory: create a strategy by name.
///
/// Supported names: "EqualWeight", "RiskParity", "MeanVariance", "MeanCVaR".
/// MeanVariance and MeanCVaR use default configs from the factory.
/// For custom configs, construct the strategy directly.
std::unique_ptr<Strategy> create_strategy(const std::string& name);

}  // namespace cpo
