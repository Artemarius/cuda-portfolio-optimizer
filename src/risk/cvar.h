#pragma once

/// @file cvar.h
/// @brief GPU and CPU CVaR/VaR computation from loss distributions.
///
/// GPU path: CUB DeviceRadixSort + DeviceReduce for VaR/CVaR,
///   custom reduction kernel for loss statistics (volatility, Sharpe, Sortino).
///
/// CPU path: std::sort + loop in double precision.
///
/// The sort-based approach is natural for risk measurement and gives VaR
/// as a byproduct. The Rockafellar-Uryasev LP formulation will be added
/// in Phase 6 for the optimizer.
///
/// References:
///   Rockafellar & Uryasev, "Optimization of Conditional Value-at-Risk",
///   J. Risk 2000 — CVaR definition and properties.

#include "core/types.h"
#include "risk/device_vector.h"
#include "risk/risk_result.h"
#include "simulation/scenario_matrix.h"

namespace cpo {

/// Compute VaR, CVaR, and statistics from a GPU-resident loss vector.
///
/// Algorithm:
///   1. Sort losses ascending (CUB RadixSort into separate buffer).
///   2. VaR = sorted[floor(alpha * N)].
///   3. CVaR = mean of sorted[floor(alpha * N) .. N-1].
///   4. Single-pass reduction for mean, variance, downside deviation.
///
/// The input loss vector is NOT modified (sorting writes to a copy).
///
/// @param d_losses GPU-resident loss vector (n_scenarios, float).
/// @param config Risk configuration (confidence_level, threads_per_block).
/// @return RiskResult with all metrics in double precision.
RiskResult compute_risk_gpu(const DeviceVector<Scalar>& d_losses,
                            const RiskConfig& config = RiskConfig{});

/// Compute VaR, CVaR, and statistics from a CPU loss vector (double).
///
/// Reference implementation: std::sort + loop.
///
/// @param losses CPU loss vector (n_scenarios, double).
/// @param config Risk configuration (confidence_level).
/// @return RiskResult with all metrics.
RiskResult compute_risk_cpu(const VectorXd& losses,
                            const RiskConfig& config = RiskConfig{});

/// Convenience: compute full risk metrics from GPU scenarios + weights.
///
/// Equivalent to compute_risk_gpu(compute_portfolio_loss_gpu(scenarios, weights)).
///
/// @param scenarios GPU-resident scenario matrix.
/// @param weights Portfolio weight vector (float).
/// @param config Risk configuration.
/// @return RiskResult with all metrics.
RiskResult compute_portfolio_risk_gpu(const ScenarioMatrix& scenarios,
                                      const VectorXs& weights,
                                      const RiskConfig& config = RiskConfig{});

/// Convenience: compute full risk metrics from CPU scenarios + weights.
///
/// @param scenarios_host Scenario matrix (double).
/// @param weights Portfolio weight vector (double).
/// @param config Risk configuration.
/// @return RiskResult with all metrics.
RiskResult compute_portfolio_risk_cpu(const MatrixXd& scenarios_host,
                                      const VectorXd& weights,
                                      const RiskConfig& config = RiskConfig{});

}  // namespace cpo
