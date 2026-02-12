#pragma once

/// @file component_cvar.h
/// @brief Per-asset CVaR decomposition (component CVaR).
///
/// Component CVaR for asset j:
///   CVaR_j = w_j * (1/n_tail) * sum_{i in tail} (-r_{i,j})
///
/// where tail = {scenarios with portfolio loss >= VaR}.
///
/// Property: sum_j(CVaR_j) = total CVaR.
///
/// This is a two-pass approach:
///   1. Compute portfolio losses and VaR via existing risk pipeline.
///   2. Second pass: for each tail scenario, accumulate per-asset contributions.
///
/// References:
///   Tasche, "Risk Contributions and Performance Measurement", 2000.
///   McNeil, Frey & Embrechts, "Quantitative Risk Management", Ch. 6.

#include <utility>

#include "core/types.h"
#include "risk/device_vector.h"
#include "risk/risk_result.h"
#include "simulation/scenario_matrix.h"

namespace cpo {

/// Compute per-asset CVaR contributions on the GPU.
///
/// Uses a two-pass threshold approach:
///   - d_losses and var are already computed by compute_risk_gpu.
///   - A kernel checks each scenario's loss against var, accumulating
///     per-asset weighted returns via atomicAdd for tail scenarios.
///
/// @param scenarios GPU-resident scenario matrix (n_scenarios x n_assets).
/// @param weights Portfolio weight vector (n_assets, float).
/// @param d_losses GPU-resident portfolio loss vector (n_scenarios, float).
/// @param var VaR threshold (from compute_risk_gpu).
/// @param config Risk configuration (confidence_level, threads_per_block).
/// @return VectorXd of per-asset CVaR contributions (n_assets, double).
VectorXd compute_component_cvar_gpu(const ScenarioMatrix& scenarios,
                                     const VectorXs& weights,
                                     const DeviceVector<Scalar>& d_losses,
                                     ScalarCPU var,
                                     const RiskConfig& config = RiskConfig{});

/// Compute per-asset CVaR contributions on the CPU.
///
/// @param scenarios_host Scenario matrix (n_scenarios x n_assets, double).
/// @param weights Portfolio weight vector (n_assets, double).
/// @param losses Portfolio loss vector (n_scenarios, double).
/// @param var VaR threshold (from compute_risk_cpu).
/// @param config Risk configuration (confidence_level).
/// @return VectorXd of per-asset CVaR contributions (n_assets, double).
VectorXd compute_component_cvar_cpu(const MatrixXd& scenarios_host,
                                     const VectorXd& weights,
                                     const VectorXd& losses,
                                     ScalarCPU var,
                                     const RiskConfig& config = RiskConfig{});

/// Convenience: compute full risk metrics + component CVaR from GPU scenarios.
///
/// Chains compute_portfolio_loss_gpu -> compute_risk_gpu -> compute_component_cvar_gpu.
///
/// @param scenarios GPU-resident scenario matrix.
/// @param weights Portfolio weight vector (float).
/// @param config Risk configuration.
/// @return Pair of (RiskResult, component CVaR vector).
std::pair<RiskResult, VectorXd> compute_portfolio_risk_decomp_gpu(
    const ScenarioMatrix& scenarios,
    const VectorXs& weights,
    const RiskConfig& config = RiskConfig{});

/// Convenience: compute full risk metrics + component CVaR from CPU scenarios.
///
/// @param scenarios_host Scenario matrix (double).
/// @param weights Portfolio weight vector (double).
/// @param config Risk configuration.
/// @return Pair of (RiskResult, component CVaR vector).
std::pair<RiskResult, VectorXd> compute_portfolio_risk_decomp_cpu(
    const MatrixXd& scenarios_host,
    const VectorXd& weights,
    const RiskConfig& config = RiskConfig{});

}  // namespace cpo
