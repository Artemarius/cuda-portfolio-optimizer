#pragma once

/// @file portfolio_loss.h
/// @brief GPU and CPU portfolio loss computation from scenario matrices.
///
/// Loss is defined as the negative of portfolio return:
///   loss_i = -w' * r_i  for each scenario i
///
/// This convention means positive loss = negative return, and CVaR
/// measures the average of the worst losses.

#include "core/types.h"
#include "risk/device_vector.h"
#include "risk/risk_result.h"
#include "simulation/scenario_matrix.h"

namespace cpo {

/// Compute portfolio loss for each scenario on the GPU.
///
/// loss_i = -sum_j(w_j * r_{i,j}) for i = 0..n_scenarios-1.
///
/// The kernel loads weights into shared memory (2KB for 500 assets,
/// well within the 48KB limit) and uses coalesced column-major reads
/// for the scenario matrix.
///
/// @param scenarios GPU-resident scenario matrix (n_scenarios x n_assets).
/// @param weights Portfolio weight vector (n_assets, float).
/// @param config Risk configuration (threads_per_block).
/// @return DeviceVector of losses (n_scenarios, float, GPU-resident).
DeviceVector<Scalar> compute_portfolio_loss_gpu(
    const ScenarioMatrix& scenarios, const VectorXs& weights,
    const RiskConfig& config = RiskConfig{});

/// Compute portfolio loss for each scenario on the CPU (double precision).
///
/// Reference implementation for validation.
/// loss = -(scenarios_host * weights) via Eigen matrix-vector multiply.
///
/// @param scenarios_host Scenario matrix (n_scenarios x n_assets, double).
/// @param weights Portfolio weight vector (n_assets, double).
/// @return VectorXd of losses (n_scenarios, double).
VectorXd compute_portfolio_loss_cpu(const MatrixXd& scenarios_host,
                                    const VectorXd& weights);

}  // namespace cpo
