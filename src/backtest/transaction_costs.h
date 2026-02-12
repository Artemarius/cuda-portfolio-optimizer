#pragma once

/// @file transaction_costs.h
/// @brief Transaction cost model for backtesting.
///
/// Proportional cost model: cost = rate * turnover * portfolio_value.
/// Supports a minimum trade threshold to suppress small rebalance trades,
/// reducing unnecessary transaction costs in practice.

#include "core/types.h"

namespace cpo {

/// Configuration for the proportional transaction cost model.
struct TransactionCostConfig {
    ScalarCPU cost_rate = 0.001;           ///< Cost per unit turnover (10 bps default).
    ScalarCPU min_trade_threshold = 0.0;   ///< Min |delta_w| to trigger trade per asset.
};

/// Result of applying the transaction cost model.
struct TransactionCostResult {
    ScalarCPU total_cost;        ///< Dollar cost of the rebalance.
    ScalarCPU cost_as_fraction;  ///< Cost / portfolio_value.
    ScalarCPU turnover;          ///< Sum of |w_effective - w_old| (L1 turnover).
    VectorXd effective_weights;  ///< Weights after threshold filtering + renormalization.
};

/// Compute transaction costs for a portfolio rebalance.
///
/// Trades with |w_new[i] - w_old[i]| < min_trade_threshold are suppressed
/// (the asset keeps w_old[i]). After suppression, weights are renormalized
/// to sum to 1. Turnover is computed on the effective (post-threshold) weights.
///
/// @param w_new          Target weights from the optimizer.
/// @param w_old          Current (pre-rebalance) weights.
/// @param portfolio_value Current portfolio dollar value (before costs).
/// @param config         Transaction cost parameters.
/// @return TransactionCostResult with cost, turnover, and effective weights.
TransactionCostResult compute_transaction_costs(
    const VectorXd& w_new, const VectorXd& w_old,
    ScalarCPU portfolio_value, const TransactionCostConfig& config);

}  // namespace cpo
