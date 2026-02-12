#include "backtest/transaction_costs.h"

#include <cmath>
#include <stdexcept>

namespace cpo {

TransactionCostResult compute_transaction_costs(
    const VectorXd& w_new, const VectorXd& w_old,
    ScalarCPU portfolio_value, const TransactionCostConfig& config) {
    const Index n = static_cast<Index>(w_new.size());
    if (w_old.size() != n) {
        throw std::runtime_error(
            "compute_transaction_costs: w_new size (" +
            std::to_string(n) + ") != w_old size (" +
            std::to_string(w_old.size()) + ")");
    }

    // Apply minimum trade threshold: suppress small trades.
    VectorXd w_eff(n);
    for (Index i = 0; i < n; ++i) {
        if (std::abs(w_new(i) - w_old(i)) < config.min_trade_threshold) {
            w_eff(i) = w_old(i);  // Keep old weight.
        } else {
            w_eff(i) = w_new(i);  // Apply new weight.
        }
    }

    // Renormalize to sum to 1 (if sum is non-zero).
    ScalarCPU w_sum = w_eff.sum();
    if (w_sum > 0.0) {
        w_eff /= w_sum;
    }

    // Compute turnover: ||w_eff - w_old||_1.
    ScalarCPU turnover = (w_eff - w_old).lpNorm<1>();

    // Proportional cost: rate * turnover * value.
    ScalarCPU total_cost = config.cost_rate * turnover * portfolio_value;
    ScalarCPU cost_fraction = (portfolio_value > 0.0)
                                  ? total_cost / portfolio_value
                                  : 0.0;

    return TransactionCostResult{
        total_cost,
        cost_fraction,
        turnover,
        std::move(w_eff)
    };
}

}  // namespace cpo
