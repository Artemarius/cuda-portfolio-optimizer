#pragma once

/// @file backtest_engine.h
/// @brief Rolling-window backtesting engine.
///
/// Iterates over historical return data, calling a Strategy at each
/// rebalance point. Tracks portfolio value, weight drift, and
/// transaction costs. Computes summary performance metrics.

#include <string>
#include <vector>

#include "backtest/backtest_config.h"
#include "backtest/strategy.h"
#include "backtest/transaction_costs.h"
#include "core/types.h"
#include "data/market_data.h"

namespace cpo {

/// Snapshot of portfolio state at a single point in time.
struct PortfolioSnapshot {
    std::string date;                ///< ISO 8601 date.
    ScalarCPU portfolio_value;       ///< Portfolio dollar value.
    ScalarCPU transaction_cost;      ///< Cost incurred this day (0 if no rebalance).
    ScalarCPU turnover;              ///< Turnover this day (0 if no rebalance).
    ScalarCPU daily_return;          ///< Portfolio return for this day.
    VectorXd weights;                ///< Portfolio weights (post-rebalance or post-drift).
    bool is_rebalance_date;          ///< True if rebalance occurred on this day.
};

/// Aggregate performance summary for a backtest.
struct BacktestSummary {
    std::string strategy_name;
    ScalarCPU total_return;           ///< (final_value / initial_value) - 1.
    ScalarCPU annualized_return;      ///< (1 + total_return)^(252/n_days) - 1.
    ScalarCPU annualized_volatility;  ///< daily_std * sqrt(252).
    ScalarCPU sharpe_ratio;           ///< annualized_return / annualized_vol.
    ScalarCPU sortino_ratio;          ///< annualized_return / annualized_downside_dev.
    ScalarCPU max_drawdown;           ///< Peak-to-trough maximum drawdown.
    ScalarCPU calmar_ratio;           ///< annualized_return / max_drawdown.
    ScalarCPU total_transaction_cost; ///< Sum of all transaction costs.
    ScalarCPU avg_turnover;           ///< Average turnover per rebalance.
    int n_rebalances;                 ///< Number of rebalance events.
    int n_days;                       ///< Total trading days in the backtest.
};

/// Complete backtest result: time series + summary.
struct BacktestResult {
    std::vector<PortfolioSnapshot> snapshots;
    BacktestSummary summary;
    std::vector<std::string> tickers;
};

/// Run a rolling-window backtest.
///
/// Rolling-window loop:
///   1. Start at t = lookback_window, equal-weight initial portfolio.
///   2. Each day: compute daily portfolio return, update value, drift weights.
///   3. Every rebalance_frequency days: call strategy.allocate() on the
///      return window [t-lookback+1 : t], apply transaction costs.
///   4. After loop: compute summary metrics.
///
/// Weight drift between rebalances:
///   w_i_new = w_i * (1 + r_i) / (1 + r_portfolio)
///
/// @param returns   Historical return data (full dataset).
/// @param strategy  Allocation strategy to use.
/// @param config    Backtest configuration (lookback, rebalance freq, costs, etc.).
/// @return BacktestResult with snapshots and summary.
/// @throws std::runtime_error if insufficient data for the lookback window.
BacktestResult run_backtest(const ReturnData& returns,
                            Strategy& strategy,
                            const BacktestConfig& config);

/// Compute aggregate summary metrics from a snapshot time series.
///
/// @param snapshots Portfolio snapshots from run_backtest.
/// @param strategy_name Name of the strategy (for labeling).
/// @return BacktestSummary with all performance metrics.
BacktestSummary compute_backtest_summary(
    const std::vector<PortfolioSnapshot>& snapshots,
    const std::string& strategy_name);

}  // namespace cpo
