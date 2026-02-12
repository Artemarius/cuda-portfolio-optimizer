#include "backtest/backtest_engine.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <spdlog/spdlog.h>

namespace cpo {

BacktestSummary compute_backtest_summary(
    const std::vector<PortfolioSnapshot>& snapshots,
    const std::string& strategy_name) {
    BacktestSummary summary{};
    summary.strategy_name = strategy_name;

    const int n_days = static_cast<int>(snapshots.size());
    summary.n_days = n_days;

    if (n_days == 0) {
        return summary;
    }

    // Total return.
    ScalarCPU initial_value = snapshots.front().portfolio_value;
    ScalarCPU final_value = snapshots.back().portfolio_value;
    summary.total_return = (final_value / initial_value) - 1.0;

    // Annualized return: (1 + total_return)^(252 / n_days) - 1.
    if (n_days > 1) {
        summary.annualized_return =
            std::pow(1.0 + summary.total_return,
                     252.0 / static_cast<double>(n_days)) - 1.0;
    } else {
        summary.annualized_return = summary.total_return;
    }

    // Daily returns for vol/Sharpe/Sortino.
    // Skip the first snapshot (no prior return).
    std::vector<ScalarCPU> daily_rets;
    daily_rets.reserve(n_days);
    for (int i = 0; i < n_days; ++i) {
        daily_rets.push_back(snapshots[i].daily_return);
    }

    // Mean and std of daily returns.
    ScalarCPU mean_ret = 0.0;
    for (auto r : daily_rets) mean_ret += r;
    mean_ret /= static_cast<double>(daily_rets.size());

    ScalarCPU sum_sq = 0.0;
    ScalarCPU sum_sq_down = 0.0;
    for (auto r : daily_rets) {
        ScalarCPU diff = r - mean_ret;
        sum_sq += diff * diff;
        if (r < 0.0) {
            sum_sq_down += r * r;
        }
    }
    ScalarCPU daily_vol = std::sqrt(sum_sq / static_cast<double>(daily_rets.size()));
    ScalarCPU daily_downside = std::sqrt(sum_sq_down / static_cast<double>(daily_rets.size()));

    summary.annualized_volatility = daily_vol * std::sqrt(252.0);

    ScalarCPU ann_downside = daily_downside * std::sqrt(252.0);

    // Sharpe ratio.
    summary.sharpe_ratio = (summary.annualized_volatility > 1e-14)
                               ? summary.annualized_return / summary.annualized_volatility
                               : 0.0;

    // Sortino ratio.
    summary.sortino_ratio = (ann_downside > 1e-14)
                                ? summary.annualized_return / ann_downside
                                : 0.0;

    // Max drawdown (peak-to-trough).
    ScalarCPU peak = snapshots.front().portfolio_value;
    ScalarCPU max_dd = 0.0;
    for (const auto& snap : snapshots) {
        if (snap.portfolio_value > peak) {
            peak = snap.portfolio_value;
        }
        ScalarCPU dd = (peak - snap.portfolio_value) / peak;
        if (dd > max_dd) {
            max_dd = dd;
        }
    }
    summary.max_drawdown = max_dd;

    // Calmar ratio.
    summary.calmar_ratio = (max_dd > 1e-14)
                               ? summary.annualized_return / max_dd
                               : 0.0;

    // Transaction cost and turnover aggregates.
    ScalarCPU total_cost = 0.0;
    ScalarCPU total_turnover = 0.0;
    int n_rebalances = 0;
    for (const auto& snap : snapshots) {
        total_cost += snap.transaction_cost;
        if (snap.is_rebalance_date) {
            total_turnover += snap.turnover;
            ++n_rebalances;
        }
    }
    summary.total_transaction_cost = total_cost;
    summary.n_rebalances = n_rebalances;
    summary.avg_turnover = (n_rebalances > 0)
                               ? total_turnover / static_cast<double>(n_rebalances)
                               : 0.0;

    return summary;
}

BacktestResult run_backtest(const ReturnData& returns,
                            Strategy& strategy,
                            const BacktestConfig& config) {
    const Index T = returns.num_periods();
    const Index N = returns.num_assets();

    if (T < config.lookback_window + 1) {
        throw std::runtime_error(
            "run_backtest: insufficient data. Have " + std::to_string(T) +
            " return periods, need at least " +
            std::to_string(config.lookback_window + 1) +
            " (lookback_window + 1)");
    }

    BacktestResult result;
    result.tickers = returns.tickers;

    // Initial portfolio: equal weight.
    VectorXd w = VectorXd::Constant(N, 1.0 / static_cast<double>(N));
    ScalarCPU portfolio_value = config.initial_capital;

    // Counter for rebalance scheduling.
    int days_since_rebalance = 0;

    // Start at t = lookback_window (first day we have a full estimation window).
    const Index t_start = config.lookback_window;

    for (Index t = t_start; t < T; ++t) {
        // Daily asset returns for day t.
        VectorXd r_t = returns.returns.row(t).transpose();

        // Portfolio return: w' * r_t.
        ScalarCPU r_portfolio = w.dot(r_t);

        // Update portfolio value.
        portfolio_value *= (1.0 + r_portfolio);

        // Drift weights: w_i *= (1 + r_i) / (1 + r_portfolio).
        ScalarCPU denom = 1.0 + r_portfolio;
        if (std::abs(denom) > 1e-14) {
            for (Index i = 0; i < N; ++i) {
                w(i) *= (1.0 + r_t(i)) / denom;
            }
        }

        // Check for rebalance.
        bool is_rebalance = (days_since_rebalance >= config.rebalance_frequency)
                            || (t == t_start);  // Initial allocation.
        ScalarCPU txn_cost = 0.0;
        ScalarCPU turnover = 0.0;

        if (is_rebalance) {
            // Extract return window: [t - lookback + 1, t] inclusive.
            // This is lookback_window rows of the return matrix.
            Index window_start = t - config.lookback_window + 1;
            MatrixXd window = returns.returns.block(
                window_start, 0, config.lookback_window, N);

            // Call strategy.
            AllocationResult alloc = strategy.allocate(window, w);

            if (alloc.success) {
                // Apply transaction costs.
                TransactionCostResult tc = compute_transaction_costs(
                    alloc.weights, w, portfolio_value, config.transaction_costs);

                txn_cost = tc.total_cost;
                turnover = tc.turnover;
                portfolio_value -= txn_cost;
                w = tc.effective_weights;
            }
            // If allocation failed, keep current weights (no rebalance).

            days_since_rebalance = 0;
        }

        // Record snapshot.
        PortfolioSnapshot snap;
        snap.date = returns.dates[t];
        snap.portfolio_value = portfolio_value;
        snap.transaction_cost = txn_cost;
        snap.turnover = turnover;
        snap.daily_return = r_portfolio;
        snap.weights = w;
        snap.is_rebalance_date = is_rebalance;

        result.snapshots.push_back(std::move(snap));
        ++days_since_rebalance;
    }

    // Compute summary.
    result.summary = compute_backtest_summary(result.snapshots, strategy.name());

    return result;
}

}  // namespace cpo
