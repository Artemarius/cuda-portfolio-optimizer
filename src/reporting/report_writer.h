#pragma once

/// @file report_writer.h
/// @brief CSV and JSON output for backtest and optimization results.

#include <string>
#include <vector>

#include "backtest/backtest_engine.h"
#include "optimizer/admm_solver.h"
#include "optimizer/efficient_frontier.h"

namespace cpo {

// ── Backtest reporting ────────────────────────────────────────────

/// Write the equity curve (date, value, return) to CSV.
void write_equity_curve_csv(const BacktestResult& result,
                            const std::string& path);

/// Write portfolio weights at rebalance dates to CSV.
void write_weights_csv(const BacktestResult& result,
                       const std::string& path);

/// Write a single backtest summary to JSON.
void write_summary_json(const BacktestSummary& summary,
                        const std::string& path);

/// Write a comparison of multiple backtest results to JSON.
void write_comparison_json(const std::vector<BacktestResult>& results,
                           const std::string& path);

/// Write a comparison summary table to CSV.
void write_comparison_csv(const std::vector<BacktestResult>& results,
                          const std::string& path);

// ── Optimization reporting ────────────────────────────────────────

/// Write efficient frontier points to CSV.
///
/// Columns: target_return, achieved_return, cvar, zeta, converged,
/// iterations, w_0, w_1, ...
void write_frontier_csv(const std::vector<FrontierPoint>& frontier,
                        const std::vector<std::string>& tickers,
                        const std::string& path);

/// Write a single optimization result to JSON.
void write_optimize_result_json(const AdmmResult& result,
                                const VectorXd& mu,
                                const std::vector<std::string>& tickers,
                                const std::string& path);

}  // namespace cpo
