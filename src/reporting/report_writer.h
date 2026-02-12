#pragma once

/// @file report_writer.h
/// @brief CSV and JSON output for backtest results.

#include <string>
#include <vector>

#include "backtest/backtest_engine.h"

namespace cpo {

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

}  // namespace cpo
