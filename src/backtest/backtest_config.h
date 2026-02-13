#pragma once

/// @file backtest_config.h
/// @brief Configuration for rolling-window backtesting.

#include <string>
#include <vector>

#include "backtest/transaction_costs.h"
#include "core/types.h"
#include "data/market_data.h"
#include "models/factor_model.h"
#include "optimizer/admm_solver.h"
#include "simulation/monte_carlo.h"

namespace cpo {

/// Configuration for a backtest run.
struct BacktestConfig {
    // Data settings.
    std::string price_csv_path;              ///< Path to wide-format price CSV.
    std::vector<std::string> tickers;        ///< Tickers to include (empty = all).
    std::string start_date;                  ///< Filter start date (inclusive, ISO 8601).
    std::string end_date;                    ///< Filter end date (inclusive, ISO 8601).
    ReturnType return_type = ReturnType::kSimple;

    // Rolling window.
    Index lookback_window = 252;             ///< Estimation window in trading days.
    Index rebalance_frequency = 21;          ///< Rebalance every N days.

    // Strategy.
    std::string strategy_name = "MeanCVaR"; ///< "EqualWeight","RiskParity","MeanVariance","MeanCVaR","all"
    ScalarCPU initial_capital = 1000000.0;

    // ADMM / Monte Carlo settings (used by MeanCVaR strategy).
    AdmmConfig admm_config;
    MonteCarloConfig mc_config;
    bool use_gpu = false;

    // Factor model settings (used by MeanCVaR strategy).
    bool use_factor_model = false;          ///< Use factor model for covariance estimation.
    FactorModelConfig factor_config;        ///< Factor model parameters.
    bool use_factor_mc = false;             ///< Use factor MC kernel (vs full Cholesky MC).

    // Mean-variance settings.
    ScalarCPU shrinkage_intensity = 0.0;
    ScalarCPU mv_target_return = 0.0;
    bool mv_has_target_return = false;

    // Ledoit-Wolf optimal shrinkage (applies to MeanVariance and MeanCVaR sample cov path).
    bool use_ledoit_wolf = false;

    // Transaction costs.
    TransactionCostConfig transaction_costs;

    // Output.
    std::string output_dir;
    bool verbose = false;
};

/// Load a BacktestConfig from a JSON file.
///
/// @param json_path Path to the JSON configuration file.
/// @return Parsed BacktestConfig.
/// @throws std::runtime_error if the file cannot be read or required fields are missing.
BacktestConfig load_backtest_config(const std::string& json_path);

}  // namespace cpo
