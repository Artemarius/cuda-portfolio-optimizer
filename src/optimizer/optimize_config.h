#pragma once

/// @file optimize_config.h
/// @brief Configuration for the optimize CLI.

#include <string>
#include <vector>

#include "core/types.h"
#include "optimizer/admm_solver.h"
#include "optimizer/efficient_frontier.h"
#include "simulation/monte_carlo.h"

namespace cpo {

/// Configuration for a portfolio optimization run.
struct OptimizeConfig {
    // Data source: either provide mu/covariance directly or load from CSV.
    std::string price_csv_path;              ///< Path to wide-format price CSV (optional).
    std::vector<std::string> tickers;        ///< Tickers to include (empty = all).
    std::string start_date;                  ///< Filter start date (optional).
    std::string end_date;                    ///< Filter end date (optional).

    // Direct mu/covariance specification (used when no CSV provided).
    std::vector<double> mu_values;           ///< Expected returns vector.
    std::vector<std::vector<double>> cov_values; ///< Covariance matrix (row-major).

    // Mode.
    bool frontier_mode = false;              ///< Compute efficient frontier vs single solve.

    // Monte Carlo settings.
    MonteCarloConfig mc_config;
    bool use_gpu = false;

    // ADMM settings.
    AdmmConfig admm_config;

    // Frontier settings.
    FrontierConfig frontier_config;

    // Output.
    std::string output_dir;
    bool verbose = false;
};

/// Load an OptimizeConfig from a JSON file.
///
/// @param json_path Path to the JSON configuration file.
/// @return Parsed OptimizeConfig.
/// @throws std::runtime_error if the file cannot be read or is invalid.
OptimizeConfig load_optimize_config(const std::string& json_path);

}  // namespace cpo
