#pragma once

/// @file efficient_frontier.h
/// @brief Efficient frontier computation via target return sweep.
///
/// Traces the Mean-CVaR efficient frontier by solving a sequence of
/// optimization problems with increasing target returns:
///
///   For each mu_target in [mu_min, mu_max]:
///     solve min CVaR_alpha(w) s.t. mu'w >= mu_target, w in C
///
/// The resulting frontier maps expected return to minimum CVaR risk.
///
/// Reference:
///   Markowitz, "Portfolio Selection", Journal of Finance, 1952 —
///   original efficient frontier concept (mean-variance).
///   Extended here to mean-CVaR following Rockafellar & Uryasev 2000.

#include <vector>

#include "core/types.h"
#include "optimizer/admm_solver.h"
#include "simulation/scenario_matrix.h"

namespace cpo {

/// A single point on the efficient frontier.
struct FrontierPoint {
    ScalarCPU target_return = 0.0;   ///< Target return for this solve.
    ScalarCPU achieved_return = 0.0; ///< Actual mu'w at optimum.
    ScalarCPU cvar = 0.0;            ///< CVaR at optimum.
    ScalarCPU zeta = 0.0;            ///< VaR at optimum.
    VectorXd weights;                ///< Optimal portfolio weights.
    int iterations = 0;              ///< ADMM iterations for this point.
    bool converged = false;          ///< Whether ADMM converged.
};

/// Configuration for efficient frontier computation.
struct FrontierConfig {
    int n_points = 20;              ///< Number of points on the frontier.
    ScalarCPU mu_min = -1.0;        ///< Minimum target return (auto if < -0.99).
    ScalarCPU mu_max = -1.0;        ///< Maximum target return (auto if < -0.99).
    bool warm_start = true;         ///< Use previous solution as initial guess.
    AdmmConfig admm_config;         ///< ADMM solver configuration.
};

/// Compute the Mean-CVaR efficient frontier.
///
/// Sweeps target returns from mu_min to mu_max and solves the
/// constrained CVaR minimization at each point.
///
/// If mu_min/mu_max are not specified (< -0.99), they are automatically
/// determined from the asset expected returns: mu_min = min(mu),
/// mu_max = max(mu).
///
/// When warm_start is enabled, each solve uses the previous solution
/// as the initial guess, improving convergence along the frontier.
///
/// @param scenarios Return matrix (n_scenarios x n_assets, double).
/// @param mu Expected return vector (n_assets, double).
/// @param config Frontier configuration.
/// @return Vector of FrontierPoint, one per target return.
std::vector<FrontierPoint> compute_efficient_frontier(
    const MatrixXd& scenarios,
    const VectorXd& mu,
    const FrontierConfig& config);

/// Compute the Mean-CVaR efficient frontier with GPU-accelerated ADMM.
///
/// Same algorithm as the CPU version, but calls the GPU admm_solve overload
/// for each target return point.
///
/// @param scenarios_gpu GPU-resident scenario matrix (float, column-major).
/// @param mu Expected return vector (n_assets, double).
/// @param config Frontier configuration.
/// @return Vector of FrontierPoint, one per target return.
std::vector<FrontierPoint> compute_efficient_frontier(
    const ScenarioMatrix& scenarios_gpu,
    const VectorXd& mu,
    const FrontierConfig& config);

}  // namespace cpo
