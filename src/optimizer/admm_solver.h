#pragma once

/// @file admm_solver.h
/// @brief ADMM solver for Mean-CVaR portfolio optimization.
///
/// Solves the Rockafellar-Uryasev formulation:
///   min_{w}  CVaR_alpha(w)
///   s.t.     mu' w >= target_return
///            1' w = 1,  w >= 0        (simplex)
///            w_min <= w <= w_max       (position limits, optional)
///            ||w - w_prev||_1 <= tau   (turnover, optional)
///            sector bounds             (sector constraints, optional)
///
/// ADMM splits this into:
///   x-update: minimize CVaR proxy + (rho/2)||x - z + u||^2
///   z-update: project onto constraint set
///   u-update: dual variable update
///
/// References:
///   Boyd et al., "Distributed Optimization and Statistical Learning
///   via the Alternating Direction Method of Multipliers", 2011.
///   - General ADMM framework: Eq. (3.1)-(3.3)
///   - Adaptive rho: Section 3.4.1, Eq. (3.13)
///   - Convergence criteria: Section 3.3, Eq. (3.11)-(3.12)
///   - Over-relaxation: Section 3.4.3, Eq. (3.19)-(3.20)
///
///   Wohlberg, "ADMM Penalty Parameter Selection by Residual Balancing",
///   2017 — continuous adaptive rho via normalized residual balancing.
///
///   Zhang, O'Donoghue, Boyd, "Globally Convergent Type-I Anderson
///   Acceleration for Non-Smooth Fixed-Point Iterations",
///   SIAM J. Optim. 2020 — Anderson acceleration for ADMM.
///
///   Rockafellar & Uryasev, "Optimization of Conditional Value-at-Risk",
///   Journal of Risk, 2000 — CVaR reformulation.

#include <functional>
#include <vector>

#include "constraints/constraint_set.h"
#include "core/types.h"
#include "risk/risk_result.h"
#include "simulation/scenario_matrix.h"

namespace cpo {

/// Configuration for the ADMM optimizer.
struct AdmmConfig {
    ScalarCPU confidence_level = 0.95;  ///< CVaR confidence (alpha for risk).
    ScalarCPU target_return = 0.0;      ///< Minimum expected return constraint.
    bool has_target_return = false;      ///< Whether to enforce target return.

    // ADMM parameters.
    ScalarCPU rho = 1.0;           ///< Initial penalty parameter.
    ScalarCPU rho_min = 1e-4;      ///< Minimum rho (adaptive).
    ScalarCPU rho_max = 1e4;       ///< Maximum rho (adaptive).
    ScalarCPU tau_incr = 2.0;      ///< rho increase factor (Boyd 2011 Eq. 3.13, legacy).
    ScalarCPU tau_decr = 2.0;      ///< rho decrease factor (legacy).
    ScalarCPU mu_adapt = 10.0;     ///< Primal/dual residual ratio threshold (legacy).
    bool adaptive_rho = true;      ///< Enable adaptive rho update.

    /// Residual balancing (Wohlberg 2017) replaces Boyd's ratio test.
    /// When enabled, rho is continuously adjusted to equalize normalized
    /// primal/dual residuals: rho *= sqrt((r_pri/eps_pri)/(r_dual/eps_dual)).
    /// Per-iteration rho change is clamped to [1/rho_balance_tau, rho_balance_tau].
    bool residual_balancing = false;       ///< Use Wohlberg residual balancing.
    ScalarCPU rho_balance_tau = 2.0;       ///< Max per-iteration rho change factor.

    /// Over-relaxation (Boyd 2011, Section 3.4.3, Eq. 3.19-3.20).
    /// Blends x and z_prev in z/u-update: x_hat = alpha * x + (1 - alpha) * z_prev.
    /// alpha_relax = 1.0 recovers vanilla ADMM. Recommended range: [1.5, 1.8].
    ScalarCPU alpha_relax = 1.0;   ///< Over-relaxation parameter (1.0 = vanilla).

    /// Anderson acceleration depth (Zhang et al. 2020, type-I).
    /// Stores the last m iterates to extrapolate the z/zeta fixed-point.
    /// Set to 0 to disable. Recommended: 3-5.
    int anderson_depth = 0;        ///< Anderson acceleration depth (0 = disabled).

    // Convergence criteria (Boyd 2011 Section 3.3).
    int max_iter = 500;            ///< Maximum ADMM iterations.
    ScalarCPU abs_tol = 1e-6;      ///< Absolute tolerance for residuals.
    ScalarCPU rel_tol = 1e-4;      ///< Relative tolerance for residuals.

    // Portfolio constraints (position limits, turnover, sector).
    ConstraintSet constraints;

    // x-update parameters.
    ScalarCPU x_update_lr = 0.01;  ///< Initial learning rate for proximal gradient x-update
                                    ///< (reduced by backtracking line search as needed).
    int x_update_steps = 20;       ///< Inner gradient steps per x-update (auto-scaled
                                    ///< to max(x_update_steps, n_assets) at runtime).

    // Logging.
    bool verbose = false;          ///< Print per-iteration convergence info.
};

/// Per-iteration convergence diagnostics.
struct AdmmIterInfo {
    int iteration = 0;
    ScalarCPU primal_residual = 0.0;  ///< ||x - z||_2
    ScalarCPU dual_residual = 0.0;    ///< ||rho * (z - z_prev)||_2
    ScalarCPU eps_pri = 0.0;          ///< Primal tolerance.
    ScalarCPU eps_dual = 0.0;         ///< Dual tolerance.
    ScalarCPU rho = 0.0;              ///< Current penalty parameter.
    ScalarCPU objective = 0.0;        ///< CVaR objective at current z.
};

/// Result of the ADMM optimization.
struct AdmmResult {
    VectorXd weights;               ///< Optimal portfolio weights.
    ScalarCPU cvar = 0.0;           ///< CVaR at the optimal weights.
    ScalarCPU expected_return = 0.0;///< Portfolio expected return (mu' w).
    ScalarCPU zeta = 0.0;           ///< Optimal auxiliary variable (VaR).
    int iterations = 0;             ///< Number of ADMM iterations to converge.
    bool converged = false;         ///< Whether convergence criteria were met.
    std::vector<AdmmIterInfo> history;  ///< Per-iteration diagnostics.
};

/// Solve Mean-CVaR optimization using ADMM.
///
/// The solver operates on CPU-resident scenario matrices (double precision).
/// GPU acceleration of the x-update is handled by admm_kernels.cu when
/// a GPU scenario matrix is provided.
///
/// @param scenarios Return matrix (n_scenarios x n_assets, double).
/// @param mu Expected return vector (n_assets, double).
/// @param config ADMM configuration.
/// @param w_init Initial weights (optional; defaults to 1/n).
/// @return AdmmResult with optimal weights and diagnostics.
AdmmResult admm_solve(const MatrixXd& scenarios,
                       const VectorXd& mu,
                       const AdmmConfig& config,
                       const VectorXd& w_init = VectorXd());

/// Solve Mean-CVaR optimization using ADMM with GPU-accelerated x-update.
///
/// The x-update inner loop runs on GPU using pre-allocated device buffers,
/// while z-update and u-update (cheap projections) remain on CPU.
///
/// Scenarios are downloaded to CPU once for cold-path operations
/// (find_optimal_zeta, final refinement).
///
/// @param scenarios_gpu GPU-resident scenario matrix (float, column-major).
/// @param mu Expected return vector (n_assets, double).
/// @param config ADMM configuration.
/// @param w_init Initial weights (optional; defaults to 1/n).
/// @return AdmmResult with optimal weights and diagnostics.
AdmmResult admm_solve(const ScenarioMatrix& scenarios_gpu,
                       const VectorXd& mu,
                       const AdmmConfig& config,
                       const VectorXd& w_init = VectorXd());

}  // namespace cpo
