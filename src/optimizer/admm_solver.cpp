#include "optimizer/admm_solver.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <spdlog/spdlog.h>

#include "optimizer/objective.h"
#include "optimizer/projections.h"

namespace cpo {

namespace {

/// Proximal gradient x-update for the Rockafellar-Uryasev objective.
///
/// Solves (approximately):
///   x^{k+1} = argmin_x { F_alpha(x, zeta) + (rho/2)||x - z + u||^2 }
///
/// where F_alpha(x, zeta) = zeta + (1/(N*alpha)) * sum_i max(0, -r_i'x - zeta)
///
/// Uses proximal gradient descent: the augmented Lagrangian term is smooth,
/// so we take gradient steps of the full augmented objective.
///
/// The zeta variable is optimized jointly with w in each inner step.
///
/// @param scenarios Return matrix (n_scenarios x n_assets).
/// @param z_minus_u Target for the augmented term (z - u).
/// @param x_prev Previous x iterate (warm start).
/// @param zeta_prev Previous zeta (warm start).
/// @param alpha Tail probability.
/// @param rho ADMM penalty parameter.
/// @param lr Learning rate for gradient steps.
/// @param n_steps Number of inner gradient steps.
/// @param[out] zeta_out Updated zeta.
/// @return Updated x vector.
VectorXd x_update_proximal(const MatrixXd& scenarios,
                            const VectorXd& z_minus_u,
                            const VectorXd& x_prev,
                            ScalarCPU zeta_prev,
                            ScalarCPU alpha,
                            ScalarCPU rho,
                            ScalarCPU lr,
                            int n_steps,
                            ScalarCPU& zeta_out) {
    VectorXd x = x_prev;
    double zeta = zeta_prev;
    const int n = static_cast<int>(x.size());

    for (int step = 0; step < n_steps; ++step) {
        // Evaluate R-U objective and subgradient.
        auto obj = evaluate_objective_cpu(scenarios, x, zeta, alpha);

        // Gradient of augmented Lagrangian w.r.t. x:
        //   grad = dF/dw + rho * (x - z_minus_u)
        VectorXd grad_x = obj.grad_w + rho * (x - z_minus_u);

        // Gradient w.r.t. zeta (no augmented term for zeta).
        double grad_zeta = obj.grad_zeta;

        // Gradient step.
        x -= lr * grad_x;
        zeta -= lr * grad_zeta;
    }

    zeta_out = zeta;
    return x;
}

/// Z-update: project onto the constraint set.
///
/// The constraint set is the intersection of:
///   - Probability simplex: {w : 1'w = 1, w >= 0}
///   - Position limits, turnover, sector (via ConstraintSet)
///   - Target return: {w : mu'w >= target_return} (if specified)
///
/// Uses project_constraints (generalized Dykstra's) for the combined
/// constraint projection, then applies target return correction.
VectorXd z_update(const VectorXd& x_plus_u,
                   const VectorXd& mu,
                   const AdmmConfig& config) {
    VectorXd z = project_constraints(x_plus_u, config.constraints);

    // Enforce target return constraint via projection.
    // If mu'z < target, shift z along mu direction and re-project.
    if (config.has_target_return) {
        double port_return = mu.dot(z);
        if (port_return < config.target_return) {
            for (int correction = 0; correction < 10; ++correction) {
                double deficit = config.target_return - mu.dot(z);
                if (deficit <= 1e-10) break;

                double mu_norm_sq = mu.squaredNorm();
                if (mu_norm_sq < 1e-15) break;
                VectorXd z_shifted = z + (deficit / mu_norm_sq) * mu;

                z = project_constraints(z_shifted, config.constraints);
            }
        }
    }

    return z;
}

}  // namespace

// ── ADMM solver ─────────────────────────────────────────────────────

AdmmResult admm_solve(const MatrixXd& scenarios,
                       const VectorXd& mu,
                       const AdmmConfig& config,
                       const VectorXd& w_init) {
    const int n_scenarios = static_cast<int>(scenarios.rows());
    const int n_assets = static_cast<int>(scenarios.cols());

    if (mu.size() != n_assets) {
        throw std::runtime_error(
            "admm_solve: mu size (" + std::to_string(mu.size()) +
            ") != n_assets (" + std::to_string(n_assets) + ")");
    }
    config.constraints.validate(n_assets);

    // Tail probability for R-U formulation.
    const double alpha = 1.0 - config.confidence_level;

    spdlog::info("ADMM solver: {} scenarios x {} assets, alpha={:.4f}, "
                 "rho={:.4f}, max_iter={}",
                 n_scenarios, n_assets, alpha, config.rho, config.max_iter);

    // ── Initialize variables ────────────────────────────────────────
    // x: primal variable (unconstrained iterate).
    // z: constrained iterate (feasible after projection).
    // u: scaled dual variable.
    VectorXd x(n_assets);
    if (w_init.size() == n_assets) {
        x = w_init;
    } else {
        x = VectorXd::Constant(n_assets, 1.0 / n_assets);
    }

    VectorXd z = x;
    VectorXd u = VectorXd::Zero(n_assets);
    VectorXd z_prev = z;

    // Initialize zeta (VaR estimate) at the optimal value for initial w.
    double zeta = find_optimal_zeta(scenarios, z, alpha);

    double rho = config.rho;

    AdmmResult result;
    result.history.reserve(config.max_iter);

    // ── ADMM iteration loop ─────────────────────────────────────────
    // Boyd et al. 2011, Eq. (3.1)-(3.3):
    //   x^{k+1} = argmin_x { f(x) + (rho/2)||x - z^k + u^k||^2 }
    //   z^{k+1} = project_C(x^{k+1} + u^k)
    //   u^{k+1} = u^k + x^{k+1} - z^{k+1}

    for (int iter = 0; iter < config.max_iter; ++iter) {
        z_prev = z;

        // ── x-update: proximal gradient on augmented R-U objective ──
        VectorXd z_minus_u = z - u;
        x = x_update_proximal(scenarios, z_minus_u, x, zeta, alpha,
                               rho, config.x_update_lr,
                               config.x_update_steps, zeta);

        // ── z-update: project (x + u) onto constraint set ───────────
        z = z_update(x + u, mu, config);

        // ── u-update: dual variable ─────────────────────────────────
        u = u + x - z;

        // ── Convergence check (Boyd 2011, Section 3.3) ──────────────
        // Primal residual: r = x - z
        // Dual residual:   s = rho * (z - z_prev)
        VectorXd r = x - z;
        VectorXd s = rho * (z - z_prev);

        double primal_res = r.norm();
        double dual_res = s.norm();

        // Tolerances (Boyd 2011, Eq. 3.12):
        //   eps_pri  = sqrt(n) * abs_tol + rel_tol * max(||x||, ||z||)
        //   eps_dual = sqrt(n) * abs_tol + rel_tol * ||rho * u||
        double sqrt_n = std::sqrt(static_cast<double>(n_assets));
        double eps_pri = sqrt_n * config.abs_tol +
                         config.rel_tol * std::max(x.norm(), z.norm());
        double eps_dual = sqrt_n * config.abs_tol +
                          config.rel_tol * (rho * u).norm();

        // Evaluate objective at current z (feasible point).
        auto obj = evaluate_objective_cpu(scenarios, z, zeta, alpha);

        // Record iteration info.
        AdmmIterInfo info;
        info.iteration = iter;
        info.primal_residual = primal_res;
        info.dual_residual = dual_res;
        info.eps_pri = eps_pri;
        info.eps_dual = eps_dual;
        info.rho = rho;
        info.objective = obj.value;
        result.history.push_back(info);

        if (config.verbose) {
            spdlog::info("  iter {:3d}: r_pri={:.2e} r_dual={:.2e} "
                         "eps_pri={:.2e} eps_dual={:.2e} rho={:.4f} "
                         "obj={:.6f}",
                         iter, primal_res, dual_res, eps_pri, eps_dual,
                         rho, obj.value);
        }

        // Check convergence.
        if (primal_res <= eps_pri && dual_res <= eps_dual) {
            spdlog::info("ADMM converged at iteration {} "
                         "(r_pri={:.2e}, r_dual={:.2e})",
                         iter, primal_res, dual_res);
            result.converged = true;
            result.iterations = iter + 1;
            break;
        }

        // ── Adaptive rho update (Boyd 2011, Section 3.4.1) ──────────
        // Eq. (3.13): balance primal and dual residuals.
        if (config.adaptive_rho) {
            if (primal_res > config.mu_adapt * dual_res) {
                rho *= config.tau_incr;
                u /= config.tau_incr;  // Rescale u to keep rho*u constant.
            } else if (dual_res > config.mu_adapt * primal_res) {
                rho /= config.tau_decr;
                u *= config.tau_decr;
            }
            rho = std::clamp(rho, config.rho_min, config.rho_max);
        }

        if (iter == config.max_iter - 1) {
            spdlog::warn("ADMM did not converge within {} iterations "
                         "(r_pri={:.2e}, r_dual={:.2e})",
                         config.max_iter, primal_res, dual_res);
            result.iterations = config.max_iter;
        }
    }

    // ── Final result ────────────────────────────────────────────────
    result.weights = z;
    result.expected_return = mu.dot(z);
    result.zeta = zeta;

    // Compute final CVaR at the solution.
    auto final_obj = evaluate_objective_cpu(scenarios, z, zeta, alpha);
    result.cvar = final_obj.value;

    // Refine: find true optimal zeta for the final weights.
    double zeta_opt = find_optimal_zeta(scenarios, z, alpha);
    auto refined_obj = evaluate_objective_cpu(scenarios, z, zeta_opt, alpha);
    if (refined_obj.value < result.cvar) {
        result.cvar = refined_obj.value;
        result.zeta = zeta_opt;
    }

    spdlog::info("ADMM result: CVaR={:.6f} E[r]={:.6f} zeta={:.6f} "
                 "iters={} converged={}",
                 result.cvar, result.expected_return, result.zeta,
                 result.iterations, result.converged);

    return result;
}

}  // namespace cpo
