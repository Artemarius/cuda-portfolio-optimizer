#include "optimizer/admm_solver.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <spdlog/spdlog.h>

#include <memory>

#include "optimizer/admm_kernels.h"
#include "optimizer/anderson_acceleration.h"
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

        // Current augmented objective value (for line search).
        double augmented_value = obj.value
            + (rho / 2.0) * (x - z_minus_u).squaredNorm();

        // Backtracking line search on x (Armijo condition).
        // f(x - lr*g) <= f(x) - c * lr * ||g||^2
        // Nocedal & Wright (2006), Algorithm 3.1.
        double grad_sq = grad_x.squaredNorm();
        double step_lr = lr;
        constexpr double armijo_c = 1e-4;
        constexpr double bt_shrink = 0.5;
        constexpr int max_bt_steps = 10;

        for (int bt = 0; bt < max_bt_steps; ++bt) {
            VectorXd x_trial = x - step_lr * grad_x;
            auto trial_obj = evaluate_objective_cpu(
                scenarios, x_trial, zeta, alpha);
            double trial_augmented = trial_obj.value
                + (rho / 2.0) * (x_trial - z_minus_u).squaredNorm();

            if (trial_augmented <= augmented_value
                                    - armijo_c * step_lr * grad_sq) {
                break;  // Armijo condition satisfied.
            }
            step_lr *= bt_shrink;
        }

        // Gradient step with line-searched learning rate.
        x -= step_lr * grad_x;
        zeta -= lr * grad_zeta;  // Zeta uses fixed lr (cheap, 1D).
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

    // Auto-scale inner gradient steps to problem dimension.
    const int effective_x_steps = std::max(config.x_update_steps, n_assets);

    spdlog::info("ADMM solver: {} scenarios x {} assets, alpha={:.4f}, "
                 "rho={:.4f}, max_iter={}",
                 n_scenarios, n_assets, alpha, config.rho, config.max_iter);
    spdlog::info("ADMM config: alpha_relax={:.2f}, residual_balancing={}, "
                 "anderson_depth={}, x_steps={}, lr={:.4f}",
                 config.alpha_relax, config.residual_balancing,
                 config.anderson_depth, effective_x_steps,
                 config.x_update_lr);

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

    // Anderson acceleration on the (z, zeta) fixed-point (Zhang et al. 2020).
    std::unique_ptr<AndersonAccelerator> anderson;
    if (config.anderson_depth > 0) {
        anderson = std::make_unique<AndersonAccelerator>(
            n_assets + 1, config.anderson_depth);
    }

    AdmmResult result;
    result.history.reserve(config.max_iter);

    // ── ADMM iteration loop ─────────────────────────────────────────
    // Boyd et al. 2011, Eq. (3.1)-(3.3):
    //   x^{k+1} = argmin_x { f(x) + (rho/2)||x - z^k + u^k||^2 }
    //   z^{k+1} = project_C(x^{k+1} + u^k)
    //   u^{k+1} = u^k + x^{k+1} - z^{k+1}

    for (int iter = 0; iter < config.max_iter; ++iter) {
        z_prev = z;
        double zeta_prev = zeta;

        // ── x-update: proximal gradient on augmented R-U objective ──
        VectorXd z_minus_u = z - u;
        x = x_update_proximal(scenarios, z_minus_u, x, zeta, alpha,
                               rho, config.x_update_lr,
                               effective_x_steps, zeta);

        // ── Over-relaxation (Boyd 2011, S3.4.3, Eq. 3.19-3.20) ─────
        // x_hat = alpha * x + (1 - alpha) * z_prev
        VectorXd x_hat = config.alpha_relax * x
                        + (1.0 - config.alpha_relax) * z_prev;

        // ── z-update: project (x_hat + u) onto constraint set ───────
        z = z_update(x_hat + u, mu, config);

        // ── u-update: dual variable (using x_hat) ───────────────────
        u = u + x_hat - z;

        // ── Anderson acceleration on z/zeta (Zhang et al. 2020) ─────
        if (anderson) {
            VectorXd state_prev(n_assets + 1);
            state_prev.head(n_assets) = z_prev;
            state_prev(n_assets) = zeta_prev;

            VectorXd state_new(n_assets + 1);
            state_new.head(n_assets) = z;
            state_new(n_assets) = zeta;

            VectorXd state_accel = anderson->accelerate(state_prev, state_new);

            // Safeguard: accept only if accelerated state is finite and
            // doesn't worsen the primal residual dramatically.
            if (state_accel.allFinite()) {
                VectorXd z_accel = state_accel.head(n_assets);
                double zeta_accel = state_accel(n_assets);
                VectorXd r_accel = x - z_accel;
                if (r_accel.norm() <= 2.0 * (x - z).norm() + 1e-8) {
                    z = z_accel;
                    zeta = zeta_accel;
                } else {
                    anderson->reset();
                    spdlog::debug("  Anderson restart at iter {}", iter);
                }
            } else {
                anderson->reset();
                spdlog::debug("  Anderson NaN restart at iter {}", iter);
            }
        }

        // Bail out if x/z have become non-finite (numerical explosion).
        if (!x.allFinite() || !z.allFinite() || !std::isfinite(zeta)) {
            spdlog::warn("ADMM non-finite state at iter {}, aborting", iter);
            result.iterations = iter + 1;
            break;
        }

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

        // ── Adaptive rho update ──────────────────────────────────────
        if (config.adaptive_rho) {
            if (config.residual_balancing) {
                // Wohlberg 2017: residual balancing.
                // rho *= sqrt((r_pri/eps_pri) / (r_dual/eps_dual))
                // clamped to [1/tau, tau] per iteration.
                double ratio_pri = primal_res / std::max(eps_pri, 1e-15);
                double ratio_dual = dual_res / std::max(eps_dual, 1e-15);
                if (ratio_dual > 1e-15) {
                    double rho_scale = std::sqrt(ratio_pri / ratio_dual);
                    rho_scale = std::clamp(rho_scale,
                                           1.0 / config.rho_balance_tau,
                                           config.rho_balance_tau);
                    double rho_new = rho * rho_scale;
                    rho_new = std::clamp(rho_new, config.rho_min,
                                         config.rho_max);
                    if (rho_new != rho) {
                        u *= (rho / rho_new);  // Rescale u to keep rho*u constant.
                        rho = rho_new;
                    }
                }
            } else {
                // Legacy: Boyd 2011, Section 3.4.1, Eq. (3.13).
                if (primal_res > config.mu_adapt * dual_res) {
                    rho *= config.tau_incr;
                    u /= config.tau_incr;
                } else if (dual_res > config.mu_adapt * primal_res) {
                    rho /= config.tau_decr;
                    u *= config.tau_decr;
                }
                rho = std::clamp(rho, config.rho_min, config.rho_max);
            }

            // Reset Anderson history when rho changes significantly,
            // since the fixed-point map has changed.
            if (anderson && result.history.size() >= 2) {
                double prev_rho = result.history[result.history.size() - 2].rho;
                if (std::abs(rho - prev_rho) / std::max(prev_rho, 1e-15) > 0.1) {
                    anderson->reset();
                }
            }
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

// ── GPU ADMM solver ─────────────────────────────────────────────────

namespace {

/// GPU-accelerated proximal gradient x-update.
///
/// Same structure as the CPU x_update_proximal, but evaluates the R-U
/// objective on GPU via pre-allocated buffers.
///
/// The weight vector is cast float→double→float each inner step since the
/// GPU kernel operates in float while ADMM state is double.
VectorXd x_update_proximal_gpu(const ScenarioMatrix& scenarios_gpu,
                                const VectorXd& z_minus_u,
                                const VectorXd& x_prev,
                                ScalarCPU zeta_prev,
                                ScalarCPU alpha,
                                ScalarCPU rho,
                                ScalarCPU lr,
                                int n_steps,
                                ScalarCPU& zeta_out,
                                GpuAdmmBuffers* buffers) {
    VectorXd x = x_prev;
    double zeta = zeta_prev;
    const int n = static_cast<int>(x.size());
    const int n_scenarios = scenarios_gpu.n_scenarios();
    const double inv_n_alpha = 1.0 / (n_scenarios * alpha);

    for (int step = 0; step < n_steps; ++step) {
        // Cast weights to float for GPU kernel.
        VectorXs w_f = x.cast<float>();

        // Evaluate on GPU (raw kernel outputs, not yet scaled by 1/(N*alpha)).
        auto gpu_res = evaluate_objective_gpu(
            scenarios_gpu, w_f, static_cast<float>(zeta), buffers);

        // Assemble gradient: dF/dw = grad_w / (N * alpha)
        VectorXd grad_w = inv_n_alpha * gpu_res.grad_w;

        // Gradient of augmented Lagrangian w.r.t. x:
        //   grad = dF/dw + rho * (x - z_minus_u)
        VectorXd grad_x = grad_w + rho * (x - z_minus_u);

        // Gradient w.r.t. zeta: dF/dzeta = 1 - count / (N * alpha)
        double grad_zeta = 1.0 - gpu_res.count_exceeding * inv_n_alpha;

        // Current augmented objective value.
        double obj_value = zeta + inv_n_alpha * gpu_res.value;
        double augmented_value = obj_value
            + (rho / 2.0) * (x - z_minus_u).squaredNorm();

        // Backtracking line search on x (Armijo condition).
        double grad_sq = grad_x.squaredNorm();
        double step_lr = lr;
        constexpr double armijo_c = 1e-4;
        constexpr double bt_shrink = 0.5;
        constexpr int max_bt_steps = 10;

        for (int bt = 0; bt < max_bt_steps; ++bt) {
            VectorXd x_trial = x - step_lr * grad_x;
            VectorXs w_trial_f = x_trial.cast<float>();
            auto trial_res = evaluate_objective_gpu(
                scenarios_gpu, w_trial_f,
                static_cast<float>(zeta), buffers);
            double trial_obj = zeta + inv_n_alpha * trial_res.value;
            double trial_augmented = trial_obj
                + (rho / 2.0) * (x_trial - z_minus_u).squaredNorm();

            if (trial_augmented <= augmented_value
                                    - armijo_c * step_lr * grad_sq) {
                break;
            }
            step_lr *= bt_shrink;
        }

        // Gradient step with line-searched learning rate.
        x -= step_lr * grad_x;
        zeta -= lr * grad_zeta;  // Zeta uses fixed lr (cheap, 1D).
    }

    zeta_out = zeta;
    return x;
}

/// Evaluate the R-U objective value on GPU (scalar only, for convergence logging).
ScalarCPU evaluate_objective_value_gpu(const ScenarioMatrix& scenarios_gpu,
                                        const VectorXd& w,
                                        ScalarCPU zeta,
                                        ScalarCPU alpha,
                                        GpuAdmmBuffers* buffers) {
    VectorXs w_f = w.cast<float>();
    auto gpu_res = evaluate_objective_gpu(
        scenarios_gpu, w_f, static_cast<float>(zeta), buffers);
    double inv_n_alpha = 1.0 / (scenarios_gpu.n_scenarios() * alpha);
    return zeta + inv_n_alpha * gpu_res.value;
}

}  // anonymous namespace

AdmmResult admm_solve(const ScenarioMatrix& scenarios_gpu,
                       const VectorXd& mu,
                       const AdmmConfig& config,
                       const VectorXd& w_init) {
    const int n_scenarios = scenarios_gpu.n_scenarios();
    const int n_assets = scenarios_gpu.n_assets();

    if (mu.size() != n_assets) {
        throw std::runtime_error(
            "admm_solve(GPU): mu size (" + std::to_string(mu.size()) +
            ") != n_assets (" + std::to_string(n_assets) + ")");
    }
    config.constraints.validate(n_assets);

    const double alpha = 1.0 - config.confidence_level;

    const int effective_x_steps = std::max(config.x_update_steps, n_assets);

    spdlog::info("ADMM solver (GPU): {} scenarios x {} assets, alpha={:.4f}, "
                 "rho={:.4f}, max_iter={}",
                 n_scenarios, n_assets, alpha, config.rho, config.max_iter);
    spdlog::info("ADMM config (GPU): alpha_relax={:.2f}, residual_balancing={}, "
                 "anderson_depth={}, x_steps={}, lr={:.4f}",
                 config.alpha_relax, config.residual_balancing,
                 config.anderson_depth, effective_x_steps,
                 config.x_update_lr);

    // Pre-allocate GPU buffers once (reused for ~6000 kernel calls).
    auto gpu_buffers = create_gpu_admm_buffers(n_assets);

    // Download scenarios to CPU once for cold-path operations
    // (find_optimal_zeta, final refinement).
    MatrixXs float_scen = scenarios_gpu.to_host();
    MatrixXd scenarios_cpu = float_scen.cast<double>();

    // ── Initialize variables ────────────────────────────────────────
    VectorXd x(n_assets);
    if (w_init.size() == n_assets) {
        x = w_init;
    } else {
        x = VectorXd::Constant(n_assets, 1.0 / n_assets);
    }

    VectorXd z = x;
    VectorXd u = VectorXd::Zero(n_assets);
    VectorXd z_prev = z;

    double zeta = find_optimal_zeta(scenarios_cpu, z, alpha);
    double rho = config.rho;

    // Anderson acceleration on the (z, zeta) fixed-point (Zhang et al. 2020).
    std::unique_ptr<AndersonAccelerator> anderson;
    if (config.anderson_depth > 0) {
        anderson = std::make_unique<AndersonAccelerator>(
            n_assets + 1, config.anderson_depth);
    }

    AdmmResult result;
    result.history.reserve(config.max_iter);

    // ── ADMM iteration loop ─────────────────────────────────────────
    for (int iter = 0; iter < config.max_iter; ++iter) {
        z_prev = z;
        double zeta_prev = zeta;

        // ── x-update: GPU-accelerated proximal gradient ─────────────
        VectorXd z_minus_u = z - u;
        x = x_update_proximal_gpu(scenarios_gpu, z_minus_u, x, zeta, alpha,
                                   rho, config.x_update_lr,
                                   effective_x_steps, zeta,
                                   gpu_buffers.get());

        // ── Over-relaxation (Boyd 2011, S3.4.3, Eq. 3.19-3.20) ─────
        VectorXd x_hat = config.alpha_relax * x
                        + (1.0 - config.alpha_relax) * z_prev;

        // ── z-update: project (x_hat + u) onto constraint set (CPU) ─
        z = z_update(x_hat + u, mu, config);

        // ── u-update: dual variable (using x_hat) ───────────────────
        u = u + x_hat - z;

        // ── Anderson acceleration on z/zeta (Zhang et al. 2020) ─────
        if (anderson) {
            VectorXd state_prev(n_assets + 1);
            state_prev.head(n_assets) = z_prev;
            state_prev(n_assets) = zeta_prev;

            VectorXd state_new(n_assets + 1);
            state_new.head(n_assets) = z;
            state_new(n_assets) = zeta;

            VectorXd state_accel = anderson->accelerate(state_prev, state_new);

            if (state_accel.allFinite()) {
                VectorXd z_accel = state_accel.head(n_assets);
                double zeta_accel = state_accel(n_assets);
                VectorXd r_accel = x - z_accel;
                if (r_accel.norm() <= 2.0 * (x - z).norm() + 1e-8) {
                    z = z_accel;
                    zeta = zeta_accel;
                } else {
                    anderson->reset();
                    spdlog::debug("  Anderson restart (GPU) at iter {}", iter);
                }
            } else {
                anderson->reset();
                spdlog::debug("  Anderson NaN restart (GPU) at iter {}", iter);
            }
        }

        // Bail out if x/z have become non-finite.
        if (!x.allFinite() || !z.allFinite() || !std::isfinite(zeta)) {
            spdlog::warn("ADMM (GPU) non-finite state at iter {}, aborting",
                         iter);
            result.iterations = iter + 1;
            break;
        }

        // ── Convergence check ───────────────────────────────────────
        VectorXd r = x - z;
        VectorXd s = rho * (z - z_prev);

        double primal_res = r.norm();
        double dual_res = s.norm();

        double sqrt_n = std::sqrt(static_cast<double>(n_assets));
        double eps_pri = sqrt_n * config.abs_tol +
                         config.rel_tol * std::max(x.norm(), z.norm());
        double eps_dual = sqrt_n * config.abs_tol +
                          config.rel_tol * (rho * u).norm();

        // Evaluate objective at current z via GPU.
        double obj_value = evaluate_objective_value_gpu(
            scenarios_gpu, z, zeta, alpha, gpu_buffers.get());

        AdmmIterInfo info;
        info.iteration = iter;
        info.primal_residual = primal_res;
        info.dual_residual = dual_res;
        info.eps_pri = eps_pri;
        info.eps_dual = eps_dual;
        info.rho = rho;
        info.objective = obj_value;
        result.history.push_back(info);

        if (config.verbose) {
            spdlog::info("  iter {:3d}: r_pri={:.2e} r_dual={:.2e} "
                         "eps_pri={:.2e} eps_dual={:.2e} rho={:.4f} "
                         "obj={:.6f}",
                         iter, primal_res, dual_res, eps_pri, eps_dual,
                         rho, obj_value);
        }

        if (primal_res <= eps_pri && dual_res <= eps_dual) {
            spdlog::info("ADMM (GPU) converged at iteration {} "
                         "(r_pri={:.2e}, r_dual={:.2e})",
                         iter, primal_res, dual_res);
            result.converged = true;
            result.iterations = iter + 1;
            break;
        }

        // ── Adaptive rho update ─────────────────────────────────────
        if (config.adaptive_rho) {
            if (config.residual_balancing) {
                // Wohlberg 2017: residual balancing.
                double ratio_pri = primal_res / std::max(eps_pri, 1e-15);
                double ratio_dual = dual_res / std::max(eps_dual, 1e-15);
                if (ratio_dual > 1e-15) {
                    double rho_scale = std::sqrt(ratio_pri / ratio_dual);
                    rho_scale = std::clamp(rho_scale,
                                           1.0 / config.rho_balance_tau,
                                           config.rho_balance_tau);
                    double rho_new = rho * rho_scale;
                    rho_new = std::clamp(rho_new, config.rho_min,
                                         config.rho_max);
                    if (rho_new != rho) {
                        u *= (rho / rho_new);
                        rho = rho_new;
                    }
                }
            } else {
                // Legacy: Boyd 2011, Section 3.4.1, Eq. (3.13).
                if (primal_res > config.mu_adapt * dual_res) {
                    rho *= config.tau_incr;
                    u /= config.tau_incr;
                } else if (dual_res > config.mu_adapt * primal_res) {
                    rho /= config.tau_decr;
                    u *= config.tau_decr;
                }
                rho = std::clamp(rho, config.rho_min, config.rho_max);
            }

            // Reset Anderson history when rho changes significantly.
            if (anderson && result.history.size() >= 2) {
                double prev_rho = result.history[result.history.size() - 2].rho;
                if (std::abs(rho - prev_rho) / std::max(prev_rho, 1e-15) > 0.1) {
                    anderson->reset();
                }
            }
        }

        if (iter == config.max_iter - 1) {
            spdlog::warn("ADMM (GPU) did not converge within {} iterations "
                         "(r_pri={:.2e}, r_dual={:.2e})",
                         config.max_iter, primal_res, dual_res);
            result.iterations = config.max_iter;
        }
    }

    // ── Final result (CPU double-precision refinement) ──────────────
    result.weights = z;
    result.expected_return = mu.dot(z);
    result.zeta = zeta;

    auto final_obj = evaluate_objective_cpu(scenarios_cpu, z, zeta, alpha);
    result.cvar = final_obj.value;

    double zeta_opt = find_optimal_zeta(scenarios_cpu, z, alpha);
    auto refined_obj = evaluate_objective_cpu(scenarios_cpu, z, zeta_opt, alpha);
    if (refined_obj.value < result.cvar) {
        result.cvar = refined_obj.value;
        result.zeta = zeta_opt;
    }

    spdlog::info("ADMM (GPU) result: CVaR={:.6f} E[r]={:.6f} zeta={:.6f} "
                 "iters={} converged={}",
                 result.cvar, result.expected_return, result.zeta,
                 result.iterations, result.converged);

    return result;
}

}  // namespace cpo
