#include "optimizer/line_search.h"

#include <algorithm>
#include <cmath>

namespace cpo {

LineSearchResult backtracking_line_search(
    double f_current,
    double grad_sq_norm,
    const std::function<double(double)>& eval_fn,
    const LineSearchConfig& config) {
    LineSearchResult result;

    // ── Stationary point check ─────────────────────────────────────
    // If the gradient is (near) zero, we are already at a stationary
    // point. No step can produce a decrease, so return immediately.
    if (grad_sq_norm <= 1e-30) {
        result.step_size = 0.0;
        result.objective_value = f_current;
        result.backtracks = 0;
        result.success = true;
        return result;
    }

    // ── Backtracking loop ──────────────────────────────────────────
    // Nocedal & Wright (2006), Algorithm 3.1:
    //   Start with alpha = initial_step.
    //   While f(x - alpha * grad) > f(x) - c * alpha * ||grad||^2:
    //     alpha *= shrink_factor
    //
    // The right-hand side is the Armijo sufficient decrease threshold.
    // The term c * alpha * ||grad||^2 is the expected decrease from a
    // first-order Taylor expansion scaled by the Armijo parameter c.
    double alpha = config.initial_step;

    for (int k = 0; k < config.max_backtracks; ++k) {
        double f_trial = eval_fn(alpha);
        double armijo_threshold = f_current - config.armijo_c * alpha * grad_sq_norm;

        if (f_trial <= armijo_threshold) {
            // Armijo condition satisfied — accept this step.
            result.step_size = alpha;
            result.objective_value = f_trial;
            result.backtracks = k;
            result.success = true;
            return result;
        }

        // Shrink step size and try again.
        alpha *= config.shrink_factor;

        // Enforce minimum step size floor.
        if (alpha < config.min_step) {
            alpha = config.min_step;

            // Evaluate at the minimum step and return regardless.
            f_trial = eval_fn(alpha);
            result.step_size = alpha;
            result.objective_value = f_trial;
            result.backtracks = k + 1;
            result.success = (f_trial <= f_current - config.armijo_c * alpha * grad_sq_norm);
            return result;
        }
    }

    // ── All backtracks exhausted ───────────────────────────────────
    // Return the last (smallest) step tried. The Armijo condition was
    // NOT satisfied, so success = false. The caller can decide whether
    // to use this step or fall back to a fixed learning rate.
    double f_final = eval_fn(alpha);
    result.step_size = alpha;
    result.objective_value = f_final;
    result.backtracks = config.max_backtracks;
    result.success = false;
    return result;
}

}  // namespace cpo
