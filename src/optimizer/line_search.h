#pragma once

/// @file line_search.h
/// @brief Backtracking line search with Armijo sufficient decrease condition.
///
/// Provides a generic backtracking line search for use in gradient-based
/// optimization. The primary use case is the ADMM x-update, where it
/// replaces the fixed learning rate with an adaptive step size.
///
/// The Armijo (sufficient decrease) condition ensures that the accepted
/// step produces a meaningful reduction in the objective:
///
///   f(x - alpha * grad) <= f(x) - c * alpha * ||grad||^2
///
/// Starting from an aggressive initial step, the algorithm repeatedly
/// shrinks alpha until this condition is satisfied or the maximum number
/// of backtracks is reached.
///
/// Reference:
///   Nocedal & Wright, "Numerical Optimization", 2nd ed., 2006,
///   Algorithm 3.1 (Backtracking Line Search).

#include <functional>

#include "core/types.h"

namespace cpo {

/// Configuration for Armijo backtracking line search.
///
/// Default parameters follow standard recommendations from
/// Nocedal & Wright (2006), Section 3.1.
struct LineSearchConfig {
    double initial_step = 1.0;     ///< Starting step size (aggressive).
    double shrink_factor = 0.5;    ///< Factor to reduce step by each backtrack.
    double armijo_c = 1e-4;        ///< Sufficient decrease parameter c (Armijo condition).
    int max_backtracks = 20;       ///< Maximum backtracking steps before giving up.
    double min_step = 1e-10;       ///< Minimum step size (floor).
};

/// Result of a line search.
struct LineSearchResult {
    double step_size = 0.0;        ///< Final accepted step size.
    double objective_value = 0.0;  ///< Objective value at the accepted point.
    int backtracks = 0;            ///< Number of backtracking steps taken.
    bool success = false;          ///< True if Armijo condition was satisfied.
};

/// Backtracking line search with Armijo sufficient decrease condition.
///
/// Given current objective value f(x), the squared gradient norm ||grad||^2,
/// and a callable that evaluates f(x - alpha * grad) for a given alpha,
/// finds the largest step size alpha in {initial_step * shrink^k : k=0,1,...}
/// satisfying the Armijo condition:
///
///   f(x - alpha * grad) <= f(x) - c * alpha * ||grad||^2
///
/// This is Algorithm 3.1 from Nocedal & Wright (2006) with the search
/// direction fixed to the negative gradient (steepest descent).
///
/// Special cases:
///   - If grad_sq_norm is near zero (<= 1e-30), the current point is
///     (approximately) a stationary point. Returns immediately with
///     step_size = 0, objective_value = f_current, success = true.
///
/// @param f_current    Current objective value f(x).
/// @param grad_sq_norm Squared L2 norm of the gradient ||grad||^2.
/// @param eval_fn      Callable evaluating f(x - alpha * grad) for a given alpha.
///                     Signature: double(double alpha) -> objective value.
/// @param config       Line search configuration parameters.
/// @return LineSearchResult with the accepted step size and diagnostics.
LineSearchResult backtracking_line_search(
    double f_current,
    double grad_sq_norm,
    const std::function<double(double)>& eval_fn,
    const LineSearchConfig& config = LineSearchConfig{});

}  // namespace cpo
