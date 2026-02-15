#include <gtest/gtest.h>

#include <cmath>
#include <functional>

#include "optimizer/line_search.h"

using namespace cpo;

// ── Quadratic objective tests ──────────────────────────────────────

TEST(BacktrackingLineSearch, Quadratic1D) {
    // f(x) = 0.5 * x^2,  grad = x.
    // Starting at x = 10, optimal step = 1.0 (lands at x = 0).
    // The line search should accept step = 1.0 on the first try
    // since f(10 - 1.0 * 10) = f(0) = 0 <= f(10) - c * 1.0 * 100.
    double x = 10.0;
    double f_current = 0.5 * x * x;   // 50.0
    double grad = x;                    // 10.0
    double grad_sq_norm = grad * grad;  // 100.0

    auto eval_fn = [&](double alpha) -> double {
        double x_new = x - alpha * grad;
        return 0.5 * x_new * x_new;
    };

    auto result = backtracking_line_search(f_current, grad_sq_norm, eval_fn);

    EXPECT_TRUE(result.success);
    EXPECT_NEAR(result.step_size, 1.0, 1e-12);
    EXPECT_NEAR(result.objective_value, 0.0, 1e-12);
    EXPECT_EQ(result.backtracks, 0);
}

TEST(BacktrackingLineSearch, QuadraticDecreases) {
    // f(x) = 0.5 * x^2, starting from x = 5.
    // Whatever step is accepted, the objective must decrease.
    double x = 5.0;
    double f_current = 0.5 * x * x;
    double grad = x;
    double grad_sq_norm = grad * grad;

    auto eval_fn = [&](double alpha) -> double {
        double x_new = x - alpha * grad;
        return 0.5 * x_new * x_new;
    };

    auto result = backtracking_line_search(f_current, grad_sq_norm, eval_fn);

    EXPECT_TRUE(result.success);
    EXPECT_LT(result.objective_value, f_current);
}

// ── Rosenbrock-like 2D function ────────────────────────────────────

TEST(BacktrackingLineSearch, Rosenbrock2D) {
    // f(x, y) = (1 - x)^2 + 100*(y - x^2)^2
    // Gradient: df/dx = -2(1-x) - 400*x*(y-x^2), df/dy = 200*(y-x^2)
    // Starting from (−1.0, 1.0). Verify sufficient decrease along
    // the negative gradient direction.
    double x = -1.0;
    double y = 1.0;

    auto f = [](double x, double y) {
        return (1.0 - x) * (1.0 - x) + 100.0 * (y - x * x) * (y - x * x);
    };

    double f_current = f(x, y);  // 4 + 0 = 4.0
    double grad_x = -2.0 * (1.0 - x) - 400.0 * x * (y - x * x);
    double grad_y = 200.0 * (y - x * x);
    double grad_sq_norm = grad_x * grad_x + grad_y * grad_y;

    auto eval_fn = [&](double alpha) -> double {
        double x_new = x - alpha * grad_x;
        double y_new = y - alpha * grad_y;
        return f(x_new, y_new);
    };

    auto result = backtracking_line_search(f_current, grad_sq_norm, eval_fn);

    EXPECT_TRUE(result.success);
    EXPECT_LT(result.objective_value, f_current);
    EXPECT_GT(result.step_size, 0.0);

    // Verify the Armijo condition explicitly.
    double armijo_rhs = f_current - 1e-4 * result.step_size * grad_sq_norm;
    EXPECT_LE(result.objective_value, armijo_rhs + 1e-15);
}

// ── Stationary point (zero gradient) ───────────────────────────────

TEST(BacktrackingLineSearch, ZeroGradient) {
    // At a minimum: gradient is zero. Should return immediately.
    double f_current = 3.14;
    double grad_sq_norm = 0.0;

    int call_count = 0;
    auto eval_fn = [&](double /*alpha*/) -> double {
        ++call_count;
        return f_current;
    };

    auto result = backtracking_line_search(f_current, grad_sq_norm, eval_fn);

    EXPECT_TRUE(result.success);
    EXPECT_NEAR(result.step_size, 0.0, 1e-30);
    EXPECT_NEAR(result.objective_value, f_current, 1e-15);
    EXPECT_EQ(result.backtracks, 0);
    // The eval_fn should NOT have been called.
    EXPECT_EQ(call_count, 0);
}

TEST(BacktrackingLineSearch, NearZeroGradient) {
    // Gradient is extremely small (1e-20). Should treat as zero.
    double f_current = 1.0;
    double grad_sq_norm = 1e-31;

    int call_count = 0;
    auto eval_fn = [&](double /*alpha*/) -> double {
        ++call_count;
        return f_current;
    };

    auto result = backtracking_line_search(f_current, grad_sq_norm, eval_fn);

    EXPECT_TRUE(result.success);
    EXPECT_NEAR(result.step_size, 0.0, 1e-30);
    EXPECT_EQ(call_count, 0);
}

// ── Steep gradient requiring backtracking ──────────────────────────

TEST(BacktrackingLineSearch, SteepGradientBacktracks) {
    // f(x) = x^4, grad = 4*x^3. Starting at x = 10.
    // Initial step = 1.0 overshoots badly (x - 4000 = -3990, f > 10^14).
    // The line search must backtrack multiple times.
    double x = 10.0;
    double f_current = std::pow(x, 4);        // 10000
    double grad = 4.0 * std::pow(x, 3);        // 4000
    double grad_sq_norm = grad * grad;          // 16e6

    auto eval_fn = [&](double alpha) -> double {
        double x_new = x - alpha * grad;
        return std::pow(x_new, 4);
    };

    auto result = backtracking_line_search(f_current, grad_sq_norm, eval_fn);

    EXPECT_TRUE(result.success);
    EXPECT_LT(result.objective_value, f_current);
    EXPECT_GT(result.backtracks, 0);
    EXPECT_LT(result.step_size, 1.0);
}

// ── Max backtracks exhausted ───────────────────────────────────────

TEST(BacktrackingLineSearch, MaxBacktracksExhausted) {
    // Pathological objective that never satisfies Armijo:
    // eval_fn always returns a value larger than f_current.
    double f_current = 1.0;
    double grad_sq_norm = 1.0;

    auto eval_fn = [&](double /*alpha*/) -> double {
        return f_current + 1.0;  // Always worse.
    };

    LineSearchConfig config;
    config.max_backtracks = 5;
    config.min_step = 1e-20;  // Set very small so min_step floor does not trigger first.

    auto result = backtracking_line_search(f_current, grad_sq_norm, eval_fn, config);

    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.backtracks, config.max_backtracks);
}

// ── Custom configuration parameters ────────────────────────────────

TEST(BacktrackingLineSearch, CustomInitialStep) {
    // Use a smaller initial step. For f(x) = 0.5*x^2 at x=2,
    // initial_step = 0.25 should be accepted on the first try.
    double x = 2.0;
    double f_current = 0.5 * x * x;
    double grad = x;
    double grad_sq_norm = grad * grad;

    auto eval_fn = [&](double alpha) -> double {
        double x_new = x - alpha * grad;
        return 0.5 * x_new * x_new;
    };

    LineSearchConfig config;
    config.initial_step = 0.25;

    auto result = backtracking_line_search(f_current, grad_sq_norm, eval_fn, config);

    EXPECT_TRUE(result.success);
    EXPECT_NEAR(result.step_size, 0.25, 1e-12);
    EXPECT_EQ(result.backtracks, 0);
}

TEST(BacktrackingLineSearch, CustomShrinkFactor) {
    // Use shrink_factor = 0.1 (aggressive shrinkage).
    // For x^4 at x=10, the required step is small; fewer backtracks
    // are needed with aggressive shrinking.
    double x = 10.0;
    double f_current = std::pow(x, 4);
    double grad = 4.0 * std::pow(x, 3);
    double grad_sq_norm = grad * grad;

    auto eval_fn = [&](double alpha) -> double {
        double x_new = x - alpha * grad;
        return std::pow(x_new, 4);
    };

    LineSearchConfig aggressive;
    aggressive.shrink_factor = 0.1;

    LineSearchConfig standard;
    standard.shrink_factor = 0.5;

    auto result_aggressive = backtracking_line_search(
        f_current, grad_sq_norm, eval_fn, aggressive);
    auto result_standard = backtracking_line_search(
        f_current, grad_sq_norm, eval_fn, standard);

    EXPECT_TRUE(result_aggressive.success);
    EXPECT_TRUE(result_standard.success);
    // Aggressive shrinkage should reach a small step in fewer iterations.
    EXPECT_LE(result_aggressive.backtracks, result_standard.backtracks);
}

TEST(BacktrackingLineSearch, CustomArmijoParameter) {
    // Stricter Armijo condition (c = 0.5) requires more decrease.
    // Looser condition (c = 1e-8) requires less.
    double x = 10.0;
    double f_current = 0.5 * x * x;
    double grad = x;
    double grad_sq_norm = grad * grad;

    auto eval_fn = [&](double alpha) -> double {
        double x_new = x - alpha * grad;
        return 0.5 * x_new * x_new;
    };

    LineSearchConfig strict;
    strict.armijo_c = 0.5;

    LineSearchConfig loose;
    loose.armijo_c = 1e-8;

    auto result_strict = backtracking_line_search(
        f_current, grad_sq_norm, eval_fn, strict);
    auto result_loose = backtracking_line_search(
        f_current, grad_sq_norm, eval_fn, loose);

    EXPECT_TRUE(result_strict.success);
    EXPECT_TRUE(result_loose.success);
    // Strict c should accept a step no larger than the loose one.
    EXPECT_LE(result_strict.step_size, result_loose.step_size + 1e-15);
}

// ── Min step floor ─────────────────────────────────────────────────

TEST(BacktrackingLineSearch, MinStepFloor) {
    // Objective that only decreases for very small steps.
    // f(x) = x^2 but eval_fn returns a large value unless alpha is tiny.
    double f_current = 100.0;
    double grad_sq_norm = 1.0;

    auto eval_fn = [&](double alpha) -> double {
        // Only very small steps produce decrease.
        if (alpha < 1e-8) {
            return f_current - 0.1;  // Small decrease.
        }
        return f_current + 1000.0;  // Large increase otherwise.
    };

    LineSearchConfig config;
    config.min_step = 1e-9;
    config.max_backtracks = 50;

    auto result = backtracking_line_search(f_current, grad_sq_norm, eval_fn, config);

    // Should hit the min_step floor and accept there.
    EXPECT_TRUE(result.success);
    EXPECT_LE(result.step_size, 1e-8);
    EXPECT_LT(result.objective_value, f_current);
}

// ── Multi-dimensional gradient norm ────────────────────────────────

TEST(BacktrackingLineSearch, MultiDimensionalQuadratic) {
    // f(x) = 0.5 * x' * A * x with A = diag(1, 10, 100).
    // x = (1, 1, 1), grad = (1, 10, 100), ||grad||^2 = 10101.
    // The optimal step along the gradient direction is
    // alpha* = ||grad||^2 / (grad' A grad) = 10101 / 100101 ~ 0.1.
    // The initial step = 1.0 should be backtracked.
    double x0 = 1.0, x1 = 1.0, x2 = 1.0;
    double a0 = 1.0, a1 = 10.0, a2 = 100.0;

    double f_current = 0.5 * (a0 * x0 * x0 + a1 * x1 * x1 + a2 * x2 * x2);
    double g0 = a0 * x0, g1 = a1 * x1, g2 = a2 * x2;
    double grad_sq_norm = g0 * g0 + g1 * g1 + g2 * g2;

    auto eval_fn = [&](double alpha) -> double {
        double y0 = x0 - alpha * g0;
        double y1 = x1 - alpha * g1;
        double y2 = x2 - alpha * g2;
        return 0.5 * (a0 * y0 * y0 + a1 * y1 * y1 + a2 * y2 * y2);
    };

    auto result = backtracking_line_search(f_current, grad_sq_norm, eval_fn);

    EXPECT_TRUE(result.success);
    EXPECT_LT(result.objective_value, f_current);
    EXPECT_GT(result.step_size, 0.0);

    // Verify the Armijo condition explicitly.
    double armijo_rhs = f_current - 1e-4 * result.step_size * grad_sq_norm;
    EXPECT_LE(result.objective_value, armijo_rhs + 1e-15);
}

// ── Eval function call count ───────────────────────────────────────

TEST(BacktrackingLineSearch, EvalFnCallCount) {
    // Verify that the number of eval_fn calls matches expectations.
    // For a simple quadratic with initial_step = 1.0 accepted immediately,
    // eval_fn should be called exactly once.
    double x = 4.0;
    double f_current = 0.5 * x * x;
    double grad = x;
    double grad_sq_norm = grad * grad;

    int call_count = 0;
    auto eval_fn = [&](double alpha) -> double {
        ++call_count;
        double x_new = x - alpha * grad;
        return 0.5 * x_new * x_new;
    };

    auto result = backtracking_line_search(f_current, grad_sq_norm, eval_fn);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.backtracks, 0);
    EXPECT_EQ(call_count, 1);  // One evaluation at initial_step.
}
