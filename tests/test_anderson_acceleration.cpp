#include <gtest/gtest.h>

#include <cmath>
#include <functional>

#include <Eigen/QR>

#include "optimizer/anderson_acceleration.h"

using namespace cpo;

// ── Helper: run fixed-point iteration with optional Anderson acceleration ──

/// Run a fixed-point iteration x_{k+1} = G(x_k) for at most max_iter steps.
/// Returns the number of iterations to reach ||G(x) - x|| < tol.
/// If not converged, returns max_iter.
static int run_fixed_point(std::function<VectorXd(const VectorXd&)> G,
                           VectorXd x0,
                           int max_iter,
                           double tol,
                           bool use_anderson,
                           int anderson_depth = 5) {
    const int n = static_cast<int>(x0.size());
    VectorXd x = x0;

    AndersonAccelerator anderson(n, anderson_depth);

    for (int k = 0; k < max_iter; ++k) {
        VectorXd g = G(x);
        double residual = (g - x).norm();

        if (residual < tol) {
            return k;
        }

        if (use_anderson) {
            x = anderson.accelerate(x, g);
        } else {
            x = g;
        }
    }
    return max_iter;
}

// ── Linear contraction map ──────────────────────────────────────────

TEST(AndersonAcceleration, LinearContractionConvergesFaster) {
    // Fixed-point map: G(x) = A*x + b, where A is a contraction (||A|| < 1).
    // The fixed point satisfies x* = (I - A)^{-1} b.
    //
    // Anderson acceleration should converge in fewer iterations than
    // plain fixed-point iteration for linear maps.
    const int n = 5;

    // Construct a contraction matrix with spectral radius < 1.
    // A = 0.8 * I + small off-diagonal perturbation.
    // Scale to ensure ||A|| < 1.
    MatrixXd A = MatrixXd::Zero(n, n);
    for (int i = 0; i < n; ++i) {
        A(i, i) = 0.5;  // diagonal dominance
        if (i + 1 < n) A(i, i + 1) = 0.1;
        if (i - 1 >= 0) A(i, i - 1) = 0.1;
    }

    VectorXd b = VectorXd::Ones(n);

    // Fixed-point map: G(x) = A*x + b.
    auto G = [&](const VectorXd& x) -> VectorXd {
        return A * x + b;
    };

    VectorXd x0 = VectorXd::Zero(n);
    const int max_iter = 500;
    const double tol = 1e-10;

    int iters_plain = run_fixed_point(G, x0, max_iter, tol,
                                       /*use_anderson=*/false);
    int iters_anderson = run_fixed_point(G, x0, max_iter, tol,
                                          /*use_anderson=*/true,
                                          /*anderson_depth=*/5);

    // Anderson should converge faster (or at least not slower).
    // For a linear map of dimension n with depth >= n, Anderson
    // terminates in at most n+1 steps (it solves the linear system).
    EXPECT_LT(iters_anderson, iters_plain)
        << "Anderson (" << iters_anderson << " iters) should beat plain ("
        << iters_plain << " iters) on a linear contraction";

    // Verify convergence: check that the iterates reach the fixed point.
    // x* = (I - A)^{-1} b
    MatrixXd I_minus_A = MatrixXd::Identity(n, n) - A;
    VectorXd x_star = I_minus_A.colPivHouseholderQr().solve(b);

    // Re-run Anderson to get final iterate.
    VectorXd x = VectorXd::Zero(n);
    AndersonAccelerator aa(n, 5);
    for (int k = 0; k < iters_anderson + 1; ++k) {
        VectorXd g = G(x);
        x = aa.accelerate(x, g);
    }
    EXPECT_LT((x - x_star).norm(), 1e-8)
        << "Anderson iterate did not reach the fixed point";
}

// ── Quadratic fixed-point with known solution ───────────────────────

TEST(AndersonAcceleration, QuadraticFixedPoint) {
    // Newton-like iteration for solving x^2 = c (scalar, embedded in R^1).
    // G(x) = 0.5 * (x + c/x)  — Babylonian method for sqrt(c).
    // Fixed point: x* = sqrt(c).
    const double c = 7.0;

    auto G = [c](const VectorXd& x) -> VectorXd {
        VectorXd g(1);
        g(0) = 0.5 * (x(0) + c / x(0));
        return g;
    };

    VectorXd x0(1);
    x0(0) = 1.0;  // initial guess

    const int max_iter = 100;
    const double tol = 1e-14;

    int iters_anderson = run_fixed_point(G, x0, max_iter, tol,
                                          /*use_anderson=*/true,
                                          /*anderson_depth=*/3);

    // Should converge (Babylonian method converges quadratically anyway,
    // but Anderson should not break it).
    EXPECT_LT(iters_anderson, max_iter)
        << "Anderson did not converge on Babylonian method";

    // Verify solution.
    VectorXd x = x0;
    AndersonAccelerator aa(1, 3);
    for (int k = 0; k <= iters_anderson; ++k) {
        VectorXd g = G(x);
        x = aa.accelerate(x, g);
    }
    EXPECT_NEAR(x(0), std::sqrt(c), 1e-12)
        << "Did not converge to sqrt(" << c << ")";
}

// ── Depth=1 degrades gracefully ─────────────────────────────────────

TEST(AndersonAcceleration, DepthOneDegrades) {
    // With depth=1, Anderson can use at most 1 residual difference.
    // It should still work, just with less acceleration.
    const int n = 3;

    MatrixXd A = 0.6 * MatrixXd::Identity(n, n);
    VectorXd b = VectorXd::Ones(n) * 2.0;

    auto G = [&](const VectorXd& x) -> VectorXd {
        return A * x + b;
    };

    VectorXd x0 = VectorXd::Zero(n);
    const int max_iter = 200;
    const double tol = 1e-10;

    // Depth=1 should still converge.
    int iters_depth1 = run_fixed_point(G, x0, max_iter, tol,
                                        /*use_anderson=*/true,
                                        /*anderson_depth=*/1);

    EXPECT_LT(iters_depth1, max_iter)
        << "Anderson with depth=1 did not converge";

    // Verify it reaches the fixed point.
    MatrixXd I_minus_A = MatrixXd::Identity(n, n) - A;
    VectorXd x_star = I_minus_A.colPivHouseholderQr().solve(b);

    VectorXd x = VectorXd::Zero(n);
    AndersonAccelerator aa(n, 1);
    for (int k = 0; k <= iters_depth1; ++k) {
        VectorXd g = G(x);
        x = aa.accelerate(x, g);
    }
    EXPECT_LT((x - x_star).norm(), 1e-8);
}

// ── Restart mechanism ───────────────────────────────────────────────

TEST(AndersonAcceleration, RestartTriggersOnBadExtrapolation) {
    // Verify should_restart returns true when accelerated residual
    // is worse than unaccelerated residual.
    AndersonAccelerator aa(3, 5, /*safeguard=*/1.0);

    // Accelerated residual worse (larger) -> should restart.
    EXPECT_TRUE(aa.should_restart(2.0, 1.0));

    // Accelerated residual equal -> should not restart (not strictly greater).
    EXPECT_FALSE(aa.should_restart(1.0, 1.0));

    // Accelerated residual better -> should not restart.
    EXPECT_FALSE(aa.should_restart(0.5, 1.0));
}

TEST(AndersonAcceleration, RestartWithCustomSafeguardFactor) {
    // With safeguard_factor=2.0, only restart if accelerated residual
    // is more than 2x the unaccelerated residual.
    AndersonAccelerator aa(3, 5, /*safeguard=*/2.0);

    // 1.5 < 2.0 * 1.0 -> should NOT restart.
    EXPECT_FALSE(aa.should_restart(1.5, 1.0));

    // 2.5 > 2.0 * 1.0 -> should restart.
    EXPECT_TRUE(aa.should_restart(2.5, 1.0));

    // Exactly 2.0 * 1.0 -> boundary, should NOT restart (not strictly greater).
    EXPECT_FALSE(aa.should_restart(2.0, 1.0));
}

TEST(AndersonAcceleration, RestartAndRecoverOnDivergentMap) {
    // Apply Anderson to a map that sometimes produces bad extrapolations.
    // After restart (reset), convergence should resume.
    const int n = 2;

    // Slowly contracting map.
    MatrixXd A = 0.9 * MatrixXd::Identity(n, n);
    VectorXd b = VectorXd::Ones(n);

    auto G = [&](const VectorXd& x) -> VectorXd {
        return A * x + b;
    };

    AndersonAccelerator aa(n, 5, /*safeguard=*/1.0);

    VectorXd x = VectorXd::Zero(n);

    // Run some iterations, then simulate a reset (as if a bad extrapolation
    // was detected), and verify continued convergence.
    for (int k = 0; k < 10; ++k) {
        VectorXd g = G(x);
        x = aa.accelerate(x, g);
    }

    // Force a reset (simulating restart after bad extrapolation).
    aa.reset();

    // Continue iterating — should still converge.
    for (int k = 0; k < 200; ++k) {
        VectorXd g = G(x);
        double residual = (g - x).norm();
        if (residual < 1e-10) {
            break;
        }
        x = aa.accelerate(x, g);
    }

    // Verify convergence to fixed point: x* = (I - A)^{-1} b = 10 * ones.
    MatrixXd I_minus_A = MatrixXd::Identity(n, n) - A;
    VectorXd x_star = I_minus_A.colPivHouseholderQr().solve(b);
    EXPECT_LT((x - x_star).norm(), 1e-8)
        << "Did not converge after restart";
}

// ── Dimension mismatch throws ───────────────────────────────────────

TEST(AndersonAcceleration, DimensionMismatchThrows) {
    AndersonAccelerator aa(3, 5);

    VectorXd x_ok(3);
    x_ok << 1.0, 2.0, 3.0;
    VectorXd g_ok(3);
    g_ok << 0.5, 1.0, 1.5;

    // Correct dimensions should not throw.
    EXPECT_NO_THROW(aa.accelerate(x_ok, g_ok));

    // Wrong dimension for x.
    VectorXd x_bad(4);
    x_bad << 1.0, 2.0, 3.0, 4.0;
    EXPECT_THROW(aa.accelerate(x_bad, g_ok), std::invalid_argument);

    // Wrong dimension for g.
    VectorXd g_bad(2);
    g_bad << 1.0, 2.0;
    EXPECT_THROW(aa.accelerate(x_ok, g_bad), std::invalid_argument);
}

// ── Reset clears history ────────────────────────────────────────────

TEST(AndersonAcceleration, ResetClearsHistory) {
    const int n = 2;

    MatrixXd A = 0.5 * MatrixXd::Identity(n, n);
    VectorXd b = VectorXd::Ones(n);

    auto G = [&](const VectorXd& x) -> VectorXd {
        return A * x + b;
    };

    AndersonAccelerator aa(n, 5);

    VectorXd x = VectorXd::Zero(n);

    // Feed several iterates to build history.
    for (int k = 0; k < 5; ++k) {
        VectorXd g = G(x);
        x = aa.accelerate(x, g);
    }

    // Reset should clear all history.
    aa.reset();

    // After reset, the first call should return g unchanged
    // (insufficient history).
    VectorXd x_new(n);
    x_new << 1.0, 1.0;
    VectorXd g_new = G(x_new);
    VectorXd result = aa.accelerate(x_new, g_new);

    // With no history (first call after reset), accelerate returns g.
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(result(i), g_new(i), 1e-15)
            << "After reset, first accelerate() should return g unchanged";
    }
}

// ── Constructor validation ──────────────────────────────────────────

TEST(AndersonAcceleration, InvalidConstructorArgs) {
    // Zero dimension.
    EXPECT_THROW(AndersonAccelerator(0, 5), std::invalid_argument);

    // Negative dimension.
    EXPECT_THROW(AndersonAccelerator(-1, 5), std::invalid_argument);

    // Zero depth.
    EXPECT_THROW(AndersonAccelerator(3, 0), std::invalid_argument);

    // Negative depth.
    EXPECT_THROW(AndersonAccelerator(3, -1), std::invalid_argument);

    // Valid construction.
    EXPECT_NO_THROW(AndersonAccelerator(1, 1));
    EXPECT_NO_THROW(AndersonAccelerator(100, 10));
}

// ── First call returns g unchanged ──────────────────────────────────

TEST(AndersonAcceleration, FirstCallReturnsG) {
    // With only one stored iterate, there are no residual differences
    // to form, so the accelerator should return g directly.
    const int n = 4;
    AndersonAccelerator aa(n, 5);

    VectorXd x(n);
    x << 1.0, 2.0, 3.0, 4.0;
    VectorXd g(n);
    g << 0.5, 1.5, 2.5, 3.5;

    VectorXd result = aa.accelerate(x, g);

    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(result(i), g(i), 1e-15)
            << "First call should return g at index " << i;
    }
}

// ── High-dimensional linear system ──────────────────────────────────

TEST(AndersonAcceleration, HighDimensionalLinear) {
    // Verify Anderson acceleration works on a larger linear system.
    // G(x) = (I - D^{-1} A) x + D^{-1} b, a Jacobi-style iteration
    // for solving Ax = b where A = D + L + U (diagonal-dominant).
    const int n = 20;

    // Build a diagonally dominant matrix.
    MatrixXd A = MatrixXd::Random(n, n) * 0.1;
    for (int i = 0; i < n; ++i) {
        A(i, i) = 3.0;  // strong diagonal dominance
    }
    A = (A + A.transpose()) / 2.0;  // symmetrize

    VectorXd b = VectorXd::Random(n);

    // Jacobi iteration matrix: M = I - D^{-1} A
    VectorXd d_inv = A.diagonal().cwiseInverse();
    MatrixXd D_inv = d_inv.asDiagonal();
    MatrixXd M = MatrixXd::Identity(n, n) - D_inv * A;
    VectorXd c = D_inv * b;

    auto G = [&](const VectorXd& x) -> VectorXd {
        return M * x + c;
    };

    VectorXd x0 = VectorXd::Zero(n);
    const int max_iter = 500;
    const double tol = 1e-10;

    int iters_plain = run_fixed_point(G, x0, max_iter, tol,
                                       /*use_anderson=*/false);
    int iters_anderson = run_fixed_point(G, x0, max_iter, tol,
                                          /*use_anderson=*/true,
                                          /*anderson_depth=*/5);

    // Both should converge.
    EXPECT_LT(iters_plain, max_iter) << "Plain iteration did not converge";
    EXPECT_LT(iters_anderson, max_iter) << "Anderson did not converge";

    // Anderson should be faster.
    EXPECT_LT(iters_anderson, iters_plain)
        << "Anderson (" << iters_anderson << ") should beat plain ("
        << iters_plain << ") on Jacobi iteration";
}

// ── Ring buffer wrapping ────────────────────────────────────────────

TEST(AndersonAcceleration, RingBufferWrapsCorrectly) {
    // Use a very small depth (m=2) and run many iterations to exercise
    // the ring buffer eviction logic.
    const int n = 3;
    const int depth = 2;

    MatrixXd A = 0.4 * MatrixXd::Identity(n, n);
    VectorXd b = VectorXd::Ones(n) * 3.0;

    auto G = [&](const VectorXd& x) -> VectorXd {
        return A * x + b;
    };

    AndersonAccelerator aa(n, depth);
    VectorXd x = VectorXd::Zero(n);

    // Run 50 iterations with depth=2 to exercise wrapping many times.
    for (int k = 0; k < 50; ++k) {
        VectorXd g = G(x);
        x = aa.accelerate(x, g);
    }

    // Should converge to x* = (I - A)^{-1} b = 5 * ones.
    VectorXd x_star = VectorXd::Constant(n, 5.0);
    EXPECT_LT((x - x_star).norm(), 1e-8)
        << "Ring buffer wrapping did not affect convergence";
}
