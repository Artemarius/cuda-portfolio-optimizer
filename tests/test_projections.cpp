#include <gtest/gtest.h>

#include <cmath>

#include "optimizer/projections.h"

using namespace cpo;

// ── Simplex projection tests ────────────────────────────────────────

TEST(SimplexProjection, AlreadyOnSimplex) {
    // A point already on the simplex should be unchanged.
    VectorXd v(3);
    v << 0.2, 0.3, 0.5;

    auto w = project_simplex(v);

    EXPECT_NEAR(w.sum(), 1.0, 1e-12);
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(w(i), v(i), 1e-12) << "index " << i;
    }
}

TEST(SimplexProjection, UniformVector) {
    // All-ones vector: sum = n. Should project to 1/n each.
    VectorXd v = VectorXd::Ones(5);
    auto w = project_simplex(v);

    EXPECT_NEAR(w.sum(), 1.0, 1e-12);
    for (int i = 0; i < 5; ++i) {
        EXPECT_NEAR(w(i), 0.2, 1e-12) << "index " << i;
    }
}

TEST(SimplexProjection, NegativeEntries) {
    // Some negative entries should be clamped to zero.
    VectorXd v(3);
    v << -1.0, 0.5, 2.0;

    auto w = project_simplex(v);

    EXPECT_NEAR(w.sum(), 1.0, 1e-12);
    for (int i = 0; i < 3; ++i) {
        EXPECT_GE(w(i), -1e-15) << "w(" << i << ") = " << w(i);
    }
    // The -1.0 entry should become 0.
    EXPECT_NEAR(w(0), 0.0, 1e-12);
}

TEST(SimplexProjection, SingleElement) {
    VectorXd v(1);
    v << 5.0;
    auto w = project_simplex(v);
    EXPECT_NEAR(w(0), 1.0, 1e-12);
}

TEST(SimplexProjection, AllNegative) {
    // All negative: only the largest should be 1.
    VectorXd v(3);
    v << -3.0, -1.0, -2.0;

    auto w = project_simplex(v);

    EXPECT_NEAR(w.sum(), 1.0, 1e-12);
    for (int i = 0; i < 3; ++i) {
        EXPECT_GE(w(i), -1e-15) << "w(" << i << ") = " << w(i);
    }
}

TEST(SimplexProjection, Idempotent) {
    // Projecting twice should give the same result.
    VectorXd v(4);
    v << 0.7, -0.3, 0.4, 0.8;

    auto w1 = project_simplex(v);
    auto w2 = project_simplex(w1);

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(w1(i), w2(i), 1e-12) << "index " << i;
    }
}

TEST(SimplexProjection, LargeVector) {
    // Stress test: 100 elements.
    VectorXd v = VectorXd::Random(100);
    auto w = project_simplex(v);

    EXPECT_NEAR(w.sum(), 1.0, 1e-10);
    for (int i = 0; i < 100; ++i) {
        EXPECT_GE(w(i), -1e-15) << "w(" << i << ") = " << w(i);
    }
}

// ── Box projection tests ───────────────────────────────────────────

TEST(BoxProjection, WithinBounds) {
    VectorXd v(3);
    v << 0.2, 0.3, 0.5;
    VectorXd lb = VectorXd::Zero(3);
    VectorXd ub = VectorXd::Ones(3);

    auto w = project_box(v, lb, ub);

    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(w(i), v(i), 1e-12) << "index " << i;
    }
}

TEST(BoxProjection, ClampToLower) {
    VectorXd v(3);
    v << -0.5, 0.3, 0.1;
    VectorXd lb(3);
    lb << 0.0, 0.0, 0.0;
    VectorXd ub(3);
    ub << 1.0, 1.0, 1.0;

    auto w = project_box(v, lb, ub);
    EXPECT_NEAR(w(0), 0.0, 1e-12);  // clamped
    EXPECT_NEAR(w(1), 0.3, 1e-12);
    EXPECT_NEAR(w(2), 0.1, 1e-12);
}

TEST(BoxProjection, ClampToUpper) {
    VectorXd v(3);
    v << 0.3, 0.8, 0.1;
    VectorXd lb = VectorXd::Zero(3);
    VectorXd ub = VectorXd::Constant(3, 0.5);

    auto w = project_box(v, lb, ub);
    EXPECT_NEAR(w(0), 0.3, 1e-12);
    EXPECT_NEAR(w(1), 0.5, 1e-12);  // clamped
    EXPECT_NEAR(w(2), 0.1, 1e-12);
}

TEST(BoxProjection, DimensionMismatch) {
    VectorXd v(3);
    v << 0.1, 0.2, 0.3;
    VectorXd lb = VectorXd::Zero(2);
    VectorXd ub = VectorXd::Ones(3);

    EXPECT_THROW(project_box(v, lb, ub), std::runtime_error);
}

TEST(BoxProjection, Idempotent) {
    VectorXd v(3);
    v << -0.5, 0.8, 0.3;
    VectorXd lb = VectorXd::Constant(3, 0.1);
    VectorXd ub = VectorXd::Constant(3, 0.5);

    auto w1 = project_box(v, lb, ub);
    auto w2 = project_box(w1, lb, ub);

    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(w1(i), w2(i), 1e-12) << "index " << i;
    }
}

// ── Combined simplex + box tests ───────────────────────────────────

TEST(SimplexBoxProjection, SimplexWithWideBounds) {
    // Box bounds are wider than simplex — should match pure simplex.
    VectorXd v(3);
    v << 0.7, -0.3, 0.8;
    VectorXd lb = VectorXd::Constant(3, -10.0);
    VectorXd ub = VectorXd::Constant(3, 10.0);

    auto w_simplex = project_simplex(v);
    auto w_combined = project_simplex_box(v, lb, ub);

    EXPECT_NEAR(w_combined.sum(), 1.0, 1e-8);
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(w_combined(i), w_simplex(i), 1e-6) << "index " << i;
    }
}

TEST(SimplexBoxProjection, WithPositionLimits) {
    // Max 40% per asset, 3 assets. Must have sum=1, 0<=w<=0.4.
    VectorXd v(3);
    v << 0.8, 0.1, 0.1;
    VectorXd lb = VectorXd::Zero(3);
    VectorXd ub = VectorXd::Constant(3, 0.4);

    auto w = project_simplex_box(v, lb, ub);

    EXPECT_NEAR(w.sum(), 1.0, 1e-6);
    for (int i = 0; i < 3; ++i) {
        EXPECT_GE(w(i), -1e-8) << "w(" << i << ") = " << w(i);
        EXPECT_LE(w(i), 0.4 + 1e-8) << "w(" << i << ") = " << w(i);
    }
}

TEST(SimplexBoxProjection, FeasiblePointUnchanged) {
    // A point that satisfies both simplex and box should be unchanged.
    VectorXd v(4);
    v << 0.25, 0.25, 0.25, 0.25;
    VectorXd lb = VectorXd::Constant(4, 0.1);
    VectorXd ub = VectorXd::Constant(4, 0.4);

    auto w = project_simplex_box(v, lb, ub);

    EXPECT_NEAR(w.sum(), 1.0, 1e-10);
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(w(i), 0.25, 1e-6) << "index " << i;
    }
}

TEST(SimplexBoxProjection, TightBounds) {
    // Very tight bounds: forces specific allocation.
    // 3 assets with bounds [0.3, 0.35] each. Sum must be 1.
    // Feasible: w = [0.3, 0.35, 0.35] or similar.
    VectorXd v(3);
    v << 0.5, 0.5, 0.0;
    VectorXd lb = VectorXd::Constant(3, 0.3);
    VectorXd ub = VectorXd::Constant(3, 0.35);

    auto w = project_simplex_box(v, lb, ub);

    EXPECT_NEAR(w.sum(), 1.0, 1e-5);
    for (int i = 0; i < 3; ++i) {
        EXPECT_GE(w(i), 0.3 - 1e-6) << "w(" << i << ") = " << w(i);
        EXPECT_LE(w(i), 0.35 + 1e-6) << "w(" << i << ") = " << w(i);
    }
}
