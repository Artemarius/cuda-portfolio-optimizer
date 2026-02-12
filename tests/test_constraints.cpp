#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "constraints/constraint_set.h"
#include "optimizer/projections.h"

using namespace cpo;

// ══════════════════════════════════════════════════════════════════════
// L1 ball projection tests
// ══════════════════════════════════════════════════════════════════════

TEST(L1BallProjection, AlreadyInBall) {
    VectorXd v(3);
    v << 0.1, -0.1, 0.05;
    VectorXd center = VectorXd::Zero(3);
    double radius = 1.0;  // ||v||_1 = 0.25 < 1.0

    VectorXd w = project_l1_ball(v, center, radius);
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(w(i), v(i), 1e-12);
    }
}

TEST(L1BallProjection, ZeroRadius) {
    VectorXd v(3);
    v << 0.5, 0.3, 0.2;
    VectorXd center(3);
    center << 0.1, 0.2, 0.3;

    VectorXd w = project_l1_ball(v, center, 0.0);
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(w(i), center(i), 1e-12);
    }
}

TEST(L1BallProjection, BasicProjection) {
    // v = [2, 1, -1], center = 0, radius = 1.
    // ||v||_1 = 4 > 1, needs projection.
    VectorXd v(3);
    v << 2.0, 1.0, -1.0;
    VectorXd center = VectorXd::Zero(3);

    VectorXd w = project_l1_ball(v, center, 1.0);

    // Result should have ||w||_1 = 1 and preserve signs.
    EXPECT_NEAR(w.lpNorm<1>(), 1.0, 1e-10);
    EXPECT_GE(w(0), 0.0);  // Same sign as v.
    EXPECT_GE(w(1), 0.0);
    EXPECT_LE(w(2), 0.0);
}

TEST(L1BallProjection, SymmetricInput) {
    VectorXd v(4);
    v << 1.0, 1.0, 1.0, 1.0;
    VectorXd center = VectorXd::Zero(4);

    VectorXd w = project_l1_ball(v, center, 2.0);
    EXPECT_NEAR(w.lpNorm<1>(), 2.0, 1e-10);
    // Symmetric input -> symmetric output.
    for (int i = 1; i < 4; ++i) {
        EXPECT_NEAR(w(i), w(0), 1e-12);
    }
}

TEST(L1BallProjection, NegativeEntries) {
    VectorXd v(3);
    v << -3.0, 2.0, -1.0;
    VectorXd center = VectorXd::Zero(3);

    VectorXd w = project_l1_ball(v, center, 2.0);
    EXPECT_NEAR(w.lpNorm<1>(), 2.0, 1e-10);
    EXPECT_LE(w(0), 0.0);  // Sign preserved.
    EXPECT_GE(w(1), 0.0);
    EXPECT_LE(w(2), 0.0);
}

TEST(L1BallProjection, Idempotent) {
    VectorXd v(3);
    v << 2.0, 1.0, -1.0;
    VectorXd center = VectorXd::Zero(3);

    VectorXd w1 = project_l1_ball(v, center, 1.0);
    VectorXd w2 = project_l1_ball(w1, center, 1.0);

    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(w2(i), w1(i), 1e-12);
    }
}

TEST(L1BallProjection, DimensionMismatch) {
    VectorXd v(3);
    v << 1.0, 2.0, 3.0;
    VectorXd center(2);
    center << 0.0, 0.0;

    EXPECT_THROW(project_l1_ball(v, center, 1.0), std::runtime_error);
}

// ══════════════════════════════════════════════════════════════════════
// Sector projection tests
// ══════════════════════════════════════════════════════════════════════

TEST(SectorProjection, WithinBounds) {
    VectorXd v(4);
    v << 0.2, 0.3, 0.1, 0.4;
    std::vector<Index> indices = {0, 1};  // Sum = 0.5.

    VectorXd w = project_sector(v, indices, 0.3, 0.6);
    // Already within [0.3, 0.6], should be unchanged.
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(w(i), v(i), 1e-12);
    }
}

TEST(SectorProjection, ExceedsMax) {
    VectorXd v(4);
    v << 0.4, 0.3, 0.1, 0.2;
    std::vector<Index> indices = {0, 1};  // Sum = 0.7.

    VectorXd w = project_sector(v, indices, 0.0, 0.5);
    // Sector sum should be reduced to 0.5.
    double sector_sum = w(0) + w(1);
    EXPECT_NEAR(sector_sum, 0.5, 1e-10);
    // Non-sector elements unchanged.
    EXPECT_NEAR(w(2), v(2), 1e-12);
    EXPECT_NEAR(w(3), v(3), 1e-12);
}

TEST(SectorProjection, BelowMin) {
    VectorXd v(4);
    v << 0.05, 0.05, 0.4, 0.5;
    std::vector<Index> indices = {0, 1};  // Sum = 0.10.

    VectorXd w = project_sector(v, indices, 0.2, 0.5);
    double sector_sum = w(0) + w(1);
    EXPECT_NEAR(sector_sum, 0.2, 1e-10);
    // Non-sector elements unchanged.
    EXPECT_NEAR(w(2), v(2), 1e-12);
    EXPECT_NEAR(w(3), v(3), 1e-12);
}

TEST(SectorProjection, EmptySector) {
    VectorXd v(3);
    v << 0.3, 0.3, 0.4;
    std::vector<Index> indices = {};

    VectorXd w = project_sector(v, indices, 0.0, 0.5);
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(w(i), v(i), 1e-12);
    }
}

// ══════════════════════════════════════════════════════════════════════
// ConstraintSet tests
// ══════════════════════════════════════════════════════════════════════

TEST(ConstraintSet, EmptyFeasible) {
    ConstraintSet cs;
    VectorXd w(3);
    w << 0.3, 0.3, 0.4;
    EXPECT_TRUE(cs.is_feasible(w));
}

TEST(ConstraintSet, SimplexViolation) {
    ConstraintSet cs;
    VectorXd w(3);
    w << 0.5, 0.5, 0.5;  // Sum = 1.5.
    EXPECT_FALSE(cs.is_feasible(w));
}

TEST(ConstraintSet, BoxViolation) {
    ConstraintSet cs;
    cs.has_position_limits = true;
    cs.position_limits.w_min = VectorXd::Zero(3);
    cs.position_limits.w_max = VectorXd::Constant(3, 0.4);

    VectorXd w(3);
    w << 0.5, 0.3, 0.2;  // w[0] = 0.5 > 0.4.
    EXPECT_FALSE(cs.is_feasible(w));
}

TEST(ConstraintSet, TurnoverViolation) {
    ConstraintSet cs;
    cs.has_turnover = true;
    cs.turnover.w_prev = VectorXd::Constant(3, 1.0 / 3.0);
    cs.turnover.tau = 0.1;

    VectorXd w(3);
    w << 0.5, 0.3, 0.2;  // ||w - w_prev||_1 = |0.167|+|-.033|+|-.133| = 0.333 > 0.1
    EXPECT_FALSE(cs.is_feasible(w));
}

TEST(ConstraintSet, SectorViolation) {
    ConstraintSet cs;
    cs.has_sector_constraints = true;
    SectorBound sb;
    sb.name = "Tech";
    sb.assets = {0, 1};
    sb.max_exposure = 0.4;
    cs.sector_constraints.sectors.push_back(sb);

    VectorXd w(3);
    w << 0.3, 0.3, 0.4;  // Sector sum = 0.6 > 0.4.
    EXPECT_FALSE(cs.is_feasible(w));
}

TEST(ConstraintSet, AllFeasible) {
    ConstraintSet cs;
    cs.has_position_limits = true;
    cs.position_limits.w_min = VectorXd::Zero(3);
    cs.position_limits.w_max = VectorXd::Constant(3, 0.5);

    cs.has_turnover = true;
    cs.turnover.w_prev = VectorXd::Constant(3, 1.0 / 3.0);
    cs.turnover.tau = 1.0;

    cs.has_sector_constraints = true;
    SectorBound sb;
    sb.name = "Sector1";
    sb.assets = {0, 1};
    sb.max_exposure = 0.7;
    cs.sector_constraints.sectors.push_back(sb);

    VectorXd w(3);
    w << 0.4, 0.3, 0.3;  // Sum=1, all in [0,0.5], sector=0.7, turnover ok.
    EXPECT_TRUE(cs.is_feasible(w));
}

// ══════════════════════════════════════════════════════════════════════
// Validation tests
// ══════════════════════════════════════════════════════════════════════

TEST(Validation, DimensionMismatch) {
    ConstraintSet cs;
    cs.has_position_limits = true;
    cs.position_limits.w_min = VectorXd::Zero(2);
    cs.position_limits.w_max = VectorXd::Constant(2, 0.5);

    EXPECT_THROW(cs.validate(3), std::runtime_error);
}

TEST(Validation, InfeasibleBounds) {
    ConstraintSet cs;
    cs.has_position_limits = true;
    cs.position_limits.w_min = VectorXd::Constant(3, 0.5);  // Sum = 1.5 > 1.
    cs.position_limits.w_max = VectorXd::Constant(3, 1.0);

    EXPECT_THROW(cs.validate(3), std::runtime_error);
}

TEST(Validation, NumConstraintSets) {
    ConstraintSet cs;
    EXPECT_EQ(cs.num_constraint_sets(), 1);  // Simplex only.

    cs.has_position_limits = true;
    EXPECT_EQ(cs.num_constraint_sets(), 2);

    cs.has_turnover = true;
    EXPECT_EQ(cs.num_constraint_sets(), 3);

    cs.has_sector_constraints = true;
    cs.sector_constraints.sectors.push_back({"A", {0}, 0.0, 0.5});
    cs.sector_constraints.sectors.push_back({"B", {1}, 0.0, 0.5});
    EXPECT_EQ(cs.num_constraint_sets(), 5);  // 1 + 1 + 1 + 2
}

// ══════════════════════════════════════════════════════════════════════
// Generalized Dykstra's projection tests
// ══════════════════════════════════════════════════════════════════════

TEST(DykstraProjection, SimplexOnlyMatchesProjectSimplex) {
    VectorXd v(4);
    v << 0.5, -0.1, 0.8, 0.3;

    ConstraintSet cs;  // No extra constraints -> simplex only.

    VectorXd w_dykstra = project_constraints(v, cs);
    VectorXd w_simplex = project_simplex(v);

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(w_dykstra(i), w_simplex(i), 1e-8);
    }
}

TEST(DykstraProjection, SimplexBoxFeasible) {
    // Verify that project_constraints with simplex + box produces a
    // feasible point: sum = 1, w >= 0, lb <= w <= ub.
    VectorXd v(3);
    v << 0.7, 0.5, -0.1;
    VectorXd lb = VectorXd::Constant(3, 0.1);
    VectorXd ub = VectorXd::Constant(3, 0.5);

    ConstraintSet cs;
    cs.has_position_limits = true;
    cs.position_limits.w_min = lb;
    cs.position_limits.w_max = ub;

    VectorXd w = project_constraints(v, cs);

    // Simplex constraint.
    EXPECT_NEAR(w.sum(), 1.0, 1e-6);
    for (int i = 0; i < 3; ++i) {
        EXPECT_GE(w(i), -1e-8);
    }
    // Box constraint.
    for (int i = 0; i < 3; ++i) {
        EXPECT_GE(w(i), lb(i) - 1e-6) << "w(" << i << ") below lb";
        EXPECT_LE(w(i), ub(i) + 1e-6) << "w(" << i << ") above ub";
    }
    EXPECT_TRUE(cs.is_feasible(w, 1e-4));
}

TEST(DykstraProjection, TurnoverZeroForcesWPrev) {
    VectorXd v(3);
    v << 0.5, 0.3, 0.2;

    ConstraintSet cs;
    cs.has_turnover = true;
    cs.turnover.w_prev = VectorXd::Constant(3, 1.0 / 3.0);
    cs.turnover.tau = 0.0;  // No trading allowed.

    VectorXd w = project_constraints(v, cs);

    // Should be forced to w_prev (which is on the simplex).
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(w(i), 1.0 / 3.0, 1e-4);
    }
}

TEST(DykstraProjection, LargeTauInactive) {
    VectorXd v(3);
    v << 0.7, 0.5, -0.1;

    ConstraintSet cs;
    cs.has_turnover = true;
    cs.turnover.w_prev = VectorXd::Constant(3, 1.0 / 3.0);
    cs.turnover.tau = 2.0;  // Unconstrained turnover.

    VectorXd w_constrained = project_constraints(v, cs);
    VectorXd w_simplex = project_simplex(v);

    // With tau = 2.0, turnover is never binding, so result matches simplex.
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(w_constrained(i), w_simplex(i), 1e-6);
    }
}

TEST(DykstraProjection, SectorBoundsRespected) {
    VectorXd v(4);
    v << 0.6, 0.3, 0.05, 0.05;

    ConstraintSet cs;
    cs.has_sector_constraints = true;
    SectorBound sb;
    sb.name = "Concentrated";
    sb.assets = {0, 1};
    sb.min_exposure = 0.0;
    sb.max_exposure = 0.5;  // Max 50% in sector {0,1}.
    cs.sector_constraints.sectors.push_back(sb);

    VectorXd w = project_constraints(v, cs);

    // Sector sum should be <= 0.5.
    double sector_sum = w(0) + w(1);
    EXPECT_LE(sector_sum, 0.5 + 1e-4);
    // Simplex satisfied.
    EXPECT_NEAR(w.sum(), 1.0, 1e-4);
    for (int i = 0; i < 4; ++i) {
        EXPECT_GE(w(i), -1e-4);
    }
}

TEST(DykstraProjection, AllConstraintsCombined) {
    VectorXd v(4);
    v << 0.8, 0.3, -0.1, 0.0;

    ConstraintSet cs;

    // Box constraints.
    cs.has_position_limits = true;
    cs.position_limits.w_min = VectorXd::Zero(4);
    cs.position_limits.w_max = VectorXd::Constant(4, 0.5);

    // Turnover constraint.
    cs.has_turnover = true;
    cs.turnover.w_prev = VectorXd::Constant(4, 0.25);
    cs.turnover.tau = 0.8;

    // Sector constraint.
    cs.has_sector_constraints = true;
    SectorBound sb;
    sb.name = "Sector1";
    sb.assets = {0, 1};
    sb.max_exposure = 0.6;
    cs.sector_constraints.sectors.push_back(sb);

    VectorXd w = project_constraints(v, cs);

    // Check all constraints are satisfied.
    EXPECT_TRUE(cs.is_feasible(w, 1e-3))
        << "w = [" << w.transpose() << "], sum=" << w.sum()
        << ", turnover=" << (w - cs.turnover.w_prev).lpNorm<1>()
        << ", sector=" << w(0) + w(1);
}
