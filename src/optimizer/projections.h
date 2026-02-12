#pragma once

/// @file projections.h
/// @brief Projection operators for convex constraint sets.
///
/// Implements projections used in the ADMM z-update step:
///   - Simplex projection: {w : 1'w = 1, w >= 0}
///   - Box projection: {w : w_min <= w <= w_max}
///   - Combined simplex + box via Dykstra's alternating projection
///
/// References:
///   Duchi et al., "Efficient Projections onto the l1-Ball for Learning
///   in High Dimensions", ICML 2008 — simplex projection algorithm.
///
///   Boyle & Dykstra, "A Method for Finding Projections onto the
///   Intersection of Convex Sets in Hilbert Spaces", 1986.

#include "constraints/constraint_set.h"
#include "core/types.h"

namespace cpo {

/// Project a vector onto the probability simplex {w : 1'w = 1, w >= 0}.
///
/// Algorithm: O(n log n) sorting-based method from Duchi et al. 2008.
///   1. Sort v in descending order: u_1 >= u_2 >= ... >= u_n
///   2. Find rho = max{j : u_j - (1/j)(sum_{i=1}^{j} u_i - 1) > 0}
///   3. theta = (1/rho)(sum_{i=1}^{rho} u_i - 1)
///   4. w_i = max(v_i - theta, 0)
///
/// @param v Input vector (n-dimensional).
/// @return Projected vector on the simplex.
VectorXd project_simplex(const VectorXd& v);

/// Project a vector onto a box constraint set {w : lb <= w <= ub}.
///
/// Simple element-wise clamping: w_i = clamp(v_i, lb_i, ub_i).
///
/// @param v Input vector.
/// @param lb Lower bounds (element-wise).
/// @param ub Upper bounds (element-wise).
/// @return Projected vector within the box.
VectorXd project_box(const VectorXd& v, const VectorXd& lb, const VectorXd& ub);

/// Project onto the intersection of the simplex and a box constraint.
///
/// Uses Dykstra's alternating projection algorithm to find the nearest
/// point in {w : 1'w = 1, w >= 0} ∩ {w : lb <= w <= ub}.
///
/// Convergence is guaranteed for the intersection of convex sets.
/// Typically converges in 10-50 iterations.
///
/// @param v Input vector.
/// @param lb Lower bounds (element-wise).
/// @param ub Upper bounds (element-wise).
/// @param max_iter Maximum Dykstra iterations (default: 100).
/// @param tol Convergence tolerance on increment norm (default: 1e-10).
/// @return Projected vector in the intersection.
VectorXd project_simplex_box(const VectorXd& v, const VectorXd& lb,
                              const VectorXd& ub, int max_iter = 100,
                              double tol = 1e-10);

/// Project onto an L1 ball: {v : ||v - center||_1 <= radius}.
///
/// Uses the Duchi et al. 2008 approach: shift to center, project onto
/// the scaled L1 ball via soft-thresholding with sign preservation,
/// then shift back. O(n log n) from the sorting step.
///
/// @param v Input vector.
/// @param center Center of the L1 ball.
/// @param radius Radius of the L1 ball (must be >= 0).
/// @return Projected vector.
VectorXd project_l1_ball(const VectorXd& v, const VectorXd& center,
                          ScalarCPU radius);

/// Project a vector so that the sum of specified sector indices
/// lies within [s_min, s_max].
///
/// If the sector sum is within bounds, returns v unchanged.
/// Otherwise, uniformly adjusts the sector elements to reach the
/// nearest bound while preserving relative proportions.
///
/// @param v Input vector (full portfolio).
/// @param indices Asset indices belonging to the sector.
/// @param s_min Minimum sector exposure.
/// @param s_max Maximum sector exposure.
/// @return Projected vector with sector sum in [s_min, s_max].
VectorXd project_sector(const VectorXd& v, const std::vector<Index>& indices,
                          ScalarCPU s_min, ScalarCPU s_max);

/// Project onto the intersection of all active constraints in a ConstraintSet.
///
/// Uses generalized N-set Dykstra's alternating projection algorithm.
/// Cycles through: simplex -> box -> L1 ball (turnover) -> each sector.
/// Maintains one increment vector per constraint set.
///
/// Extends the 2-set project_simplex_box pattern to N constraint sets.
///
/// Reference:
///   Boyle & Dykstra, "A Method for Finding Projections onto the
///   Intersection of Convex Sets in Hilbert Spaces", 1986.
///
/// @param v Input vector.
/// @param constraints Active constraint set.
/// @param max_iter Maximum Dykstra iterations (default: 200).
/// @param tol Convergence tolerance (default: 1e-10).
/// @return Projected vector in the intersection of all constraint sets.
VectorXd project_constraints(const VectorXd& v,
                               const ConstraintSet& constraints,
                               int max_iter = 200, ScalarCPU tol = 1e-10);

}  // namespace cpo
