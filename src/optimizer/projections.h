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

}  // namespace cpo
