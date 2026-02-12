#pragma once

/// @file constraint_set.h
/// @brief Portfolio constraint data structures for the ADMM optimizer.
///
/// Defines position limits, turnover, and sector constraints that
/// augment the base simplex constraint (1'w = 1, w >= 0).
///
/// All constraints are enforced via projection in the ADMM z-update
/// using generalized Dykstra's alternating projection algorithm.
///
/// References:
///   Boyle & Dykstra, "A Method for Finding Projections onto the
///   Intersection of Convex Sets in Hilbert Spaces", 1986.

#include <string>
#include <vector>

#include <nlohmann/json_fwd.hpp>

#include "core/types.h"

namespace cpo {

/// Per-asset position limits: w_min <= w <= w_max.
struct PositionLimits {
    VectorXd w_min;  ///< Lower bound per asset.
    VectorXd w_max;  ///< Upper bound per asset.
};

/// L1 turnover constraint: ||w - w_prev||_1 <= tau.
///
/// Limits total trading from the previous portfolio. A turnover of
/// tau = 2.0 is unconstrained (full liquidation and rebalance).
/// tau = 0.0 forces w = w_prev (no trading).
struct TurnoverConstraint {
    VectorXd w_prev;        ///< Previous portfolio weights.
    ScalarCPU tau = 2.0;    ///< Maximum L1 turnover (sum of abs trades).
};

/// A single sector exposure bound: s_min <= sum(w[i] for i in assets) <= s_max.
struct SectorBound {
    std::string name;                ///< Sector name (e.g., "Technology").
    std::vector<Index> assets;       ///< Asset indices belonging to this sector.
    ScalarCPU min_exposure = 0.0;    ///< Minimum sector weight.
    ScalarCPU max_exposure = 1.0;    ///< Maximum sector weight.
};

/// Collection of sector constraints.
struct SectorConstraints {
    std::vector<SectorBound> sectors;
};

/// Unified constraint set for the portfolio optimizer.
///
/// Combines optional position limits, turnover, and sector constraints.
/// The base simplex constraint (1'w = 1, w >= 0) is always active.
struct ConstraintSet {
    bool has_position_limits = false;
    PositionLimits position_limits;

    bool has_turnover = false;
    TurnoverConstraint turnover;

    bool has_sector_constraints = false;
    SectorConstraints sector_constraints;

    /// Count active constraint sets for Dykstra's algorithm.
    /// Always includes simplex (1). Adds 1 for position limits,
    /// 1 for turnover, and 1 per sector bound.
    int num_constraint_sets() const;

    /// Validate constraint dimensions and feasibility.
    ///
    /// Checks:
    ///   - w_min/w_max dimensions match n_assets
    ///   - w_min <= w_max element-wise
    ///   - sum(w_min) <= 1 <= sum(w_max) (simplex feasibility)
    ///   - Sector asset indices in [0, n_assets)
    ///   - tau >= 0
    ///   - w_prev dimension matches n_assets
    ///
    /// @param n_assets Number of assets in the portfolio.
    /// @throws std::runtime_error if validation fails.
    void validate(Index n_assets) const;

    /// Check whether a weight vector satisfies all active constraints.
    ///
    /// @param w Weight vector to check.
    /// @param tol Tolerance for constraint satisfaction.
    /// @return True if w is feasible within tolerance.
    bool is_feasible(const VectorXd& w, ScalarCPU tol = 1e-6) const;
};

/// Parse a ConstraintSet from a JSON object.
///
/// Expected JSON structure:
/// ```json
/// {
///   "position_limits": { "w_min": [0.0, ...], "w_max": [0.5, ...] },
///   "turnover": { "w_prev": [0.3, ...], "tau": 0.5 },
///   "sectors": [
///     { "name": "Tech", "assets": [0, 1], "min": 0.1, "max": 0.4 }
///   ]
/// }
/// ```
///
/// @param j JSON object containing constraint specification.
/// @param n_assets Number of assets (for dimension validation).
/// @return Parsed and validated ConstraintSet.
ConstraintSet parse_constraints(const nlohmann::json& j, Index n_assets);

}  // namespace cpo
