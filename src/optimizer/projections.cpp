#include "optimizer/projections.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace cpo {

// ── Simplex projection ──────────────────────────────────────────────

VectorXd project_simplex(const VectorXd& v) {
    // Duchi et al. 2008, "Efficient Projections onto the l1-Ball",
    // Algorithm 1 (adapted for the simplex = l1-ball intersected with
    // the positive orthant, with sum = 1).
    //
    // 1. Sort v descending: u_1 >= ... >= u_n
    // 2. rho = max{j in [n] : u_j - (1/j)(sum_{i=1}^{j} u_i - 1) > 0}
    // 3. theta = (1/rho)(sum_{i=1}^{rho} u_i - 1)
    // 4. w_i = max(v_i - theta, 0)

    const int n = static_cast<int>(v.size());
    if (n == 0) {
        throw std::runtime_error("project_simplex: empty vector");
    }

    // Step 1: Sort descending.
    std::vector<double> u(v.data(), v.data() + n);
    std::sort(u.begin(), u.end(), std::greater<double>());

    // Step 2: Find rho.
    double cumsum = 0.0;
    int rho = 0;
    for (int j = 0; j < n; ++j) {
        cumsum += u[j];
        double test = u[j] - (cumsum - 1.0) / (j + 1);
        if (test > 0.0) {
            rho = j + 1;  // 1-indexed
        }
    }

    // Step 3: Compute theta.
    double rho_sum = 0.0;
    for (int i = 0; i < rho; ++i) {
        rho_sum += u[i];
    }
    double theta = (rho_sum - 1.0) / rho;

    // Step 4: Project.
    VectorXd w(n);
    for (int i = 0; i < n; ++i) {
        w(i) = std::max(v(i) - theta, 0.0);
    }
    return w;
}

// ── Box projection ──────────────────────────────────────────────────

VectorXd project_box(const VectorXd& v, const VectorXd& lb,
                      const VectorXd& ub) {
    const int n = static_cast<int>(v.size());
    if (lb.size() != n || ub.size() != n) {
        throw std::runtime_error(
            "project_box: dimension mismatch (v=" + std::to_string(n) +
            ", lb=" + std::to_string(lb.size()) +
            ", ub=" + std::to_string(ub.size()) + ")");
    }

    VectorXd w(n);
    for (int i = 0; i < n; ++i) {
        w(i) = std::clamp(v(i), lb(i), ub(i));
    }
    return w;
}

// ── Combined simplex + box projection (Dykstra's algorithm) ─────────

VectorXd project_simplex_box(const VectorXd& v, const VectorXd& lb,
                              const VectorXd& ub, int max_iter,
                              double tol) {
    // Dykstra's alternating projection algorithm for the intersection
    // of two convex sets C1 (simplex) and C2 (box).
    //
    // Boyle & Dykstra 1986: unlike simple alternating projection,
    // Dykstra's method includes "increment" vectors that ensure
    // convergence to the true projection onto the intersection.
    //
    // x = v
    // p1 = p2 = 0  (increments)
    // repeat:
    //   y = project_C1(x + p1);  p1 = (x + p1) - y
    //   x = project_C2(y + p2);  p2 = (y + p2) - x

    const int n = static_cast<int>(v.size());
    if (lb.size() != n || ub.size() != n) {
        throw std::runtime_error(
            "project_simplex_box: dimension mismatch");
    }

    VectorXd x = v;
    VectorXd p1 = VectorXd::Zero(n);
    VectorXd p2 = VectorXd::Zero(n);

    for (int iter = 0; iter < max_iter; ++iter) {
        // Project onto simplex (C1).
        VectorXd y_input = x + p1;
        VectorXd y = project_simplex(y_input);
        p1 = y_input - y;

        // Project onto box (C2).
        VectorXd x_input = y + p2;
        VectorXd x_new = project_box(x_input, lb, ub);
        p2 = x_input - x_new;

        // Check convergence: change in x.
        double delta = (x_new - x).norm();
        x = x_new;

        if (delta < tol) {
            break;
        }
    }

    return x;
}

// ── L1 ball projection ──────────────────────────────────────────────

VectorXd project_l1_ball(const VectorXd& v, const VectorXd& center,
                          ScalarCPU radius) {
    // Project onto {x : ||x - center||_1 <= radius}.
    //
    // Duchi et al. 2008: project the shifted absolute values onto
    // the simplex scaled by radius, then restore signs.
    //
    // 1. d = v - center
    // 2. If ||d||_1 <= radius, return v (already feasible).
    // 3. Project |d| onto the L1 ball of radius `radius` via
    //    soft-thresholding: find theta s.t. sum max(|d_i| - theta, 0) = radius
    // 4. w_i = center_i + sign(d_i) * max(|d_i| - theta, 0)

    const int n = static_cast<int>(v.size());
    if (center.size() != n) {
        throw std::runtime_error(
            "project_l1_ball: dimension mismatch (v=" + std::to_string(n) +
            ", center=" + std::to_string(center.size()) + ")");
    }
    if (radius < 0.0) {
        throw std::runtime_error(
            "project_l1_ball: radius must be >= 0");
    }

    VectorXd d = v - center;
    double l1_norm = d.lpNorm<1>();

    // Already inside the ball.
    if (l1_norm <= radius + 1e-15) {
        return v;
    }

    // Special case: radius = 0 -> project to center.
    if (radius < 1e-15) {
        return center;
    }

    // Sort |d| descending to find the soft-threshold parameter.
    // Same structure as simplex projection but for L1 ball.
    std::vector<double> abs_d(n);
    for (int i = 0; i < n; ++i) {
        abs_d[i] = std::abs(d(i));
    }
    std::sort(abs_d.begin(), abs_d.end(), std::greater<double>());

    // Find theta: largest j s.t. abs_d[j] - (sum_{i=0}^{j} abs_d[i] - radius) / (j+1) > 0
    double cumsum = 0.0;
    int rho = 0;
    for (int j = 0; j < n; ++j) {
        cumsum += abs_d[j];
        double test = abs_d[j] - (cumsum - radius) / (j + 1);
        if (test > 0.0) {
            rho = j + 1;
        }
    }

    double rho_sum = 0.0;
    for (int i = 0; i < rho; ++i) {
        rho_sum += abs_d[i];
    }
    double theta = (rho_sum - radius) / rho;

    // Soft-threshold with sign preservation.
    VectorXd w(n);
    for (int i = 0; i < n; ++i) {
        double sign = (d(i) >= 0.0) ? 1.0 : -1.0;
        w(i) = center(i) + sign * std::max(std::abs(d(i)) - theta, 0.0);
    }
    return w;
}

// ── Sector projection ───────────────────────────────────────────────

VectorXd project_sector(const VectorXd& v, const std::vector<Index>& indices,
                          ScalarCPU s_min, ScalarCPU s_max) {
    // Project so that sum(v[i] for i in indices) lies in [s_min, s_max].
    //
    // If already feasible, return v unchanged.
    // Otherwise, uniformly shift sector elements to reach the nearest bound.

    if (indices.empty()) return v;

    double sector_sum = 0.0;
    for (Index idx : indices) {
        sector_sum += v(idx);
    }

    if (sector_sum >= s_min - 1e-15 && sector_sum <= s_max + 1e-15) {
        return v;
    }

    VectorXd w = v;
    double target = (sector_sum < s_min) ? s_min : s_max;
    double adjustment = (target - sector_sum) / static_cast<double>(indices.size());

    for (Index idx : indices) {
        w(idx) += adjustment;
    }

    return w;
}

// ── Generalized Dykstra's alternating projection ────────────────────

VectorXd project_constraints(const VectorXd& v,
                               const ConstraintSet& constraints,
                               int max_iter, ScalarCPU tol) {
    // Generalized N-set Dykstra's alternating projection algorithm.
    //
    // Boyle & Dykstra 1986: for N convex sets C_1, ..., C_N, maintain
    // increment vectors p_1, ..., p_N initialized to zero.
    //
    // Each cycle:
    //   for k = 1..N:
    //     y = project_{C_k}(x + p_k)
    //     p_k = (x + p_k) - y
    //     x = y
    //
    // Converges to the projection onto the intersection.

    const int n = static_cast<int>(v.size());

    // Build the list of projection functions.
    // Order: simplex -> box -> L1 ball (turnover) -> each sector.
    const int n_sets = constraints.num_constraint_sets();

    // Increment vectors (one per constraint set).
    std::vector<VectorXd> increments(n_sets, VectorXd::Zero(n));

    VectorXd x = v;

    for (int iter = 0; iter < max_iter; ++iter) {
        VectorXd x_prev = x;
        auto increments_prev = increments;
        int set_idx = 0;

        // Set 0: Simplex projection (always active).
        {
            VectorXd y_input = x + increments[set_idx];
            VectorXd y = project_simplex(y_input);
            increments[set_idx] = y_input - y;
            x = y;
            ++set_idx;
        }

        // Set 1: Box projection (if active).
        if (constraints.has_position_limits) {
            VectorXd y_input = x + increments[set_idx];
            VectorXd y = project_box(y_input,
                                      constraints.position_limits.w_min,
                                      constraints.position_limits.w_max);
            increments[set_idx] = y_input - y;
            x = y;
            ++set_idx;
        }

        // Set 2: L1 ball projection for turnover (if active).
        if (constraints.has_turnover) {
            VectorXd y_input = x + increments[set_idx];
            VectorXd y = project_l1_ball(y_input,
                                          constraints.turnover.w_prev,
                                          constraints.turnover.tau);
            increments[set_idx] = y_input - y;
            x = y;
            ++set_idx;
        }

        // Sets 3+: Sector projections (one per sector bound).
        if (constraints.has_sector_constraints) {
            for (const auto& sector : constraints.sector_constraints.sectors) {
                VectorXd y_input = x + increments[set_idx];
                VectorXd y = project_sector(y_input, sector.assets,
                                             sector.min_exposure,
                                             sector.max_exposure);
                increments[set_idx] = y_input - y;
                x = y;
                ++set_idx;
            }
        }

        // Check convergence: both x and all increments must be stable.
        // Checking only x can cause premature termination when x is
        // temporarily stationary but increments are still evolving.
        double delta = (x - x_prev).norm();
        for (int k = 0; k < n_sets; ++k) {
            delta += (increments[k] - increments_prev[k]).norm();
        }
        if (delta < tol) {
            break;
        }
    }

    return x;
}

}  // namespace cpo
