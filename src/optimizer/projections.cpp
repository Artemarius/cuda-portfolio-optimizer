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

}  // namespace cpo
