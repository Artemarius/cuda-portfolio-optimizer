#include "optimizer/anderson_acceleration.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <Eigen/Dense>

namespace cpo {

AndersonAccelerator::AndersonAccelerator(int dimension, int depth,
                                          ScalarCPU safeguard)
    : m(depth), n(dimension), safeguard_factor(safeguard), count_(0) {
    if (dimension < 1) {
        throw std::invalid_argument(
            "AndersonAccelerator: dimension must be >= 1, got " +
            std::to_string(dimension));
    }
    if (depth < 1) {
        throw std::invalid_argument(
            "AndersonAccelerator: depth must be >= 1, got " +
            std::to_string(depth));
    }

    // Pre-allocate ring buffers to maximum capacity (m + 1 entries).
    // We need at most m+1 past iterates to form m residual differences.
    g_history_.reserve(m + 1);
    r_history_.reserve(m + 1);
}

VectorXd AndersonAccelerator::accelerate(const VectorXd& x,
                                           const VectorXd& g) {
    if (x.size() != n) {
        throw std::invalid_argument(
            "AndersonAccelerator::accelerate: x.size()=" +
            std::to_string(x.size()) + " != n=" + std::to_string(n));
    }
    if (g.size() != n) {
        throw std::invalid_argument(
            "AndersonAccelerator::accelerate: g.size()=" +
            std::to_string(g.size()) + " != n=" + std::to_string(n));
    }

    // Compute residual r_k = g_k - x_k.
    VectorXd r = g - x;

    // Store g and r in ring buffers.
    if (static_cast<int>(g_history_.size()) < m + 1) {
        g_history_.push_back(g);
        r_history_.push_back(r);
    } else {
        // Shift buffer: discard oldest, append newest.
        // This keeps the most recent m+1 entries.
        for (int i = 0; i < m; ++i) {
            g_history_[i] = g_history_[i + 1];
            r_history_[i] = r_history_[i + 1];
        }
        g_history_[m] = g;
        r_history_[m] = r;
    }
    ++count_;

    const int stored = static_cast<int>(r_history_.size());

    // Need at least 2 stored iterates to form 1 residual difference.
    // Zhang et al. 2020, Algorithm 1: m_k = min(m, k).
    if (stored < 2) {
        return g;
    }

    // Number of residual differences available: m_k = stored - 1,
    // capped at depth m.
    const int m_k = std::min(m, stored - 1);

    // Build the matrix of residual differences Delta_R (n x m_k).
    // Delta_R[:,i] = r_{stored-m_k+i} - r_{stored-m_k+i-1}
    //             = r_history_[stored-m_k+i] - r_history_[stored-m_k+i-1]
    //
    // Similarly for Delta_G (iterate differences).
    //
    // Zhang et al. 2020, Eq. (2.2):
    //   min_{alpha} || r_k - Delta_R * alpha ||_2^2
    //
    // where r_k = r_history_[stored-1] (the latest residual).

    MatrixXd delta_r(n, m_k);
    MatrixXd delta_g(n, m_k);

    const int base = stored - m_k - 1;  // index of the "oldest - 1" entry
    for (int i = 0; i < m_k; ++i) {
        delta_r.col(i) = r_history_[base + i + 1] - r_history_[base + i];
        delta_g.col(i) = g_history_[base + i + 1] - g_history_[base + i];
    }

    // Solve the least-squares problem: min || r_k - Delta_R * alpha ||_2
    // Using Eigen's column-pivoting Householder QR for numerical stability
    // on this small (n x m_k) system.
    //
    // The normal equations are: Delta_R^T Delta_R alpha = Delta_R^T r_k
    // but we solve via QR for better conditioning.
    const VectorXd& r_k = r_history_[stored - 1];

    // Use ColPivHouseholderQR for robustness with potentially
    // rank-deficient Delta_R (can happen when iterates stagnate).
    Eigen::ColPivHouseholderQR<MatrixXd> qr(delta_r);
    VectorXd alpha = qr.solve(r_k);

    // Compute accelerated iterate (Zhang et al. 2020, Eq. 2.3):
    //   x_acc = g_k - Delta_G * alpha
    const VectorXd& g_k = g_history_[stored - 1];
    VectorXd x_acc = g_k - delta_g * alpha;

    return x_acc;
}

void AndersonAccelerator::reset() {
    g_history_.clear();
    r_history_.clear();
    count_ = 0;
}

bool AndersonAccelerator::should_restart(
    ScalarCPU accelerated_residual,
    ScalarCPU unaccelerated_residual) const {
    // Zhang et al. 2020, Section 3 (safeguarding):
    // Restart when the accelerated step is worse than the unaccelerated step.
    return accelerated_residual > safeguard_factor * unaccelerated_residual;
}

}  // namespace cpo
