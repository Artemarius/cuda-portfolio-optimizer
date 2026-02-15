#pragma once

/// @file anderson_acceleration.h
/// @brief Anderson acceleration (Type-I, safeguarded) for fixed-point iterations.
///
/// Accelerates the convergence of fixed-point iterations x_{k+1} = G(x_k)
/// by extrapolating from the last m iterates. Applicable to any fixed-point
/// map, including ADMM, where G(x) = one full ADMM step.
///
/// Type-I Anderson acceleration maintains a history of iterates and residuals
/// (r_k = G(x_k) - x_k), solves a small least-squares problem to find
/// optimal mixing coefficients, and produces an accelerated iterate.
///
/// Safeguarding: if the accelerated residual exceeds the unaccelerated
/// residual by more than safeguard_factor, the history is cleared and
/// the unaccelerated iterate is returned (restart).
///
/// Reference:
///   Zhang, O'Donoghue, Boyd, "Globally Convergent Type-I Anderson
///   Acceleration for Non-Smooth Fixed-Point Iterations",
///   SIAM J. Optim. 30(4), 2020, pp. 3170-3197.
///   - Algorithm 1 (Type-I AA with safeguarding)
///   - Eq. (2.2): mixing coefficient least-squares problem
///   - Eq. (2.3): accelerated iterate formula
///   - Section 3: safeguarding criterion

#include <vector>

#include "core/types.h"

namespace cpo {

/// Anderson acceleration (Type-I) for fixed-point iterations.
///
/// Usage pattern:
/// @code
///   AndersonAccelerator aa(dimension, depth);
///   for (int k = 0; k < max_iter; ++k) {
///       VectorXd g = G(x);              // one fixed-point step
///       VectorXd x_acc = aa.accelerate(x, g);
///       x = x_acc;
///   }
/// @endcode
///
/// For ADMM integration, the state vector packs (z, zeta) and G(x) is one
/// full ADMM iteration (x-update, z-update, u-update).
struct AndersonAccelerator {
    /// History depth (number of past iterates to use, typically 3-5).
    /// The least-squares problem is at most m x m.
    int m;

    /// Dimension of the iterate vectors.
    int n;

    /// Safeguard factor: restart if accelerated residual exceeds this
    /// multiple of the unaccelerated residual.
    /// Zhang et al. 2020, Section 3: D_k >= safeguard_factor * ||r_k||.
    /// Default 1.0 means restart whenever acceleration does not help.
    ScalarCPU safeguard_factor;

    /// Construct an Anderson accelerator.
    ///
    /// @param dimension Dimension of the iterate vectors.
    /// @param depth     History depth m (number of past residual differences
    ///                  to store). Must be >= 1. Typical values: 3-5.
    /// @param safeguard Safeguard factor for restart criterion (default 1.0).
    /// @throws std::invalid_argument if dimension < 1 or depth < 1.
    explicit AndersonAccelerator(int dimension, int depth = 5,
                                  ScalarCPU safeguard = 1.0);

    /// Feed a new iterate x_k and its fixed-point image g_k = G(x_k).
    ///
    /// Computes the residual r_k = g_k - x_k. If sufficient history
    /// is available (>= 2 iterates), solves the least-squares problem
    /// (Zhang et al. 2020, Eq. 2.2):
    ///
    ///   min_{alpha} || r_k - sum_{i=0}^{m_k-1} alpha_i (Delta_r_i) ||_2
    ///
    /// where Delta_r_i = r_{k-m_k+i+1} - r_{k-m_k+i} are successive
    /// residual differences, and m_k = min(m, k) is the current depth.
    ///
    /// The accelerated iterate is then (Eq. 2.3):
    ///
    ///   x_acc = g_k - sum_{i=0}^{m_k-1} alpha_i * Delta_g_i
    ///
    /// where Delta_g_i = g_{k-m_k+i+1} - g_{k-m_k+i}.
    ///
    /// If history is insufficient (< 2 iterates), returns g unchanged.
    ///
    /// @param x Current iterate x_k.
    /// @param g Fixed-point image G(x_k).
    /// @return Accelerated iterate x_{k+1}.
    /// @throws std::invalid_argument if x.size() != n or g.size() != n.
    VectorXd accelerate(const VectorXd& x, const VectorXd& g);

    /// Reset all history buffers. Call after a restart or when the
    /// fixed-point map changes (e.g., after rho adaptation in ADMM).
    void reset();

    /// Check whether acceleration should be restarted.
    ///
    /// Restart criterion (Zhang et al. 2020, Section 3):
    ///   accelerated_residual > safeguard_factor * unaccelerated_residual
    ///
    /// When this returns true, the caller should discard the accelerated
    /// iterate and use the unaccelerated one instead.
    ///
    /// @param accelerated_residual   ||G(x_acc) - x_acc|| or similar norm.
    /// @param unaccelerated_residual ||g_k - x_k|| (before acceleration).
    /// @return True if restart is recommended.
    bool should_restart(ScalarCPU accelerated_residual,
                        ScalarCPU unaccelerated_residual) const;

private:
    /// Ring buffer of past iterates g_k = G(x_k).
    std::vector<VectorXd> g_history_;

    /// Ring buffer of past residuals r_k = g_k - x_k.
    std::vector<VectorXd> r_history_;

    /// Number of iterates stored so far (before reaching full depth).
    int count_;
};

}  // namespace cpo
