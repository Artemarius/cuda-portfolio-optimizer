#pragma once

/// @file objective.h
/// @brief Rockafellar-Uryasev CVaR objective evaluation and subgradient.
///
/// The Mean-CVaR optimization is reformulated following Rockafellar &
/// Uryasev (J. Risk, 2000) as:
///
///   min_{w, zeta}  F(w, zeta) = zeta + (1 / (N * alpha)) * sum_i max(0, loss_i - zeta)
///
/// where loss_i = -r_i' w  (negative portfolio return for scenario i),
/// alpha = 1 - confidence_level (tail probability),
/// and N = number of scenarios.
///
/// This is convex in (w, zeta) jointly. For fixed w, the optimal zeta
/// equals VaR_alpha, and F evaluates to CVaR_alpha.
///
/// Reference:
///   Rockafellar & Uryasev, "Optimization of Conditional Value-at-Risk",
///   Journal of Risk, Vol. 2, No. 3, 2000, Eq. (9)-(10).

#include "core/types.h"

namespace cpo {

/// Result of evaluating the Rockafellar-Uryasev objective.
struct ObjectiveResult {
    ScalarCPU value = 0.0;         ///< F(w, zeta) objective value.
    VectorXd grad_w;               ///< Subgradient w.r.t. w (n_assets).
    ScalarCPU grad_zeta = 0.0;     ///< Subgradient w.r.t. zeta.
    ScalarCPU optimal_zeta = 0.0;  ///< Optimal zeta for given w (= VaR).
};

/// Evaluate the Rockafellar-Uryasev CVaR objective and its subgradient.
///
/// F(w, zeta) = zeta + (1/(N*alpha)) * sum_i max(0, -r_i'w - zeta)
///
/// Subgradients:
///   dF/dw    = -(1/(N*alpha)) * sum_{i in tail} r_i
///   dF/dzeta = 1 - (1/(N*alpha)) * |{i : -r_i'w > zeta}|
///            = 1 - (count_exceeding / (N * alpha))
///
/// @param scenarios Return matrix (n_scenarios x n_assets, double).
/// @param w Portfolio weights (n_assets, double).
/// @param zeta Auxiliary variable (VaR estimate).
/// @param alpha Tail probability = 1 - confidence_level (e.g., 0.05).
/// @return ObjectiveResult with value and subgradients.
ObjectiveResult evaluate_objective_cpu(const MatrixXd& scenarios,
                                        const VectorXd& w,
                                        ScalarCPU zeta,
                                        ScalarCPU alpha);

/// Find the optimal zeta (= VaR) for a given w by sorting losses.
///
/// The R-U objective is minimized over zeta at zeta* = VaR_alpha(w),
/// which is the alpha-quantile of the loss distribution -r'w.
///
/// @param scenarios Return matrix (n_scenarios x n_assets, double).
/// @param w Portfolio weights (n_assets, double).
/// @param alpha Tail probability = 1 - confidence_level.
/// @return Optimal zeta (= VaR_alpha).
ScalarCPU find_optimal_zeta(const MatrixXd& scenarios,
                             const VectorXd& w,
                             ScalarCPU alpha);

}  // namespace cpo
