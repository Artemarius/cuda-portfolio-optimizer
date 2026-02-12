#pragma once

/// @file admm_kernels.h
/// @brief GPU-accelerated kernels for the ADMM x-update.
///
/// The x-update bottleneck is evaluating the Rockafellar-Uryasev
/// objective gradient across all scenarios:
///   dF/dw = -(1/(N*alpha)) * sum_{i: loss_i > zeta} r_i
///
/// This requires computing loss_i = -r_i'w for each scenario and
/// accumulating the gradient contribution from tail scenarios.
/// Perfect GPU workload: independent per-scenario computation.

#include "core/types.h"
#include "simulation/scenario_matrix.h"

namespace cpo {

/// GPU result buffers for the objective evaluation kernel.
struct GpuObjectiveResult {
    ScalarCPU value = 0.0;       ///< Sum of max(0, loss_i - zeta).
    VectorXd grad_w;             ///< Accumulated gradient (n_assets).
    int count_exceeding = 0;     ///< Number of scenarios in the tail.
};

/// Evaluate the R-U objective components on GPU.
///
/// Computes across all scenarios:
///   sum_excess = sum_i max(0, -r_i'w - zeta)
///   grad_w = -sum_{i: -r_i'w > zeta} r_i
///   count  = |{i : -r_i'w > zeta}|
///
/// The caller assembles the final objective and gradient:
///   F(w,zeta) = zeta + sum_excess / (N * alpha)
///   dF/dw     = grad_w / (N * alpha)
///
/// @param scenarios GPU-resident scenario matrix (float, column-major).
/// @param w Weight vector (float, n_assets).
/// @param zeta Auxiliary variable (VaR estimate).
/// @param threads_per_block CUDA threads per block.
/// @return GpuObjectiveResult with accumulated values.
GpuObjectiveResult evaluate_objective_gpu(const ScenarioMatrix& scenarios,
                                           const VectorXs& w,
                                           Scalar zeta,
                                           int threads_per_block = 256);

}  // namespace cpo
