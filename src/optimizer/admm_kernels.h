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

#include <memory>

#include "core/types.h"
#include "simulation/scenario_matrix.h"

namespace cpo {

// ── Pre-allocated GPU buffers (opaque type) ─────────────────────────
// Defined in admm_buffers.cuh; forward-declared here to keep CUDA
// headers out of C++ translation units.

/// Opaque handle to pre-allocated ADMM device buffers.
struct GpuAdmmBuffers;

/// Free ADMM device buffers (no-throw).
void destroy_gpu_admm_buffers(GpuAdmmBuffers* buffers);

/// Custom deleter for use with unique_ptr.
struct GpuAdmmBuffersDeleter {
    void operator()(GpuAdmmBuffers* p) const { destroy_gpu_admm_buffers(p); }
};

/// RAII guard for ADMM GPU buffers. Move-only.
using GpuAdmmBuffersGuard = std::unique_ptr<GpuAdmmBuffers, GpuAdmmBuffersDeleter>;

/// Allocate pre-allocated ADMM device buffers.
/// @param n_assets Number of assets (determines buffer sizes).
/// @return RAII guard owning the device-allocated buffers.
GpuAdmmBuffersGuard create_gpu_admm_buffers(int n_assets);

// ── GPU objective evaluation ────────────────────────────────────────

/// GPU result buffers for the objective evaluation kernel.
struct GpuObjectiveResult {
    ScalarCPU value = 0.0;       ///< Sum of max(0, loss_i - zeta).
    VectorXd grad_w;             ///< Accumulated gradient (n_assets).
    int count_exceeding = 0;     ///< Number of scenarios in the tail.
};

/// Evaluate the R-U objective components on GPU (allocates per call).
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

/// Evaluate the R-U objective components on GPU (pre-allocated buffers).
///
/// Same computation as above, but reuses device buffers from
/// GpuAdmmBuffers instead of allocating/freeing per call. Use this
/// in tight loops (ADMM x-update: ~6000 calls per solve).
///
/// @param scenarios GPU-resident scenario matrix (float, column-major).
/// @param w Weight vector (float, n_assets).
/// @param zeta Auxiliary variable (VaR estimate).
/// @param buffers Pre-allocated device buffers (must match n_assets).
/// @param threads_per_block CUDA threads per block.
/// @return GpuObjectiveResult with accumulated values.
GpuObjectiveResult evaluate_objective_gpu(const ScenarioMatrix& scenarios,
                                           const VectorXs& w,
                                           Scalar zeta,
                                           GpuAdmmBuffers* buffers,
                                           int threads_per_block = 256);

}  // namespace cpo
