#pragma once

/// @file admm_buffers.cuh
/// @brief GpuAdmmBuffers struct definition for use by CUDA translation units.
///
/// This header requires cuda_runtime.h and must only be included from .cu files.
/// The public API in admm_kernels.h uses a forward declaration of GpuAdmmBuffers
/// to keep CUDA headers out of C++ headers.

#include <cuda_runtime.h>

namespace cpo {

/// Pre-allocated device buffers for the ADMM x-update kernel.
///
/// Avoids per-call cudaMalloc/cudaFree overhead during the ADMM loop
/// (typically ~6000 evaluate_objective_gpu calls per solve).
///
/// Pattern mirrors CurandStates (opaque type, forward-declared in public header).
struct GpuAdmmBuffers {
    float* d_weights = nullptr;       ///< Weight vector (n_assets floats).
    double* d_sum_excess = nullptr;   ///< Scalar accumulator for sum of excess.
    double* d_grad_w = nullptr;       ///< Gradient accumulator (n_assets doubles).
    int* d_count = nullptr;           ///< Tail scenario count.
    int n_assets = 0;                 ///< Number of assets (buffer dimension).
};

}  // namespace cpo
