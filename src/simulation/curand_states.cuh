#pragma once

/// @file curand_states.cuh
/// @brief CurandStates struct definition for use by CUDA translation units.
///
/// This header requires curand_kernel.h and must only be included from .cu files.
/// The public API in monte_carlo.h uses a forward declaration of CurandStates
/// to keep curand_kernel.h out of C++ headers.

#include <curand_kernel.h>

#include "core/types.h"

namespace cpo {

/// Device-allocated cuRAND RNG states.
///
/// Defined here (rather than in monte_carlo.cu) so multiple .cu files
/// can access the d_states member. The public header forward-declares
/// this struct to maintain the opaque-type pattern.
struct CurandStates {
    curandState_t* d_states = nullptr;
    Index n_states = 0;
};

}  // namespace cpo
