#pragma once

/// @file cuda_utils.h
/// @brief CUDA error checking macro and device query utilities.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

/// Check CUDA API call and abort on error.
#define CUDA_CHECK(err)                                                    \
    do {                                                                   \
        cudaError_t err_ = (err);                                          \
        if (err_ != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,        \
                    __LINE__, cudaGetErrorString(err_));                    \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

namespace cpo {

/// Print GPU device name, SM count, memory, and compute capability.
void device_query();

/// @return Free VRAM in bytes on the current device.
size_t get_free_vram();

/// Log current VRAM usage (used / total) via spdlog.
void log_vram_usage();

}  // namespace cpo
