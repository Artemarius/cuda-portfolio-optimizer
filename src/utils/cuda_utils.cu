#include "utils/cuda_utils.h"

#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

namespace cpo {

void device_query() {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    if (device_count == 0) {
        spdlog::error("No CUDA-capable devices found");
        return;
    }

    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));

        spdlog::info("GPU {}: {}", i, prop.name);
        spdlog::info("  Compute capability: {}.{}", prop.major, prop.minor);
        spdlog::info("  SM count:           {}", prop.multiProcessorCount);
        spdlog::info("  Global memory:      {:.0f} MB",
                     prop.totalGlobalMem / (1024.0 * 1024.0));
        spdlog::info("  Memory clock:       {:.0f} MHz",
                     prop.memoryClockRate / 1000.0);
        spdlog::info("  Memory bus width:   {} bits", prop.memoryBusWidth);
        spdlog::info("  L2 cache size:      {} KB", prop.l2CacheSize / 1024);
        spdlog::info("  Max threads/block:  {}", prop.maxThreadsPerBlock);
        spdlog::info("  Warp size:          {}", prop.warpSize);
    }
}

size_t get_free_vram() {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    return free_bytes;
}

void log_vram_usage() {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));

    size_t used_bytes = total_bytes - free_bytes;
    spdlog::info("VRAM: {:.1f} MB used / {:.1f} MB total ({:.1f}% free)",
                 used_bytes / (1024.0 * 1024.0),
                 total_bytes / (1024.0 * 1024.0),
                 100.0 * free_bytes / total_bytes);
}

}  // namespace cpo
