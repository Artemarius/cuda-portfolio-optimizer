#pragma once

/// @file timer.h
/// @brief RAII CPU and CUDA event timers. Log elapsed time on destruction.

#include <chrono>
#include <string>

#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

namespace cpo {

/// RAII CPU timer using std::chrono high-resolution clock.
/// Logs elapsed wall-clock time (ms) via spdlog on destruction.
class CpuTimer {
public:
    explicit CpuTimer(std::string label)
        : label_(std::move(label)),
          start_(std::chrono::high_resolution_clock::now()) {}

    ~CpuTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto ms =
            std::chrono::duration<double, std::milli>(end - start_).count();
        spdlog::info("{}: {:.3f} ms", label_, ms);
    }

    CpuTimer(const CpuTimer&) = delete;
    CpuTimer& operator=(const CpuTimer&) = delete;

    /// Get elapsed time in milliseconds without stopping the timer.
    double elapsed_ms() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(now - start_).count();
    }

private:
    std::string label_;
    std::chrono::high_resolution_clock::time_point start_;
};

/// RAII CUDA event timer. Records GPU-side elapsed time.
/// Logs elapsed GPU time (ms) via spdlog on destruction.
class CudaTimer {
public:
    explicit CudaTimer(std::string label, cudaStream_t stream = 0)
        : label_(std::move(label)), stream_(stream) {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
        cudaEventRecord(start_, stream_);
    }

    ~CudaTimer() {
        cudaEventRecord(stop_, stream_);
        cudaEventSynchronize(stop_);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start_, stop_);
        spdlog::info("{}: {:.3f} ms (GPU)", label_, ms);
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    CudaTimer(const CudaTimer&) = delete;
    CudaTimer& operator=(const CudaTimer&) = delete;

    /// Get elapsed GPU time in milliseconds. Synchronizes the stop event.
    float elapsed_ms() {
        cudaEventRecord(stop_, stream_);
        cudaEventSynchronize(stop_);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start_, stop_);
        return ms;
    }

private:
    std::string label_;
    cudaStream_t stream_;
    cudaEvent_t start_;
    cudaEvent_t stop_;
};

}  // namespace cpo
