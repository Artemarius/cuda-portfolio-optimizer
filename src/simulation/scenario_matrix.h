#pragma once

/// @file scenario_matrix.h
/// @brief RAII wrapper for a GPU-resident scenario matrix.
///
/// Layout: column-major, N_scenarios x N_assets.
/// Element (i, j) is at index j * n_scenarios + i.
/// This layout gives coalesced reads when computing portfolio loss:
/// all threads in a warp access the same column (asset) simultaneously.

#include <vector>

#include "core/types.h"

namespace cpo {

/// RAII wrapper for a device-allocated float matrix (column-major).
///
/// Move-only. cudaFree in destructor (no CUDA_CHECK to avoid throwing).
class ScenarioMatrix {
public:
    /// Allocate a column-major device matrix of size n_scenarios x n_assets.
    ScenarioMatrix(Index n_scenarios, Index n_assets);

    ~ScenarioMatrix();

    // Move-only: transfer ownership of device pointer.
    ScenarioMatrix(ScenarioMatrix&& other) noexcept;
    ScenarioMatrix& operator=(ScenarioMatrix&& other) noexcept;

    ScenarioMatrix(const ScenarioMatrix&) = delete;
    ScenarioMatrix& operator=(const ScenarioMatrix&) = delete;

    /// Copy device data to host as Eigen column-major float matrix.
    MatrixXs to_host() const;

    /// Upload host Eigen matrix to device. Matrix must match dimensions.
    void from_host(const MatrixXs& host_matrix);

    /// Upload host flat vector to device. Size must match n_scenarios * n_assets.
    void from_host(const std::vector<Scalar>& host_data);

    /// @return Raw device pointer (column-major float array).
    Scalar* device_ptr() { return d_data_; }
    const Scalar* device_ptr() const { return d_data_; }

    Index n_scenarios() const { return n_scenarios_; }
    Index n_assets() const { return n_assets_; }

    /// @return Total number of elements (n_scenarios * n_assets).
    size_t size() const {
        return static_cast<size_t>(n_scenarios_) * n_assets_;
    }

    /// @return Size in bytes of the device allocation.
    size_t bytes() const { return size() * sizeof(Scalar); }

private:
    Scalar* d_data_ = nullptr;
    Index n_scenarios_ = 0;
    Index n_assets_ = 0;
};

}  // namespace cpo
