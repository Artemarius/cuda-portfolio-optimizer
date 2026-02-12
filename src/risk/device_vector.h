#pragma once

/// @file device_vector.h
/// @brief RAII wrapper for a 1D GPU device array.
///
/// Templated move-only wrapper. Mirrors ScenarioMatrix patterns:
///   CUDA_CHECK in constructor, no-throw destructor, cudaMemcpy transfers.
/// Explicit instantiation for Scalar (float) in device_vector.cu.

#include <vector>

#include "core/types.h"

namespace cpo {

/// Move-only RAII wrapper for a device-allocated 1D array.
template <typename T>
class DeviceVector {
public:
    /// Allocate a device array of n elements.
    explicit DeviceVector(Index n);

    ~DeviceVector();

    // Move-only: transfer ownership of device pointer.
    DeviceVector(DeviceVector&& other) noexcept;
    DeviceVector& operator=(DeviceVector&& other) noexcept;

    DeviceVector(const DeviceVector&) = delete;
    DeviceVector& operator=(const DeviceVector&) = delete;

    /// Copy device data to host as std::vector.
    std::vector<T> to_host() const;

    /// Upload host data to device. Size must match.
    void from_host(const std::vector<T>& host_data);

    /// @return Raw device pointer.
    T* device_ptr() { return d_data_; }
    const T* device_ptr() const { return d_data_; }

    /// @return Number of elements.
    Index size() const { return n_; }

    /// @return Size in bytes of the device allocation.
    size_t bytes() const { return static_cast<size_t>(n_) * sizeof(T); }

private:
    T* d_data_ = nullptr;
    Index n_ = 0;
};

// Explicit instantiation declarations — defined in device_vector.cu.
extern template class DeviceVector<Scalar>;

}  // namespace cpo
