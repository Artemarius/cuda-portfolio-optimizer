#include "risk/device_vector.h"

#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

#include "utils/cuda_utils.h"

namespace cpo {

template <typename T>
DeviceVector<T>::DeviceVector(Index n) : n_(n) {
    CUDA_CHECK(cudaMalloc(&d_data_, bytes()));
}

template <typename T>
DeviceVector<T>::~DeviceVector() {
    if (d_data_) {
        // No CUDA_CHECK in destructor — avoid throwing.
        cudaFree(d_data_);
    }
}

template <typename T>
DeviceVector<T>::DeviceVector(DeviceVector&& other) noexcept
    : d_data_(other.d_data_), n_(other.n_) {
    other.d_data_ = nullptr;
    other.n_ = 0;
}

template <typename T>
DeviceVector<T>& DeviceVector<T>::operator=(DeviceVector&& other) noexcept {
    if (this != &other) {
        if (d_data_) {
            cudaFree(d_data_);
        }
        d_data_ = other.d_data_;
        n_ = other.n_;
        other.d_data_ = nullptr;
        other.n_ = 0;
    }
    return *this;
}

template <typename T>
std::vector<T> DeviceVector<T>::to_host() const {
    std::vector<T> host(n_);
    CUDA_CHECK(cudaMemcpy(host.data(), d_data_, bytes(),
                          cudaMemcpyDeviceToHost));
    return host;
}

template <typename T>
void DeviceVector<T>::from_host(const std::vector<T>& host_data) {
    if (static_cast<Index>(host_data.size()) != n_) {
        throw std::runtime_error(
            "DeviceVector::from_host: size mismatch (" +
            std::to_string(host_data.size()) + " vs " +
            std::to_string(n_) + ")");
    }
    CUDA_CHECK(cudaMemcpy(d_data_, host_data.data(), bytes(),
                          cudaMemcpyHostToDevice));
}

// Explicit instantiation for Scalar (float).
template class DeviceVector<Scalar>;

}  // namespace cpo
