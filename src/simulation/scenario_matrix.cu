#include "simulation/scenario_matrix.h"

#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

#include "utils/cuda_utils.h"

namespace cpo {

ScenarioMatrix::ScenarioMatrix(Index n_scenarios, Index n_assets)
    : n_scenarios_(n_scenarios), n_assets_(n_assets) {
    size_t alloc_bytes = bytes();
    CUDA_CHECK(cudaMalloc(&d_data_, alloc_bytes));
    spdlog::info("ScenarioMatrix: allocated {} x {} ({:.1f} MB on GPU)",
                 n_scenarios_, n_assets_, alloc_bytes / (1024.0 * 1024.0));
}

ScenarioMatrix::~ScenarioMatrix() {
    if (d_data_) {
        // No CUDA_CHECK in destructor — avoid throwing.
        cudaFree(d_data_);
    }
}

ScenarioMatrix::ScenarioMatrix(ScenarioMatrix&& other) noexcept
    : d_data_(other.d_data_),
      n_scenarios_(other.n_scenarios_),
      n_assets_(other.n_assets_) {
    other.d_data_ = nullptr;
    other.n_scenarios_ = 0;
    other.n_assets_ = 0;
}

ScenarioMatrix& ScenarioMatrix::operator=(ScenarioMatrix&& other) noexcept {
    if (this != &other) {
        if (d_data_) {
            cudaFree(d_data_);
        }
        d_data_ = other.d_data_;
        n_scenarios_ = other.n_scenarios_;
        n_assets_ = other.n_assets_;
        other.d_data_ = nullptr;
        other.n_scenarios_ = 0;
        other.n_assets_ = 0;
    }
    return *this;
}

MatrixXs ScenarioMatrix::to_host() const {
    // Eigen MatrixXf is column-major by default — layout matches GPU.
    MatrixXs host(n_scenarios_, n_assets_);
    CUDA_CHECK(cudaMemcpy(host.data(), d_data_, bytes(),
                          cudaMemcpyDeviceToHost));
    return host;
}

void ScenarioMatrix::from_host(const MatrixXs& host_matrix) {
    if (host_matrix.rows() != n_scenarios_ ||
        host_matrix.cols() != n_assets_) {
        throw std::runtime_error(
            "ScenarioMatrix::from_host: dimension mismatch (" +
            std::to_string(host_matrix.rows()) + " x " +
            std::to_string(host_matrix.cols()) + ") vs (" +
            std::to_string(n_scenarios_) + " x " +
            std::to_string(n_assets_) + ")");
    }
    CUDA_CHECK(cudaMemcpy(d_data_, host_matrix.data(), bytes(),
                          cudaMemcpyHostToDevice));
}

void ScenarioMatrix::from_host(const std::vector<Scalar>& host_data) {
    if (host_data.size() != size()) {
        throw std::runtime_error(
            "ScenarioMatrix::from_host: size mismatch (" +
            std::to_string(host_data.size()) + " vs " +
            std::to_string(size()) + ")");
    }
    CUDA_CHECK(cudaMemcpy(d_data_, host_data.data(), bytes(),
                          cudaMemcpyHostToDevice));
}

}  // namespace cpo
