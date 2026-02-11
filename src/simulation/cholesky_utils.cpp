#include "simulation/cholesky_utils.h"

#include <cmath>

#include <Eigen/Cholesky>
#include <spdlog/spdlog.h>

namespace cpo {

CholeskyResult compute_cholesky(const MatrixXd& cov) {
    if (cov.rows() != cov.cols()) {
        throw std::runtime_error(
            "compute_cholesky: covariance matrix is not square (" +
            std::to_string(cov.rows()) + " x " + std::to_string(cov.cols()) +
            ")");
    }

    const Index n = static_cast<Index>(cov.rows());

    // Eigen LLT in double precision for numerical stability.
    Eigen::LLT<MatrixXd> llt(cov);
    if (llt.info() != Eigen::Success) {
        throw std::runtime_error(
            "compute_cholesky: matrix is not positive-definite");
    }

    // L is lower-triangular (Eigen column-major storage).
    MatrixXd L_cpu = llt.matrixL();

    // Pack to flat float, row-major, full n x n.
    // Upper triangle stored as zero for simpler GPU indexing.
    std::vector<Scalar> L_flat(static_cast<size_t>(n) * n, 0.0f);
    for (Index i = 0; i < n; ++i) {
        for (Index j = 0; j <= i; ++j) {
            L_flat[static_cast<size_t>(i) * n + j] =
                static_cast<Scalar>(L_cpu(i, j));
        }
    }

    spdlog::debug("Cholesky decomposition: {} x {} -> L_flat {:.2f} KB", n, n,
                  L_flat.size() * sizeof(Scalar) / 1024.0);

    return CholeskyResult{std::move(L_cpu), std::move(L_flat), n};
}

bool validate_cholesky(const CholeskyResult& result, const MatrixXd& cov,
                       double tol) {
    MatrixXd reconstructed = result.L_cpu * result.L_cpu.transpose();
    double error = (reconstructed - cov).lpNorm<Eigen::Infinity>();
    spdlog::debug("Cholesky validation: ||LLT - cov||_inf = {:.2e}, tol = {:.2e}",
                  error, tol);
    return error < tol;
}

}  // namespace cpo
