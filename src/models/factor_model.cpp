#include "models/factor_model.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <Eigen/Eigenvalues>
#include <spdlog/spdlog.h>

namespace cpo {

FactorModelResult fit_factor_model(const MatrixXd& returns,
                                    const FactorModelConfig& config) {
    const auto T = static_cast<Index>(returns.rows());
    const auto N = static_cast<Index>(returns.cols());

    if (T < 2) {
        throw std::invalid_argument(
            "fit_factor_model: need at least 2 periods, got " + std::to_string(T));
    }
    if (N < 1) {
        throw std::invalid_argument(
            "fit_factor_model: need at least 1 asset, got " + std::to_string(N));
    }

    int k = config.n_factors;
    if (config.min_variance_explained <= 0.0 && k > N) {
        throw std::invalid_argument(
            "fit_factor_model: n_factors (" + std::to_string(k) +
            ") exceeds n_assets (" + std::to_string(N) + ")");
    }

    // 1. Center returns: X = R - 1*mu'.
    VectorXd mu = returns.colwise().mean();
    MatrixXd X = returns.rowwise() - mu.transpose();

    // 2. Sample covariance: S = X'X / (T-1).
    MatrixXd S = (X.transpose() * X) / static_cast<ScalarCPU>(T - 1);

    // 3. Eigendecompose S.
    // SelfAdjointEigenSolver returns eigenvalues in ascending order.
    Eigen::SelfAdjointEigenSolver<MatrixXd> solver(S);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error(
            "fit_factor_model: eigendecomposition failed");
    }

    // Reverse to descending order.
    VectorXd eigenvalues_asc = solver.eigenvalues();
    MatrixXd eigenvectors_asc = solver.eigenvectors();

    VectorXd eigenvalues(N);
    MatrixXd eigenvectors(N, N);
    for (Index i = 0; i < N; ++i) {
        eigenvalues(i) = eigenvalues_asc(N - 1 - i);
        eigenvectors.col(i) = eigenvectors_asc.col(N - 1 - i);
    }

    // 4. Select number of factors k.
    ScalarCPU total_variance = eigenvalues.sum();

    if (config.min_variance_explained > 0.0) {
        // Auto-select k to explain at least the requested fraction.
        ScalarCPU cumulative = 0.0;
        k = 0;
        for (Index i = 0; i < N; ++i) {
            cumulative += eigenvalues(i);
            ++k;
            if (cumulative / total_variance >= config.min_variance_explained) {
                break;
            }
        }
        spdlog::info("Factor model: auto-selected k={} factors "
                     "(explains {:.1f}% of variance)",
                     k, 100.0 * cumulative / total_variance);
    } else {
        // Clamp k to at most N.
        k = std::min(k, N);
    }

    // 5. Extract loadings B = top-k eigenvectors (N x k).
    MatrixXd B = eigenvectors.leftCols(k);

    // 6. Factor returns: F = X * B  (T x k).
    MatrixXd F = X * B;

    // 7. Factor covariance: Sigma_f = F'F / (T-1)  (k x k).
    MatrixXd Sigma_f = (F.transpose() * F) / static_cast<ScalarCPU>(T - 1);

    // 8. Residuals and idiosyncratic variance.
    // E = X - F * B'  (T x N).
    MatrixXd E = X - F * B.transpose();

    // D_i = Var(E_i) = sum(E_i^2) / (T-1).
    VectorXd D(N);
    for (Index i = 0; i < N; ++i) {
        D(i) = E.col(i).squaredNorm() / static_cast<ScalarCPU>(T - 1);
        // Floor at 1e-10 to guarantee positive-definiteness of
        // the reconstructed covariance.
        if (D(i) < 1e-10) {
            D(i) = 1e-10;
        }
    }

    // Variance explained by top-k factors.
    ScalarCPU explained = eigenvalues.head(k).sum();
    ScalarCPU variance_explained = (total_variance > 0.0)
                                       ? explained / total_variance
                                       : 0.0;

    spdlog::debug("Factor model: N={}, T={}, k={}, variance_explained={:.4f}",
                  N, T, k, variance_explained);

    return FactorModelResult{
        std::move(B),
        std::move(F),
        std::move(Sigma_f),
        std::move(D),
        std::move(eigenvalues),
        std::move(mu),
        k,
        N,
        T,
        variance_explained
    };
}

MatrixXd reconstruct_covariance(const FactorModelResult& model) {
    // Sigma = B * Sigma_f * B' + diag(D)
    //
    // Reference: Connor & Korajczyk 1988, Eq. (2).
    MatrixXd cov = model.loadings * model.factor_covariance
                   * model.loadings.transpose();
    cov += model.idiosyncratic_var.asDiagonal();
    return cov;
}

CholeskyResult compute_cholesky_from_factor_model(
    const FactorModelResult& model) {
    MatrixXd cov = reconstruct_covariance(model);
    return compute_cholesky(cov);
}

}  // namespace cpo
