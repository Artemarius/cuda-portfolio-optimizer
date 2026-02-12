#pragma once

/// @file factor_model.h
/// @brief PCA-based factor model for covariance estimation.
///
/// Decomposes asset returns R (T x N) into systematic and idiosyncratic
/// components via Principal Component Analysis:
///
///   R = F * B' + E
///
/// where:
///   B  (N x k): factor loadings matrix (top-k eigenvectors of sample cov)
///   F  (T x k): factor return time series (centered returns projected onto B)
///   E  (T x N): residual matrix
///   D  (N):     diagonal idiosyncratic variance, D_i = Var(E_i)
///
/// Reconstructed covariance:
///   Sigma = B * Sigma_f * B' + diag(D)
///
/// where Sigma_f = Cov(F) is the k x k factor covariance.
///
/// Key advantage: reduces estimation from N*(N+1)/2 parameters (sample cov)
/// to N*k + k*(k+1)/2 + N parameters (loadings + factor cov + diagonal).
/// For N=500, k=10: 11,065 vs 125,250 — better conditioning when T < N.
///
/// References:
///   Connor & Korajczyk, "Risk and Return in an Equilibrium APT",
///     Journal of Financial Economics, 1988.
///   Jolliffe, "Principal Component Analysis", Springer, 2002.
///   Ledoit & Wolf, "Honey, I Shrunk the Sample Covariance Matrix",
///     Journal of Portfolio Management, 2004, Section 4.

#include "core/types.h"
#include "simulation/cholesky_utils.h"

namespace cpo {

/// Configuration for PCA-based factor model estimation.
struct FactorModelConfig {
    int n_factors = 10;  ///< Number of principal components to extract (k).

    /// If > 0, auto-select k to explain at least this fraction of total
    /// variance (overrides n_factors). E.g., 0.90 = 90%.
    ScalarCPU min_variance_explained = 0.0;
};

/// Result of PCA-based factor decomposition.
struct FactorModelResult {
    MatrixXd loadings;            ///< B: factor loadings (N x k).
    MatrixXd factor_returns;      ///< F: factor return time series (T x k).
    MatrixXd factor_covariance;   ///< Sigma_f: factor covariance (k x k).
    VectorXd idiosyncratic_var;   ///< D: per-asset residual variance (N).
    VectorXd eigenvalues;         ///< All eigenvalues sorted descending (N).
    VectorXd mu;                  ///< Sample mean of asset returns (N).
    int n_factors;                ///< Actual number of factors used (k).
    int n_assets;                 ///< Number of assets (N).
    int n_periods;                ///< Number of return periods (T).
    ScalarCPU variance_explained; ///< Fraction of total variance captured by k factors.
};

/// Fit a PCA-based factor model from a return matrix.
///
/// Algorithm:
///   1. Center returns: X = R - 1*mu'                     (T x N)
///   2. Sample covariance: S = X'X / (T-1)                (N x N)
///   3. Eigendecompose S: S = V * Lambda * V'              (Eigen SelfAdjointEigenSolver)
///   4. Select top-k eigenvectors: B = V[:, top-k]         (N x k)
///   5. Factor returns: F = X * B                          (T x k)
///   6. Factor covariance: Sigma_f = F'F / (T-1)          (k x k)
///   7. Residuals: E = X - F * B'                          (T x N)
///   8. Idiosyncratic variance: D_i = Var(E_i) for i=1..N
///
/// Uses Eigen::SelfAdjointEigenSolver on S (N x N). Eigenvalues from
/// SelfAdjointEigenSolver are ascending; we reverse to descending order.
///
/// @param returns Return matrix (T x N, double). Raw (non-centered) returns.
/// @param config  Factor model configuration.
/// @return FactorModelResult with all decomposition components.
/// @throws std::invalid_argument if T < 2, N < 1, or n_factors > N.
FactorModelResult fit_factor_model(const MatrixXd& returns,
                                    const FactorModelConfig& config = {});

/// Reconstruct the full N x N covariance matrix from the factor model.
///
///   Sigma = B * Sigma_f * B' + diag(D)
///
/// This is the primary output for plugging into the existing pipeline:
///   reconstruct_covariance() -> compute_cholesky() -> generate_scenarios_gpu()
///
/// @param model Fitted factor model result.
/// @return Reconstructed covariance matrix (N x N, double).
MatrixXd reconstruct_covariance(const FactorModelResult& model);

/// Compute Cholesky factor from a factor model result.
///
/// Convenience wrapper: reconstructs the full covariance then delegates
/// to the existing compute_cholesky(). For N=500 this is adequate — the
/// optimized path (factor Monte Carlo) avoids full Cholesky entirely.
///
/// @param model Fitted factor model result.
/// @return CholeskyResult compatible with existing simulation code.
CholeskyResult compute_cholesky_from_factor_model(const FactorModelResult& model);

}  // namespace cpo
