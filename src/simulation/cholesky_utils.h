#pragma once

/// @file cholesky_utils.h
/// @brief Cholesky decomposition for covariance matrices.
///
/// Computes L such that Sigma = L * L^T using Eigen's LLT in double precision,
/// then packs L into a flat float array (row-major, full n x n) for GPU upload.
/// Full storage (not triangular-packed) simplifies GPU kernel indexing.
/// Cost: 500 x 500 = 1 MB in float32.

#include <stdexcept>
#include <vector>

#include "core/types.h"

namespace cpo {

/// Result of Cholesky decomposition.
struct CholeskyResult {
    /// Lower-triangular Cholesky factor in double precision (Eigen column-major).
    MatrixXd L_cpu;

    /// L packed as flat float array in row-major order (n x n).
    /// Element (i,j) is at index i * n + j.
    /// Upper-triangular entries are zero.
    std::vector<Scalar> L_flat;

    /// Dimension (number of assets).
    Index n;
};

/// Compute Cholesky decomposition of a covariance matrix.
///
/// Uses Eigen LLT in double precision for numerical stability, then converts
/// to a flat float row-major array for GPU upload.
///
/// @param cov Symmetric positive-definite covariance matrix (n x n, double).
/// @return CholeskyResult with L_cpu (double), L_flat (float row-major), and n.
/// @throws std::runtime_error if cov is not square or not positive-definite.
CholeskyResult compute_cholesky(const MatrixXd& cov);

/// Validate Cholesky decomposition: checks ||L * L^T - cov||_inf < tol.
///
/// @param result Cholesky result to validate.
/// @param cov Original covariance matrix.
/// @param tol Maximum allowed infinity-norm error (default: 1e-10).
/// @return true if reconstruction error is within tolerance.
bool validate_cholesky(const CholeskyResult& result, const MatrixXd& cov,
                       double tol = 1e-10);

}  // namespace cpo
