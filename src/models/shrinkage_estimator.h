#pragma once

/// @file shrinkage_estimator.h
/// @brief Ledoit-Wolf optimal shrinkage covariance estimator.
///
/// Computes the optimal shrinkage intensity analytically from the data,
/// eliminating manual tuning. Shrinkage target: (trace(S)/N) * I (scaled
/// identity). Especially valuable when T (sample size) is close to N
/// (number of assets).
///
/// Reference: Ledoit & Wolf, "Honey, I Shrunk the Sample Covariance Matrix",
/// J. Portfolio Management 30(4), 2004. Theorem 1, Lemma 3.1.

#include "core/types.h"

namespace cpo {

/// Result of Ledoit-Wolf shrinkage estimation.
struct ShrinkageResult {
    MatrixXd covariance;    ///< Shrunk covariance matrix: (1-d)*S + d*mu*I.
    ScalarCPU intensity;    ///< Optimal shrinkage intensity in [0, 1].
    MatrixXd sample_cov;    ///< Unshrunk sample covariance (for diagnostics).
};

/// Compute the Ledoit-Wolf shrinkage covariance estimator.
///
/// Takes a return matrix (T x N) and computes:
///   1. Sample covariance S = (1/(T-1)) * X'X  (X centered)
///   2. Optimal shrinkage intensity d* via Lemma 3.1
///   3. Shrunk covariance: (1 - d*) * S + d* * (trace(S)/N) * I
///
/// All computation in double precision (ScalarCPU). Pure Eigen, no CUDA.
/// Complexity: O(T * N^2) — called once per estimation window.
///
/// @param returns Return matrix (T x N, double). Rows are time periods,
///                columns are assets. Need not be pre-centered.
/// @return ShrinkageResult with shrunk covariance, intensity, and sample cov.
/// @throws std::invalid_argument if T < 2 or N < 1.
ShrinkageResult ledoit_wolf_shrink(const MatrixXd& returns);

}  // namespace cpo
