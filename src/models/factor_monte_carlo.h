#pragma once

/// @file factor_monte_carlo.h
/// @brief Factor-based Monte Carlo scenario generation on GPU and CPU.
///
/// Instead of full N x N Cholesky + O(N^2) correlation multiply per scenario,
/// generates scenarios through the factor structure:
///
///   1. f ~ N(0, Sigma_f) via k x k Cholesky       (cheap: k << N)
///   2. eps_i ~ N(0, D_i) independently per asset   (N independent draws)
///   3. r = mu + B * f + eps
///
/// Complexity comparison (per scenario):
///   Full Cholesky:   O(N^2)  multiply  (L * z, where L is N x N)
///   Factor model:    O(N*k)  multiply  (B * f) + O(N) for eps
///   For N=500, k=10: 5,000 vs 125,000 FLOPs — 25x reduction.
///
/// Memory comparison (device uploads):
///   Full Cholesky L:  N*N*4 bytes = 1 MB for N=500
///   Factor model:     B(N*k) + L_f(k*k) + sqrt_D(N) = ~22 KB for N=500, k=10
///
/// The generated scenarios are statistically identical to full-Cholesky
/// generation when Sigma = B * Sigma_f * B' + diag(D), since:
///   Cov(B*f + eps) = B * Sigma_f * B' + diag(D) = Sigma.
///
/// Reference: same mathematical foundation as the Cholesky approach —
///   correlated sampling via linear transformation of independent normals.

#include "core/types.h"
#include "models/factor_model.h"
#include "simulation/monte_carlo.h"
#include "simulation/scenario_matrix.h"

namespace cpo {

// ── GPU factor-based scenario generation ─────────────────────────────

/// Generate scenarios using the factor model structure on GPU.
///
/// Kernel design (one thread per scenario):
///   a) Generate k standard normals z_f[0..k-1]
///   b) Compute correlated factor return: f = L_f * z_f (k x k lower-tri)
///   c) For each asset i: generate z_e, write r_i = mu_i + B_i'*f + sqrt(D_i)*z_e
///
/// Shared memory loads B (N x k), mu (N), sqrt_D (N) per block.
/// For N=500, k=10: ~24 KB shared memory (fits in SM 86's 48 KB default).
///
/// @param mu Expected return vector (N, double — converted to float).
/// @param model Fitted factor model result.
/// @param config Monte Carlo configuration (n_scenarios, seed, etc.).
/// @param states Optional pre-allocated cuRAND states. If null, creates temporary.
/// @return ScenarioMatrix on GPU (column-major, n_scenarios x n_assets, float).
ScenarioMatrix generate_scenarios_factor_gpu(
    const VectorXd& mu,
    const FactorModelResult& model,
    const MonteCarloConfig& config,
    CurandStates* states = nullptr);

// ── CPU reference implementation ─────────────────────────────────────

/// Generate scenarios using the factor model structure on CPU.
///
/// Reference implementation for validation. Uses std::mt19937 in double
/// precision, same algorithm as the GPU kernel.
///
/// @param mu Expected return vector (N, double).
/// @param model Fitted factor model result.
/// @param config Monte Carlo configuration.
/// @return MatrixXd (n_scenarios x n_assets, double).
MatrixXd generate_scenarios_factor_cpu(
    const VectorXd& mu,
    const FactorModelResult& model,
    const MonteCarloConfig& config);

}  // namespace cpo
