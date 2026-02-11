#pragma once

/// @file monte_carlo.h
/// @brief GPU and CPU Monte Carlo scenario generation.
///
/// Generates correlated return scenarios using Cholesky decomposition:
///   r = mu + L * z,  where z ~ N(0, I)
///
/// GPU path uses cuRAND for RNG and a custom kernel for the Cholesky multiply.
/// CPU path uses std::mt19937 + std::normal_distribution in double precision.
///
/// Reference: Cholesky decomposition for correlated sampling —
///   if Sigma = L * L^T, then L * z (z ~ N(0,I)) has covariance Sigma.

#include <cstdint>
#include <memory>

#include "core/types.h"
#include "simulation/cholesky_utils.h"
#include "simulation/scenario_matrix.h"

namespace cpo {

/// Configuration for Monte Carlo scenario generation.
struct MonteCarloConfig {
    Index n_scenarios = 100000;
    Index n_assets = 0;      ///< Inferred from mu/L if zero.
    uint64_t seed = 42;
    int threads_per_block = 256;
};

// ── cuRAND state management ────────────────────────────────────────
// Opaque type — curand_kernel.h stays in .cu files only.

/// Opaque handle to cuRAND device states.
struct CurandStates;

/// Free cuRAND device states.
void destroy_curand_states(CurandStates* states);

/// Custom deleter for use with unique_ptr (handles incomplete type).
struct CurandStatesDeleter {
    void operator()(CurandStates* p) const { destroy_curand_states(p); }
};

/// RAII guard for cuRAND states. Move-only.
using CurandStatesGuard = std::unique_ptr<CurandStates, CurandStatesDeleter>;

/// Allocate and initialize cuRAND states on the GPU.
/// @param n_states Number of RNG states (one per thread/scenario).
/// @param seed Base seed for initialization.
/// @return RAII guard owning the device-allocated states.
CurandStatesGuard create_curand_states(Index n_states, uint64_t seed);

// ── GPU scenario generation ────────────────────────────────────────

/// Generate correlated return scenarios on the GPU.
///
/// Each scenario i: r_i = mu + L * z_i, where z_i ~ N(0, I).
/// Result is a column-major ScenarioMatrix (n_scenarios x n_assets, float).
///
/// @param mu Expected return vector (n_assets, double — converted to float).
/// @param cholesky Cholesky result containing L_flat (float row-major).
/// @param config Monte Carlo configuration (n_scenarios, seed, etc.).
/// @param states Optional pre-allocated cuRAND states. If null, creates temporary.
/// @return ScenarioMatrix on the GPU.
ScenarioMatrix generate_scenarios_gpu(const VectorXd& mu,
                                      const CholeskyResult& cholesky,
                                      const MonteCarloConfig& config,
                                      CurandStates* states = nullptr);

// ── CPU reference implementation ───────────────────────────────────

/// Generate correlated return scenarios on the CPU (double precision).
///
/// Reference implementation for validation. Uses std::mt19937 + normal_distribution.
///
/// @param mu Expected return vector (n_assets, double).
/// @param cholesky Cholesky result containing L_cpu (double).
/// @param config Monte Carlo configuration.
/// @return MatrixXd (n_scenarios x n_assets) in double precision.
MatrixXd generate_scenarios_cpu(const VectorXd& mu,
                                const CholeskyResult& cholesky,
                                const MonteCarloConfig& config);

}  // namespace cpo
