#pragma once

/// @file tiled_scenario_generator.h
/// @brief VRAM-safe tiled scenario generation for large configurations.
///
/// When n_scenarios * n_assets * sizeof(float) exceeds available VRAM,
/// scenarios are generated in tiles that each fit in GPU memory. Each tile
/// is generated on the GPU, downloaded to host, and the GPU memory is reused
/// for the next tile. cuRAND states are allocated for tile_size (not
/// n_scenarios), further reducing peak VRAM.
///
/// For configurations that fit in VRAM, delegates directly to the non-tiled
/// GPU generator with no overhead.

#include "core/types.h"
#include "models/factor_model.h"
#include "simulation/cholesky_utils.h"
#include "simulation/monte_carlo.h"

namespace cpo {

/// Configuration for tiled scenario generation.
struct TiledConfig {
    ScalarCPU vram_fraction = 0.7;  ///< Fraction of free VRAM to use (0, 1].
    Index min_tile_size = 1024;     ///< Minimum tile size (scenarios per tile).
};

/// Generate scenarios using full Cholesky MC, tiling if needed.
///
/// Queries free VRAM, computes tile size, and generates in chunks.
/// Returns a CPU-side MatrixXd (n_scenarios x n_assets, double).
///
/// @param mu Expected return vector (N, double).
/// @param cholesky Precomputed Cholesky factor.
/// @param mc_config Monte Carlo configuration.
/// @param tiled_config Tiling parameters.
/// @return MatrixXd (n_scenarios x n_assets, double).
MatrixXd generate_scenarios_tiled(
    const VectorXd& mu,
    const CholeskyResult& cholesky,
    const MonteCarloConfig& mc_config,
    const TiledConfig& tiled_config = {});

/// Generate scenarios using factor MC, tiling if needed.
///
/// Same tiling logic as generate_scenarios_tiled(), but uses the factor
/// Monte Carlo kernel (O(Nk) per scenario instead of O(N^2)).
///
/// @param mu Expected return vector (N, double).
/// @param model Fitted factor model result.
/// @param mc_config Monte Carlo configuration.
/// @param tiled_config Tiling parameters.
/// @return MatrixXd (n_scenarios x n_assets, double).
MatrixXd generate_scenarios_factor_tiled(
    const VectorXd& mu,
    const FactorModelResult& model,
    const MonteCarloConfig& mc_config,
    const TiledConfig& tiled_config = {});

}  // namespace cpo
