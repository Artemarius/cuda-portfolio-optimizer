#include "models/tiled_scenario_generator.h"

#include <algorithm>
#include <cmath>

#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

#include "models/factor_monte_carlo.h"
#include "simulation/monte_carlo.h"
#include "utils/cuda_utils.h"

namespace cpo {

// ── Tile size computation ────────────────────────────────────────────

/// Compute tile size based on available VRAM.
///
/// Each tile needs:
///   - ScenarioMatrix: tile_size * n_assets * sizeof(float)
///   - cuRAND states: tile_size * sizeof(curandState_t)  (~48 bytes each)
///
/// Additional device memory (Cholesky L or factor B/L_f/sqrt_D/mu) is
/// small relative to the scenario matrix.
///
/// @param n_assets Number of assets.
/// @param n_scenarios Total scenarios requested.
/// @param tiled_config Tiling parameters.
/// @return tile_size (clamped to [min_tile_size, n_scenarios]).
static Index compute_tile_size(Index n_assets, Index n_scenarios,
                                const TiledConfig& tiled_config) {
    size_t free_vram = get_free_vram();

    size_t usable = static_cast<size_t>(
        static_cast<double>(free_vram) * tiled_config.vram_fraction);

    // Per-scenario VRAM: scenario matrix + cuRAND state.
    size_t per_scenario = static_cast<size_t>(n_assets) * sizeof(Scalar) + 48;

    // Reserve 64 MB for Cholesky/factor uploads, CUB temp buffers, etc.
    constexpr size_t overhead = 64 * 1024 * 1024;
    if (usable <= overhead) {
        return std::max(tiled_config.min_tile_size,
                        static_cast<Index>(1024));
    }
    usable -= overhead;

    auto tile = static_cast<Index>(usable / per_scenario);
    tile = std::max(tile, tiled_config.min_tile_size);
    tile = std::min(tile, n_scenarios);

    return tile;
}

// ── Full Cholesky tiled generation ───────────────────────────────────

MatrixXd generate_scenarios_tiled(
    const VectorXd& mu,
    const CholeskyResult& cholesky,
    const MonteCarloConfig& mc_config,
    const TiledConfig& tiled_config) {

    const Index n_assets = cholesky.n;
    const Index n_scenarios = mc_config.n_scenarios;

    Index tile_size = compute_tile_size(n_assets, n_scenarios, tiled_config);

    if (tile_size >= n_scenarios) {
        // Everything fits — single GPU pass.
        spdlog::info("Tiled MC: single pass ({} scenarios fit in VRAM)",
                     n_scenarios);
        auto curand_states = create_curand_states(n_scenarios, mc_config.seed);
        auto gpu_scenarios = generate_scenarios_gpu(
            mu, cholesky, mc_config, curand_states.get());
        MatrixXs host_float = gpu_scenarios.to_host();
        return host_float.cast<double>();
    }

    // Tiled generation.
    int n_tiles = (n_scenarios + tile_size - 1) / tile_size;
    spdlog::info("Tiled MC: {} tiles x {} scenarios (total {})",
                 n_tiles, tile_size, n_scenarios);

    MatrixXd result(n_scenarios, n_assets);

    // Allocate cuRAND states for tile_size (reused across tiles).
    auto curand_states = create_curand_states(tile_size, mc_config.seed);

    for (int t = 0; t < n_tiles; ++t) {
        Index offset = t * tile_size;
        Index this_tile = std::min(tile_size, n_scenarios - offset);

        MonteCarloConfig tile_cfg = mc_config;
        tile_cfg.n_scenarios = this_tile;

        auto gpu_scenarios = generate_scenarios_gpu(
            mu, cholesky, tile_cfg, curand_states.get());
        MatrixXs host_float = gpu_scenarios.to_host();

        // Copy tile into the result matrix.
        result.block(offset, 0, this_tile, n_assets) =
            host_float.cast<double>();

        spdlog::debug("Tiled MC: tile {}/{} done ({} scenarios)",
                      t + 1, n_tiles, this_tile);
    }

    return result;
}

// ── Factor model tiled generation ────────────────────────────────────

MatrixXd generate_scenarios_factor_tiled(
    const VectorXd& mu,
    const FactorModelResult& model,
    const MonteCarloConfig& mc_config,
    const TiledConfig& tiled_config) {

    const Index n_assets = model.n_assets;
    const Index n_scenarios = mc_config.n_scenarios;

    Index tile_size = compute_tile_size(n_assets, n_scenarios, tiled_config);

    if (tile_size >= n_scenarios) {
        // Everything fits — single GPU pass.
        spdlog::info("Tiled factor MC: single pass ({} scenarios fit in VRAM)",
                     n_scenarios);
        auto curand_states = create_curand_states(n_scenarios, mc_config.seed);
        auto gpu_scenarios = generate_scenarios_factor_gpu(
            mu, model, mc_config, curand_states.get());
        MatrixXs host_float = gpu_scenarios.to_host();
        return host_float.cast<double>();
    }

    // Tiled generation.
    int n_tiles = (n_scenarios + tile_size - 1) / tile_size;
    spdlog::info("Tiled factor MC: {} tiles x {} scenarios (total {})",
                 n_tiles, tile_size, n_scenarios);

    MatrixXd result(n_scenarios, n_assets);

    // Allocate cuRAND states for tile_size (reused across tiles).
    auto curand_states = create_curand_states(tile_size, mc_config.seed);

    for (int t = 0; t < n_tiles; ++t) {
        Index offset = t * tile_size;
        Index this_tile = std::min(tile_size, n_scenarios - offset);

        MonteCarloConfig tile_cfg = mc_config;
        tile_cfg.n_scenarios = this_tile;

        auto gpu_scenarios = generate_scenarios_factor_gpu(
            mu, model, tile_cfg, curand_states.get());
        MatrixXs host_float = gpu_scenarios.to_host();

        // Copy tile into the result matrix.
        result.block(offset, 0, this_tile, n_assets) =
            host_float.cast<double>();

        spdlog::debug("Tiled factor MC: tile {}/{} done ({} scenarios)",
                      t + 1, n_tiles, this_tile);
    }

    return result;
}

}  // namespace cpo
