#include <benchmark/benchmark.h>

#include <vector>

#include "simulation/cholesky_utils.h"
#include "simulation/monte_carlo.h"
#include "utils/cuda_utils.h"

using namespace cpo;

// ── Helper: build a synthetic PD covariance matrix ─────────────────

static std::pair<VectorXd, CholeskyResult> make_test_data(Index n_assets) {
    VectorXd mu = VectorXd::Constant(n_assets, 0.05);

    // Build PD covariance: Sigma = D * R * D.
    // R has off-diagonal correlation 0.3.
    MatrixXd R = MatrixXd::Constant(n_assets, n_assets, 0.3);
    for (Index i = 0; i < n_assets; ++i) R(i, i) = 1.0;

    VectorXd sigmas = VectorXd::Constant(n_assets, 0.20);
    MatrixXd D = sigmas.asDiagonal();
    MatrixXd cov = D * R * D;

    auto chol = compute_cholesky(cov);
    return {mu, std::move(chol)};
}

// ── GPU benchmarks ─────────────────────────────────────────────────

static void BM_MonteCarloGPU(benchmark::State& state) {
    const Index n_scenarios = static_cast<Index>(state.range(0));
    const Index n_assets = static_cast<Index>(state.range(1));

    auto [mu, chol] = make_test_data(n_assets);

    MonteCarloConfig cfg;
    cfg.n_scenarios = n_scenarios;
    cfg.seed = 42;

    // Pre-allocate cuRAND states outside the timing loop.
    auto curand_states = create_curand_states(n_scenarios, cfg.seed);

    for (auto _ : state) {
        auto scenarios = generate_scenarios_gpu(mu, chol, cfg,
                                                curand_states.get());
        benchmark::DoNotOptimize(scenarios.device_ptr());
        benchmark::ClobberMemory();
    }

    double items = static_cast<double>(n_scenarios) * n_assets;
    state.SetItemsProcessed(static_cast<int64_t>(items * state.iterations()));

    double vram_mb = static_cast<double>(n_scenarios) * n_assets *
                     sizeof(Scalar) / (1024.0 * 1024.0);
    state.counters["VRAM_MB"] =
        benchmark::Counter(vram_mb, benchmark::Counter::kDefaults);
}

// GPU: 10K/50K/100K scenarios x 50/100/500 assets
BENCHMARK(BM_MonteCarloGPU)
    ->Args({10000, 50})
    ->Args({10000, 100})
    ->Args({10000, 500})
    ->Args({50000, 50})
    ->Args({50000, 100})
    ->Args({50000, 500})
    ->Args({100000, 50})
    ->Args({100000, 100})
    ->Args({100000, 500})
    ->Unit(benchmark::kMillisecond);

// ── CPU benchmarks ─────────────────────────────────────────────────

static void BM_MonteCarloCPU(benchmark::State& state) {
    const Index n_scenarios = static_cast<Index>(state.range(0));
    const Index n_assets = static_cast<Index>(state.range(1));

    auto [mu, chol] = make_test_data(n_assets);

    MonteCarloConfig cfg;
    cfg.n_scenarios = n_scenarios;
    cfg.seed = 42;

    for (auto _ : state) {
        auto result = generate_scenarios_cpu(mu, chol, cfg);
        benchmark::DoNotOptimize(result.data());
        benchmark::ClobberMemory();
    }

    double items = static_cast<double>(n_scenarios) * n_assets;
    state.SetItemsProcessed(static_cast<int64_t>(items * state.iterations()));
}

// CPU: 10K/50K/100K x 50/100 (skip 500 — too slow for benchmarks)
BENCHMARK(BM_MonteCarloCPU)
    ->Args({10000, 50})
    ->Args({10000, 100})
    ->Args({50000, 50})
    ->Args({50000, 100})
    ->Args({100000, 50})
    ->Args({100000, 100})
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
