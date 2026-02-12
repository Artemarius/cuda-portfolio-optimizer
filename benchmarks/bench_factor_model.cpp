#include <benchmark/benchmark.h>

#include <random>
#include <vector>

#include "models/factor_model.h"
#include "models/factor_monte_carlo.h"
#include "models/tiled_scenario_generator.h"
#include "optimizer/admm_solver.h"
#include "simulation/cholesky_utils.h"
#include "simulation/monte_carlo.h"
#include "utils/cuda_utils.h"

using namespace cpo;

// ── Helper: generate synthetic returns ──────────────────────────────

static MatrixXd make_synthetic_returns(Index n_assets, Index n_periods,
                                        unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    // Build a factor structure: k = 10 factors, then add noise.
    const int k = std::min(10, static_cast<int>(n_assets));

    MatrixXd B(n_assets, k);
    for (Index i = 0; i < n_assets; ++i) {
        for (int j = 0; j < k; ++j) {
            B(i, j) = 0.3 * normal(rng);
        }
    }

    MatrixXd F(n_periods, k);
    for (Index t = 0; t < n_periods; ++t) {
        for (int j = 0; j < k; ++j) {
            F(t, j) = normal(rng);
        }
    }

    MatrixXd returns = F * B.transpose();
    // Add idiosyncratic noise.
    for (Index t = 0; t < n_periods; ++t) {
        for (Index i = 0; i < n_assets; ++i) {
            returns(t, i) += 0.01 * normal(rng);
        }
    }

    // Add small mean.
    VectorXd mu = VectorXd::Constant(n_assets, 0.0005);
    returns.rowwise() += mu.transpose();

    return returns;
}

// ── Helper: build PD covariance for Cholesky MC baseline ──────────

static std::pair<VectorXd, CholeskyResult> make_chol_data(Index n_assets) {
    VectorXd mu = VectorXd::Constant(n_assets, 0.05);

    MatrixXd R = MatrixXd::Constant(n_assets, n_assets, 0.3);
    for (Index i = 0; i < n_assets; ++i) R(i, i) = 1.0;

    VectorXd sigmas = VectorXd::Constant(n_assets, 0.20);
    MatrixXd D = sigmas.asDiagonal();
    MatrixXd cov = D * R * D;

    auto chol = compute_cholesky(cov);
    return {mu, std::move(chol)};
}

// ════════════════════════════════════════════════════════════════════
// Factor model fitting benchmarks
// ════════════════════════════════════════════════════════════════════

static void BM_FactorModelFit(benchmark::State& state) {
    const Index n_assets = static_cast<Index>(state.range(0));
    const Index n_periods = 252;
    const int k = static_cast<int>(state.range(1));

    MatrixXd returns = make_synthetic_returns(n_assets, n_periods);

    FactorModelConfig config;
    config.n_factors = k;

    for (auto _ : state) {
        auto model = fit_factor_model(returns, config);
        benchmark::DoNotOptimize(model.loadings.data());
        benchmark::ClobberMemory();
    }

    state.counters["assets"] =
        benchmark::Counter(n_assets, benchmark::Counter::kDefaults);
    state.counters["factors"] =
        benchmark::Counter(k, benchmark::Counter::kDefaults);
}

BENCHMARK(BM_FactorModelFit)
    ->Args({50, 10})
    ->Args({100, 10})
    ->Args({500, 10})
    ->Unit(benchmark::kMillisecond);

// ════════════════════════════════════════════════════════════════════
// Covariance reconstruction benchmarks
// ════════════════════════════════════════════════════════════════════

static void BM_CovarianceReconstruction(benchmark::State& state) {
    const Index n_assets = static_cast<Index>(state.range(0));
    const int k = static_cast<int>(state.range(1));

    MatrixXd returns = make_synthetic_returns(n_assets, 252);

    FactorModelConfig config;
    config.n_factors = k;
    auto model = fit_factor_model(returns, config);

    for (auto _ : state) {
        auto cov = reconstruct_covariance(model);
        benchmark::DoNotOptimize(cov.data());
        benchmark::ClobberMemory();
    }

    state.counters["assets"] =
        benchmark::Counter(n_assets, benchmark::Counter::kDefaults);
}

BENCHMARK(BM_CovarianceReconstruction)
    ->Args({50, 10})
    ->Args({100, 10})
    ->Args({500, 10})
    ->Unit(benchmark::kMillisecond);

// ════════════════════════════════════════════════════════════════════
// Factor MC GPU vs full Cholesky MC GPU
// ════════════════════════════════════════════════════════════════════

static void BM_FactorMC_GPU(benchmark::State& state) {
    const Index n_scenarios = static_cast<Index>(state.range(0));
    const Index n_assets = static_cast<Index>(state.range(1));

    MatrixXd returns = make_synthetic_returns(n_assets, 252);

    FactorModelConfig config;
    config.n_factors = 10;
    auto model = fit_factor_model(returns, config);

    MonteCarloConfig mc;
    mc.n_scenarios = n_scenarios;
    mc.seed = 42;

    auto curand_states = create_curand_states(n_scenarios, mc.seed);

    for (auto _ : state) {
        auto scenarios = generate_scenarios_factor_gpu(
            model.mu, model, mc, curand_states.get());
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

BENCHMARK(BM_FactorMC_GPU)
    ->Args({100000, 50})
    ->Args({100000, 100})
    ->Args({100000, 500})
    ->Unit(benchmark::kMillisecond);

static void BM_FullCholeskyMC_GPU(benchmark::State& state) {
    const Index n_scenarios = static_cast<Index>(state.range(0));
    const Index n_assets = static_cast<Index>(state.range(1));

    auto [mu, chol] = make_chol_data(n_assets);

    MonteCarloConfig mc;
    mc.n_scenarios = n_scenarios;
    mc.seed = 42;

    auto curand_states = create_curand_states(n_scenarios, mc.seed);

    for (auto _ : state) {
        auto scenarios = generate_scenarios_gpu(
            mu, chol, mc, curand_states.get());
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

BENCHMARK(BM_FullCholeskyMC_GPU)
    ->Args({100000, 50})
    ->Args({100000, 100})
    ->Args({100000, 500})
    ->Unit(benchmark::kMillisecond);

// ════════════════════════════════════════════════════════════════════
// Factor MC CPU
// ════════════════════════════════════════════════════════════════════

static void BM_FactorMC_CPU(benchmark::State& state) {
    const Index n_scenarios = static_cast<Index>(state.range(0));
    const Index n_assets = static_cast<Index>(state.range(1));

    MatrixXd returns = make_synthetic_returns(n_assets, 252);

    FactorModelConfig config;
    config.n_factors = 10;
    auto model = fit_factor_model(returns, config);

    MonteCarloConfig mc;
    mc.n_scenarios = n_scenarios;
    mc.seed = 42;

    for (auto _ : state) {
        auto result = generate_scenarios_factor_cpu(model.mu, model, mc);
        benchmark::DoNotOptimize(result.data());
        benchmark::ClobberMemory();
    }

    double items = static_cast<double>(n_scenarios) * n_assets;
    state.SetItemsProcessed(static_cast<int64_t>(items * state.iterations()));
}

BENCHMARK(BM_FactorMC_CPU)
    ->Args({100000, 50})
    ->Args({100000, 100})
    ->Args({100000, 500})
    ->Unit(benchmark::kMillisecond);

// ════════════════════════════════════════════════════════════════════
// End-to-end 500-asset benchmark: factor fit + factor MC GPU + ADMM
// ════════════════════════════════════════════════════════════════════

static void BM_EndToEnd_500Asset(benchmark::State& state) {
    const Index n_assets = 500;
    const Index n_scenarios = static_cast<Index>(state.range(0));

    MatrixXd returns = make_synthetic_returns(n_assets, 252);

    FactorModelConfig fc;
    fc.n_factors = 10;

    MonteCarloConfig mc;
    mc.n_scenarios = n_scenarios;
    mc.seed = 42;

    AdmmConfig admm;
    admm.confidence_level = 0.95;
    admm.max_iter = 200;

    for (auto _ : state) {
        // 1. Fit factor model.
        auto model = fit_factor_model(returns, fc);

        // 2. Generate scenarios on GPU.
        auto curand_states = create_curand_states(n_scenarios, mc.seed);
        auto gpu_scenarios = generate_scenarios_factor_gpu(
            model.mu, model, mc, curand_states.get());
        MatrixXs host_float = gpu_scenarios.to_host();
        MatrixXd scenarios = host_float.cast<double>();

        // 3. ADMM solve.
        auto result = admm_solve(scenarios, model.mu, admm);

        benchmark::DoNotOptimize(result.weights.data());
        benchmark::ClobberMemory();
    }

    state.counters["assets"] =
        benchmark::Counter(n_assets, benchmark::Counter::kDefaults);
    state.counters["scenarios"] =
        benchmark::Counter(n_scenarios, benchmark::Counter::kDefaults);
}

BENCHMARK(BM_EndToEnd_500Asset)
    ->Args({50000})
    ->Args({100000})
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
