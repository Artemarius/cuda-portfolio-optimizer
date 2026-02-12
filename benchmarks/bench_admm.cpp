#include <benchmark/benchmark.h>

#include <vector>

#include "optimizer/admm_solver.h"
#include "optimizer/efficient_frontier.h"
#include "simulation/cholesky_utils.h"
#include "simulation/monte_carlo.h"

using namespace cpo;

// ── Helper: build a synthetic PD covariance matrix ─────────────────

static std::pair<VectorXd, CholeskyResult> make_test_data(Index n_assets) {
    VectorXd mu(n_assets);
    for (Index i = 0; i < n_assets; ++i) {
        mu(i) = 0.02 + 0.02 * i / std::max(1, n_assets - 1);
    }

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

// ── ADMM single-solve benchmarks ──────────────────────────────────

static void BM_AdmmSolve(benchmark::State& state) {
    const Index n_assets = static_cast<Index>(state.range(0));
    const Index n_scenarios = static_cast<Index>(state.range(1));

    auto [mu, chol] = make_test_data(n_assets);

    MonteCarloConfig mc_cfg;
    mc_cfg.n_scenarios = n_scenarios;
    mc_cfg.seed = 42;
    MatrixXd scenarios = generate_scenarios_cpu(mu, chol, mc_cfg);

    AdmmConfig admm_cfg;
    admm_cfg.confidence_level = 0.95;
    admm_cfg.max_iter = 300;

    for (auto _ : state) {
        auto result = admm_solve(scenarios, mu, admm_cfg);
        benchmark::DoNotOptimize(result.weights.data());
        benchmark::ClobberMemory();
        state.counters["iterations"] =
            benchmark::Counter(static_cast<double>(result.iterations),
                               benchmark::Counter::kDefaults);
        state.counters["converged"] =
            benchmark::Counter(result.converged ? 1.0 : 0.0,
                               benchmark::Counter::kDefaults);
    }
}

// 2/5/10 assets x 10K/50K scenarios (6 configs)
BENCHMARK(BM_AdmmSolve)
    ->Args({2, 10000})
    ->Args({2, 50000})
    ->Args({5, 10000})
    ->Args({5, 50000})
    ->Args({10, 10000})
    ->Args({10, 50000})
    ->Unit(benchmark::kMillisecond);

// ── Efficient frontier benchmarks ─────────────────────────────────

static void BM_EfficientFrontier(benchmark::State& state) {
    const Index n_assets = static_cast<Index>(state.range(0));
    const Index n_scenarios = 20000;

    auto [mu, chol] = make_test_data(n_assets);

    MonteCarloConfig mc_cfg;
    mc_cfg.n_scenarios = n_scenarios;
    mc_cfg.seed = 42;
    MatrixXd scenarios = generate_scenarios_cpu(mu, chol, mc_cfg);

    FrontierConfig f_cfg;
    f_cfg.n_points = 5;
    f_cfg.admm_config.confidence_level = 0.95;
    f_cfg.admm_config.max_iter = 200;

    for (auto _ : state) {
        auto frontier = compute_efficient_frontier(scenarios, mu, f_cfg);
        benchmark::DoNotOptimize(frontier.data());
        benchmark::ClobberMemory();

        int total_iters = 0;
        for (const auto& pt : frontier) total_iters += pt.iterations;
        state.counters["total_iterations"] =
            benchmark::Counter(static_cast<double>(total_iters),
                               benchmark::Counter::kDefaults);
    }
}

// 5-point frontier for 3/5/10 assets
BENCHMARK(BM_EfficientFrontier)
    ->Args({3})
    ->Args({5})
    ->Args({10})
    ->Unit(benchmark::kMillisecond);

// ── Full pipeline: scenario generation + ADMM solve ───────────────

static void BM_FullPipeline(benchmark::State& state) {
    const Index n_assets = static_cast<Index>(state.range(0));
    const Index n_scenarios = static_cast<Index>(state.range(1));

    auto [mu, chol] = make_test_data(n_assets);

    MonteCarloConfig mc_cfg;
    mc_cfg.n_scenarios = n_scenarios;
    mc_cfg.seed = 42;

    AdmmConfig admm_cfg;
    admm_cfg.confidence_level = 0.95;
    admm_cfg.max_iter = 300;

    for (auto _ : state) {
        // Scenario generation + ADMM solve together.
        MatrixXd scenarios = generate_scenarios_cpu(mu, chol, mc_cfg);
        auto result = admm_solve(scenarios, mu, admm_cfg);
        benchmark::DoNotOptimize(result.weights.data());
        benchmark::ClobberMemory();
    }
}

// Full pipeline: 5/10 assets x 50K scenarios
BENCHMARK(BM_FullPipeline)
    ->Args({5, 50000})
    ->Args({10, 50000})
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
