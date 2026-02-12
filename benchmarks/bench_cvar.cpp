#include <benchmark/benchmark.h>

#include <vector>

#include "risk/cvar.h"
#include "risk/device_vector.h"
#include "risk/portfolio_loss.h"
#include "simulation/cholesky_utils.h"
#include "simulation/monte_carlo.h"
#include "utils/cuda_utils.h"

using namespace cpo;

// ── Helper: build synthetic scenario data ───────────────────────────

static std::pair<VectorXd, CholeskyResult> make_test_data(Index n_assets) {
    VectorXd mu = VectorXd::Constant(n_assets, 0.05);

    // PD covariance: Sigma = D * R * D, off-diagonal rho = 0.3.
    MatrixXd R = MatrixXd::Constant(n_assets, n_assets, 0.3);
    for (Index i = 0; i < n_assets; ++i) R(i, i) = 1.0;

    VectorXd sigmas = VectorXd::Constant(n_assets, 0.20);
    MatrixXd D = sigmas.asDiagonal();
    MatrixXd cov = D * R * D;

    auto chol = compute_cholesky(cov);
    return {mu, std::move(chol)};
}

// ── GPU: portfolio loss computation ─────────────────────────────────

static void BM_PortfolioLossGPU(benchmark::State& state) {
    const Index n_scenarios = static_cast<Index>(state.range(0));
    const Index n_assets = static_cast<Index>(state.range(1));

    auto [mu, chol] = make_test_data(n_assets);

    MonteCarloConfig mc_cfg;
    mc_cfg.n_scenarios = n_scenarios;
    mc_cfg.seed = 42;
    auto scenarios = generate_scenarios_gpu(mu, chol, mc_cfg);

    VectorXs weights = VectorXs::Constant(n_assets, 1.0f / n_assets);

    for (auto _ : state) {
        auto d_losses = compute_portfolio_loss_gpu(scenarios, weights);
        benchmark::DoNotOptimize(d_losses.device_ptr());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(
        static_cast<int64_t>(n_scenarios) * state.iterations());
}

BENCHMARK(BM_PortfolioLossGPU)
    ->Args({10000, 50})
    ->Args({50000, 100})
    ->Args({100000, 100})
    ->Args({100000, 500})
    ->Unit(benchmark::kMillisecond);

// ── CPU: portfolio loss computation ─────────────────────────────────

static void BM_PortfolioLossCPU(benchmark::State& state) {
    const Index n_scenarios = static_cast<Index>(state.range(0));
    const Index n_assets = static_cast<Index>(state.range(1));

    auto [mu, chol] = make_test_data(n_assets);

    MonteCarloConfig mc_cfg;
    mc_cfg.n_scenarios = n_scenarios;
    mc_cfg.seed = 42;
    auto scenarios_host = generate_scenarios_cpu(mu, chol, mc_cfg);

    VectorXd weights = VectorXd::Constant(n_assets, 1.0 / n_assets);

    for (auto _ : state) {
        auto losses = compute_portfolio_loss_cpu(scenarios_host, weights);
        benchmark::DoNotOptimize(losses.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(
        static_cast<int64_t>(n_scenarios) * state.iterations());
}

BENCHMARK(BM_PortfolioLossCPU)
    ->Args({10000, 50})
    ->Args({50000, 100})
    ->Args({100000, 100})
    ->Unit(benchmark::kMillisecond);

// ── GPU: CVaR computation (from pre-computed losses) ────────────────

static void BM_CVaRGPU(benchmark::State& state) {
    const Index n_scenarios = static_cast<Index>(state.range(0));
    const Index n_assets = static_cast<Index>(state.range(1));

    auto [mu, chol] = make_test_data(n_assets);

    MonteCarloConfig mc_cfg;
    mc_cfg.n_scenarios = n_scenarios;
    mc_cfg.seed = 42;
    auto scenarios = generate_scenarios_gpu(mu, chol, mc_cfg);

    VectorXs weights = VectorXs::Constant(n_assets, 1.0f / n_assets);
    auto d_losses = compute_portfolio_loss_gpu(scenarios, weights);

    RiskConfig risk_cfg;
    risk_cfg.confidence_level = 0.95;

    for (auto _ : state) {
        auto result = compute_risk_gpu(d_losses, risk_cfg);
        benchmark::DoNotOptimize(result.cvar);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(
        static_cast<int64_t>(n_scenarios) * state.iterations());
}

BENCHMARK(BM_CVaRGPU)
    ->Args({10000, 50})
    ->Args({50000, 100})
    ->Args({100000, 100})
    ->Args({100000, 500})
    ->Unit(benchmark::kMillisecond);

// ── CPU: CVaR computation ───────────────────────────────────────────

static void BM_CVaRCPU(benchmark::State& state) {
    const Index n_scenarios = static_cast<Index>(state.range(0));
    const Index n_assets = static_cast<Index>(state.range(1));

    auto [mu, chol] = make_test_data(n_assets);

    MonteCarloConfig mc_cfg;
    mc_cfg.n_scenarios = n_scenarios;
    mc_cfg.seed = 42;
    auto scenarios_host = generate_scenarios_cpu(mu, chol, mc_cfg);

    VectorXd weights = VectorXd::Constant(n_assets, 1.0 / n_assets);
    VectorXd losses = compute_portfolio_loss_cpu(scenarios_host, weights);

    RiskConfig risk_cfg;
    risk_cfg.confidence_level = 0.95;

    for (auto _ : state) {
        auto result = compute_risk_cpu(losses, risk_cfg);
        benchmark::DoNotOptimize(result.cvar);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(
        static_cast<int64_t>(n_scenarios) * state.iterations());
}

BENCHMARK(BM_CVaRCPU)
    ->Args({10000, 50})
    ->Args({50000, 100})
    ->Args({100000, 100})
    ->Unit(benchmark::kMillisecond);

// ── GPU: full pipeline (scenarios -> loss -> CVaR) ──────────────────

static void BM_FullPipelineGPU(benchmark::State& state) {
    const Index n_scenarios = static_cast<Index>(state.range(0));
    const Index n_assets = static_cast<Index>(state.range(1));

    auto [mu, chol] = make_test_data(n_assets);

    MonteCarloConfig mc_cfg;
    mc_cfg.n_scenarios = n_scenarios;
    mc_cfg.seed = 42;
    auto scenarios = generate_scenarios_gpu(mu, chol, mc_cfg);

    VectorXs weights = VectorXs::Constant(n_assets, 1.0f / n_assets);

    RiskConfig risk_cfg;
    risk_cfg.confidence_level = 0.95;

    for (auto _ : state) {
        auto result = compute_portfolio_risk_gpu(scenarios, weights, risk_cfg);
        benchmark::DoNotOptimize(result.cvar);
        benchmark::ClobberMemory();
    }

    double items = static_cast<double>(n_scenarios) * n_assets;
    state.SetItemsProcessed(
        static_cast<int64_t>(items * state.iterations()));
}

BENCHMARK(BM_FullPipelineGPU)
    ->Args({10000, 50})
    ->Args({50000, 100})
    ->Args({100000, 100})
    ->Args({100000, 500})
    ->Unit(benchmark::kMillisecond);

// ── CPU: full pipeline ──────────────────────────────────────────────

static void BM_FullPipelineCPU(benchmark::State& state) {
    const Index n_scenarios = static_cast<Index>(state.range(0));
    const Index n_assets = static_cast<Index>(state.range(1));

    auto [mu, chol] = make_test_data(n_assets);

    MonteCarloConfig mc_cfg;
    mc_cfg.n_scenarios = n_scenarios;
    mc_cfg.seed = 42;
    auto scenarios_host = generate_scenarios_cpu(mu, chol, mc_cfg);

    VectorXd weights = VectorXd::Constant(n_assets, 1.0 / n_assets);

    RiskConfig risk_cfg;
    risk_cfg.confidence_level = 0.95;

    for (auto _ : state) {
        auto result = compute_portfolio_risk_cpu(scenarios_host, weights,
                                                 risk_cfg);
        benchmark::DoNotOptimize(result.cvar);
        benchmark::ClobberMemory();
    }

    double items = static_cast<double>(n_scenarios) * n_assets;
    state.SetItemsProcessed(
        static_cast<int64_t>(items * state.iterations()));
}

BENCHMARK(BM_FullPipelineCPU)
    ->Args({10000, 50})
    ->Args({50000, 100})
    ->Args({100000, 100})
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
