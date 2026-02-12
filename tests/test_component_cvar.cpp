#include <gtest/gtest.h>

#include <cmath>
#include <numeric>
#include <vector>

#include "risk/component_cvar.h"
#include "risk/cvar.h"
#include "risk/device_vector.h"
#include "risk/portfolio_loss.h"
#include "simulation/cholesky_utils.h"
#include "simulation/monte_carlo.h"
#include "simulation/scenario_matrix.h"

using namespace cpo;

// ── Test 1: Deterministic hand-computed example ─────────────────────

TEST(ComponentCVaR, Deterministic) {
    // 2 assets, 5 scenarios.
    // Scenario returns (row = scenario, col = asset):
    //   s0: [ 0.10,  0.05]  -> loss = -(0.6*0.10 + 0.4*0.05) = -0.08
    //   s1: [-0.05,  0.02]  -> loss = -(0.6*(-0.05) + 0.4*0.02) = 0.022
    //   s2: [ 0.03, -0.01]  -> loss = -(0.6*0.03 + 0.4*(-0.01)) = -0.014
    //   s3: [-0.02, -0.03]  -> loss = -(0.6*(-0.02) + 0.4*(-0.03)) = 0.024
    //   s4: [-0.08, -0.06]  -> loss = -(0.6*(-0.08) + 0.4*(-0.06)) = 0.072
    //
    // Sorted losses: [-0.08, -0.014, 0.022, 0.024, 0.072]
    // alpha = 0.60: var_index = floor(0.6 * 5) = 3
    //   VaR = sorted[3] = 0.024
    //   Tail = {s3 (0.024), s4 (0.072)} — 2 scenarios
    //   CVaR = (0.024 + 0.072) / 2 = 0.048
    //
    // Component CVaR (tail scenarios: s3 and s4):
    //   Asset 0: (1/2) * [0.6*(-(-0.02)) + 0.6*(-(-0.08))]
    //          = (1/2) * [0.6*0.02 + 0.6*0.08]
    //          = (1/2) * [0.012 + 0.048] = 0.030
    //   Asset 1: (1/2) * [0.4*(-(-0.03)) + 0.4*(-(-0.06))]
    //          = (1/2) * [0.4*0.03 + 0.4*0.06]
    //          = (1/2) * [0.012 + 0.024] = 0.018
    //   Sum = 0.030 + 0.018 = 0.048 = CVaR ✓

    const Index n_scenarios = 5;
    const Index n_assets = 2;

    // CPU path.
    MatrixXd scenarios_host(5, 2);
    scenarios_host << 0.10, 0.05,
                     -0.05, 0.02,
                      0.03, -0.01,
                     -0.02, -0.03,
                     -0.08, -0.06;

    VectorXd weights_d(2);
    weights_d << 0.6, 0.4;

    VectorXd losses = compute_portfolio_loss_cpu(scenarios_host, weights_d);

    RiskConfig cfg;
    cfg.confidence_level = 0.60;
    auto risk = compute_risk_cpu(losses, cfg);

    // Verify VaR and CVaR first.
    EXPECT_NEAR(risk.var, 0.024, 1e-10);
    EXPECT_NEAR(risk.cvar, 0.048, 1e-10);

    auto component = compute_component_cvar_cpu(
        scenarios_host, weights_d, losses, risk.var, cfg);

    ASSERT_EQ(component.size(), 2);
    EXPECT_NEAR(component(0), 0.030, 1e-10);
    EXPECT_NEAR(component(1), 0.018, 1e-10);
    EXPECT_NEAR(component.sum(), risk.cvar, 1e-10);
}

// ── Test 2: Sum property — sum(component_cvar) == total CVaR ────────

TEST(ComponentCVaR, SumProperty) {
    // 5 assets, 50K scenarios — statistical test.
    const Index n_assets = 5;
    const Index n_scenarios = 50000;

    VectorXd mu(n_assets);
    mu << 0.02, 0.04, 0.06, 0.03, 0.05;

    VectorXd sigmas(n_assets);
    sigmas << 0.15, 0.20, 0.25, 0.18, 0.22;

    MatrixXd R = MatrixXd::Identity(n_assets, n_assets);
    R(0, 1) = R(1, 0) = 0.5;
    R(0, 2) = R(2, 0) = 0.3;
    R(1, 2) = R(2, 1) = 0.4;
    R(2, 3) = R(3, 2) = 0.2;
    R(3, 4) = R(4, 3) = 0.35;
    MatrixXd D = sigmas.asDiagonal();
    MatrixXd cov = D * R * D;
    auto chol = compute_cholesky(cov);

    MonteCarloConfig mc_cfg;
    mc_cfg.n_scenarios = n_scenarios;
    mc_cfg.seed = 12345;
    MatrixXd scenarios = generate_scenarios_cpu(mu, chol, mc_cfg);

    VectorXd weights(n_assets);
    weights << 0.25, 0.20, 0.15, 0.25, 0.15;

    RiskConfig risk_cfg;
    risk_cfg.confidence_level = 0.95;

    auto [risk, component] = compute_portfolio_risk_decomp_cpu(
        scenarios, weights, risk_cfg);

    // sum(component_cvar) == total CVaR within tight tolerance.
    EXPECT_NEAR(component.sum(), risk.cvar, 1e-10)
        << "Sum of component CVaR (" << component.sum()
        << ") != total CVaR (" << risk.cvar << ")";
}

// ── Test 3: Zero weight — asset with w=0 contributes zero ───────────

TEST(ComponentCVaR, ZeroWeight) {
    const Index n_scenarios = 1000;
    const Index n_assets = 3;

    VectorXd mu(n_assets);
    mu << 0.03, 0.05, 0.04;

    MatrixXd cov = MatrixXd::Identity(n_assets, n_assets) * 0.04;
    auto chol = compute_cholesky(cov);

    MonteCarloConfig mc_cfg;
    mc_cfg.n_scenarios = n_scenarios;
    mc_cfg.seed = 999;
    MatrixXd scenarios = generate_scenarios_cpu(mu, chol, mc_cfg);

    // Asset 1 has zero weight.
    VectorXd weights(n_assets);
    weights << 0.5, 0.0, 0.5;

    RiskConfig risk_cfg;
    risk_cfg.confidence_level = 0.95;

    auto [risk, component] = compute_portfolio_risk_decomp_cpu(
        scenarios, weights, risk_cfg);

    EXPECT_NEAR(component(1), 0.0, 1e-15)
        << "Zero-weight asset should have zero component CVaR";
    EXPECT_NEAR(component.sum(), risk.cvar, 1e-10);
}

// ── Test 4: GPU vs CPU parity ───────────────────────────────────────

TEST(ComponentCVaR, GPUvsCPU) {
    const Index n_assets = 3;
    const Index n_scenarios = 50000;

    VectorXd mu(n_assets);
    mu << 0.03, 0.06, 0.09;

    VectorXd sigmas(n_assets);
    sigmas << 0.15, 0.20, 0.25;

    MatrixXd R = MatrixXd::Identity(n_assets, n_assets);
    R(0, 1) = R(1, 0) = 0.5;
    R(0, 2) = R(2, 0) = 0.3;
    R(1, 2) = R(2, 1) = 0.4;
    MatrixXd D = sigmas.asDiagonal();
    MatrixXd cov = D * R * D;
    auto chol = compute_cholesky(cov);

    // Generate on CPU, upload to GPU.
    MonteCarloConfig mc_cfg;
    mc_cfg.n_scenarios = n_scenarios;
    mc_cfg.seed = 7777;
    MatrixXd cpu_scenarios = generate_scenarios_cpu(mu, chol, mc_cfg);

    ScenarioMatrix gpu_scenarios(n_scenarios, n_assets);
    MatrixXs float_scenarios = cpu_scenarios.cast<float>();
    gpu_scenarios.from_host(float_scenarios);

    VectorXd weights_d(n_assets);
    weights_d << 0.4, 0.35, 0.25;
    VectorXs weights_f = weights_d.cast<float>();

    RiskConfig cfg;
    cfg.confidence_level = 0.95;

    auto [cpu_risk, cpu_component] = compute_portfolio_risk_decomp_cpu(
        cpu_scenarios, weights_d, cfg);
    auto [gpu_risk, gpu_component] = compute_portfolio_risk_decomp_gpu(
        gpu_scenarios, weights_f, cfg);

    // Component CVaR parity (float vs double tolerance).
    for (Index j = 0; j < n_assets; ++j) {
        double tol = std::max(1e-3, 0.02 * std::abs(cpu_component(j)));
        EXPECT_NEAR(gpu_component(j), cpu_component(j), tol)
            << "Component CVaR mismatch at asset " << j
            << ": GPU=" << gpu_component(j) << " CPU=" << cpu_component(j);
    }

    // Sum property holds on GPU path too.
    EXPECT_NEAR(gpu_component.sum(), gpu_risk.cvar, 0.01)
        << "GPU sum of component CVaR (" << gpu_component.sum()
        << ") != GPU total CVaR (" << gpu_risk.cvar << ")";
}

// ── Test 5: Single asset — component_cvar[0] == total CVaR ──────────

TEST(ComponentCVaR, SingleAsset) {
    const Index n_scenarios = 5;
    const Index n_assets = 1;

    MatrixXd scenarios_host(5, 1);
    scenarios_host << 0.10, -0.05, 0.03, -0.02, 0.07;

    VectorXd weights(1);
    weights << 1.0;

    RiskConfig cfg;
    cfg.confidence_level = 0.80;

    auto [risk, component] = compute_portfolio_risk_decomp_cpu(
        scenarios_host, weights, cfg);

    ASSERT_EQ(component.size(), 1);
    EXPECT_NEAR(component(0), risk.cvar, 1e-10)
        << "Single asset: component CVaR (" << component(0)
        << ") should equal total CVaR (" << risk.cvar << ")";
}
