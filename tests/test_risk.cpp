#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "risk/cvar.h"
#include "risk/device_vector.h"
#include "risk/portfolio_loss.h"
#include "risk/risk_result.h"
#include "simulation/cholesky_utils.h"
#include "simulation/monte_carlo.h"
#include "simulation/scenario_matrix.h"

using namespace cpo;

// ── DeviceVector tests ──────────────────────────────────────────────

TEST(DeviceVector, AllocateAndRoundtrip) {
    const Index n = 1000;
    DeviceVector<Scalar> dv(n);
    EXPECT_EQ(dv.size(), n);
    EXPECT_EQ(dv.bytes(), static_cast<size_t>(n) * sizeof(Scalar));
    EXPECT_NE(dv.device_ptr(), nullptr);

    // Upload known data, download, verify roundtrip.
    std::vector<Scalar> host_data(n);
    for (Index i = 0; i < n; ++i) {
        host_data[i] = static_cast<Scalar>(i) * 0.01f;
    }
    dv.from_host(host_data);

    auto roundtrip = dv.to_host();
    ASSERT_EQ(roundtrip.size(), static_cast<size_t>(n));
    for (Index i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(roundtrip[i], host_data[i]) << "index " << i;
    }
}

TEST(DeviceVector, MoveSemantics) {
    DeviceVector<Scalar> a(100);
    Scalar* ptr = a.device_ptr();
    EXPECT_NE(ptr, nullptr);

    // Move constructor.
    DeviceVector<Scalar> b(std::move(a));
    EXPECT_EQ(b.device_ptr(), ptr);
    EXPECT_EQ(b.size(), 100);
    EXPECT_EQ(a.device_ptr(), nullptr);
    EXPECT_EQ(a.size(), 0);

    // Move assignment.
    DeviceVector<Scalar> c(50);
    c = std::move(b);
    EXPECT_EQ(c.device_ptr(), ptr);
    EXPECT_EQ(c.size(), 100);
    EXPECT_EQ(b.device_ptr(), nullptr);
    EXPECT_EQ(b.size(), 0);
}

TEST(DeviceVector, FromHostSizeMismatch) {
    DeviceVector<Scalar> dv(100);
    std::vector<Scalar> wrong_size(50, 0.0f);
    EXPECT_THROW(dv.from_host(wrong_size), std::runtime_error);
}

// ── Deterministic portfolio loss tests ──────────────────────────────

class PortfolioLossTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 2 assets, 4 scenarios (column-major).
        // Scenario returns:
        //   s0: r = [0.10, 0.05]
        //   s1: r = [-0.05, 0.02]
        //   s2: r = [0.03, -0.01]
        //   s3: r = [-0.02, -0.03]
        n_scenarios_ = 4;
        n_assets_ = 2;

        // Column-major layout: col0 = [0.10, -0.05, 0.03, -0.02],
        //                      col1 = [0.05, 0.02, -0.01, -0.03]
        std::vector<Scalar> flat = {
            0.10f, -0.05f, 0.03f, -0.02f,   // asset 0
            0.05f,  0.02f, -0.01f, -0.03f    // asset 1
        };
        scenarios_ = std::make_unique<ScenarioMatrix>(n_scenarios_, n_assets_);
        scenarios_->from_host(flat);

        // Equal weights.
        weights_.resize(n_assets_);
        weights_ << 0.6f, 0.4f;

        // Expected portfolio returns: w'r for each scenario.
        // s0: 0.6*0.10 + 0.4*0.05 = 0.08
        // s1: 0.6*(-0.05) + 0.4*0.02 = -0.022
        // s2: 0.6*0.03 + 0.4*(-0.01) = 0.014
        // s3: 0.6*(-0.02) + 0.4*(-0.03) = -0.024
        // Expected losses = -returns: [-0.08, 0.022, -0.014, 0.024]
    }

    Index n_scenarios_;
    Index n_assets_;
    std::unique_ptr<ScenarioMatrix> scenarios_;
    VectorXs weights_;
};

TEST_F(PortfolioLossTest, GPUExactValues) {
    auto d_losses = compute_portfolio_loss_gpu(*scenarios_, weights_);
    auto losses = d_losses.to_host();

    ASSERT_EQ(losses.size(), 4u);
    EXPECT_NEAR(losses[0], -0.08f,  1e-5f);
    EXPECT_NEAR(losses[1],  0.022f, 1e-5f);
    EXPECT_NEAR(losses[2], -0.014f, 1e-5f);
    EXPECT_NEAR(losses[3],  0.024f, 1e-5f);
}

TEST_F(PortfolioLossTest, CPUExactValues) {
    // Build CPU-precision inputs.
    MatrixXd scenarios_host(4, 2);
    scenarios_host << 0.10, 0.05,
                     -0.05, 0.02,
                      0.03, -0.01,
                     -0.02, -0.03;

    VectorXd weights_d(2);
    weights_d << 0.6, 0.4;

    VectorXd losses = compute_portfolio_loss_cpu(scenarios_host, weights_d);

    ASSERT_EQ(losses.size(), 4);
    EXPECT_NEAR(losses(0), -0.08,  1e-12);
    EXPECT_NEAR(losses(1),  0.022, 1e-12);
    EXPECT_NEAR(losses(2), -0.014, 1e-12);
    EXPECT_NEAR(losses(3),  0.024, 1e-12);
}

TEST_F(PortfolioLossTest, WeightsSizeMismatch) {
    VectorXs bad_weights(3);
    bad_weights << 0.3f, 0.3f, 0.4f;
    EXPECT_THROW(compute_portfolio_loss_gpu(*scenarios_, bad_weights),
                 std::runtime_error);
}

// ── Deterministic CVaR tests ────────────────────────────────────────

TEST(CVaRDeterministic, KnownLossVector) {
    // 10-element loss vector: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    // Sorted ascending: same.
    // alpha = 0.90: var_index = floor(0.9 * 10) = 9
    //   VaR = sorted[9] = 10
    //   CVaR = mean(sorted[9..9]) = 10
    // alpha = 0.80: var_index = floor(0.8 * 10) = 8
    //   VaR = sorted[8] = 9
    //   CVaR = mean(sorted[8..9]) = (9+10)/2 = 9.5
    // alpha = 0.50: var_index = floor(0.5 * 10) = 5
    //   VaR = sorted[5] = 6
    //   CVaR = mean(sorted[5..9]) = (6+7+8+9+10)/5 = 8.0

    std::vector<Scalar> losses_host = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    DeviceVector<Scalar> d_losses(10);
    d_losses.from_host(losses_host);

    // alpha = 0.90
    {
        RiskConfig cfg;
        cfg.confidence_level = 0.90;
        auto result = compute_risk_gpu(d_losses, cfg);
        EXPECT_NEAR(result.var, 10.0, 1e-4);
        EXPECT_NEAR(result.cvar, 10.0, 1e-4);
    }

    // alpha = 0.80
    {
        RiskConfig cfg;
        cfg.confidence_level = 0.80;
        auto result = compute_risk_gpu(d_losses, cfg);
        EXPECT_NEAR(result.var, 9.0, 1e-4);
        EXPECT_NEAR(result.cvar, 9.5, 1e-4);
    }

    // alpha = 0.50
    {
        RiskConfig cfg;
        cfg.confidence_level = 0.50;
        auto result = compute_risk_gpu(d_losses, cfg);
        EXPECT_NEAR(result.var, 6.0, 1e-4);
        EXPECT_NEAR(result.cvar, 8.0, 1e-4);
    }
}

TEST(CVaRDeterministic, KnownLossVectorCPU) {
    VectorXd losses(10);
    losses << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10;

    // alpha = 0.80: VaR = 9, CVaR = 9.5
    RiskConfig cfg;
    cfg.confidence_level = 0.80;
    auto result = compute_risk_cpu(losses, cfg);
    EXPECT_NEAR(result.var, 9.0, 1e-12);
    EXPECT_NEAR(result.cvar, 9.5, 1e-12);
}

TEST(CVaRDeterministic, UnsortedInput) {
    // Verify that unsorted input works correctly.
    std::vector<Scalar> losses_host = {7, 2, 10, 4, 1, 8, 3, 6, 9, 5};
    DeviceVector<Scalar> d_losses(10);
    d_losses.from_host(losses_host);

    RiskConfig cfg;
    cfg.confidence_level = 0.80;
    auto result = compute_risk_gpu(d_losses, cfg);
    EXPECT_NEAR(result.var, 9.0, 1e-4);
    EXPECT_NEAR(result.cvar, 9.5, 1e-4);
}

TEST(CVaRDeterministic, InputNotModified) {
    // Verify the original loss vector is not modified by compute_risk_gpu.
    std::vector<Scalar> losses_host = {5, 3, 1, 4, 2};
    DeviceVector<Scalar> d_losses(5);
    d_losses.from_host(losses_host);

    RiskConfig cfg;
    cfg.confidence_level = 0.80;
    compute_risk_gpu(d_losses, cfg);

    auto after = d_losses.to_host();
    for (size_t i = 0; i < losses_host.size(); ++i) {
        EXPECT_FLOAT_EQ(after[i], losses_host[i])
            << "Loss vector was modified at index " << i;
    }
}

// ── CVaR properties tests ───────────────────────────────────────────

TEST(CVaRProperties, CVaRGreaterEqualVaR) {
    // CVaR >= VaR is a defining property (CVaR averages the tail beyond VaR).
    std::vector<Scalar> losses_host(1000);
    for (int i = 0; i < 1000; ++i) {
        losses_host[i] = static_cast<Scalar>(i) * 0.01f;
    }
    DeviceVector<Scalar> d_losses(1000);
    d_losses.from_host(losses_host);

    for (double alpha : {0.50, 0.80, 0.90, 0.95, 0.99}) {
        RiskConfig cfg;
        cfg.confidence_level = alpha;
        auto result = compute_risk_gpu(d_losses, cfg);
        EXPECT_GE(result.cvar, result.var)
            << "CVaR < VaR at alpha=" << alpha
            << " CVaR=" << result.cvar << " VaR=" << result.var;
    }
}

TEST(CVaRProperties, MonotonicityInAlpha) {
    // Higher alpha -> higher VaR and CVaR (deeper into the tail).
    std::vector<Scalar> losses_host(10000);
    for (int i = 0; i < 10000; ++i) {
        losses_host[i] = static_cast<Scalar>(i) * 0.001f;
    }
    DeviceVector<Scalar> d_losses(10000);
    d_losses.from_host(losses_host);

    double prev_var = -1e30;
    double prev_cvar = -1e30;
    for (double alpha : {0.50, 0.80, 0.90, 0.95, 0.99}) {
        RiskConfig cfg;
        cfg.confidence_level = alpha;
        auto result = compute_risk_gpu(d_losses, cfg);
        EXPECT_GE(result.var, prev_var)
            << "VaR not monotonic at alpha=" << alpha;
        EXPECT_GE(result.cvar, prev_cvar)
            << "CVaR not monotonic at alpha=" << alpha;
        prev_var = result.var;
        prev_cvar = result.cvar;
    }
}

// ── Statistical CVaR test (normal distribution) ─────────────────────

TEST(CVaRStatistical, NormalDistributionGPU) {
    // Generate N(mu, sigma^2) scenarios for a single-asset portfolio
    // with weight = 1, so portfolio return = scenario return.
    // Loss = -return ~ N(-mu, sigma^2).
    //
    // Analytical normal CVaR formula:
    //   VaR_alpha = mu_L + sigma * Phi^{-1}(alpha)
    //   CVaR_alpha = mu_L + sigma * phi(Phi^{-1}(alpha)) / (1-alpha)
    // where mu_L = -mu (mean of loss), sigma = portfolio vol.
    //
    // Using mu = 0.05, sigma = 0.20, alpha = 0.95:
    //   Phi^{-1}(0.95) = 1.6449
    //   phi(1.6449) = 0.10314
    //   VaR_analytical = -0.05 + 0.20 * 1.6449 = 0.27898
    //   CVaR_analytical = -0.05 + 0.20 * 0.10314 / 0.05 = 0.36256

    const double mu = 0.05;
    const double sigma = 0.20;
    const Index n_assets = 1;
    const Index n_scenarios = 200000;

    VectorXd mu_vec(1);
    mu_vec << mu;
    MatrixXd cov(1, 1);
    cov << sigma * sigma;
    auto chol = compute_cholesky(cov);

    MonteCarloConfig mc_cfg;
    mc_cfg.n_scenarios = n_scenarios;
    mc_cfg.seed = 42;

    auto scenarios = generate_scenarios_gpu(mu_vec, chol, mc_cfg);

    VectorXs weights(1);
    weights << 1.0f;

    RiskConfig risk_cfg;
    risk_cfg.confidence_level = 0.95;

    auto result = compute_portfolio_risk_gpu(scenarios, weights, risk_cfg);

    // Analytical values.
    const double z_alpha = 1.6448536269514729;     // Phi^{-1}(0.95)
    const double phi_z = 0.10313564064;             // phi(1.6449)
    const double alpha = 0.95;
    double var_analytical = -mu + sigma * z_alpha;
    double cvar_analytical = -mu + sigma * phi_z / (1.0 - alpha);

    // 5% relative tolerance for float precision + Monte Carlo sampling noise.
    EXPECT_NEAR(result.var, var_analytical, 0.05 * std::abs(var_analytical))
        << "VaR: actual=" << result.var << " analytical=" << var_analytical;
    EXPECT_NEAR(result.cvar, cvar_analytical, 0.05 * std::abs(cvar_analytical))
        << "CVaR: actual=" << result.cvar << " analytical=" << cvar_analytical;

    // Expected return should be close to mu.
    EXPECT_NEAR(result.expected_return, mu, 0.01)
        << "E[r]: actual=" << result.expected_return << " expected=" << mu;

    // Volatility should be close to sigma.
    EXPECT_NEAR(result.volatility, sigma, 0.01)
        << "Vol: actual=" << result.volatility << " expected=" << sigma;
}

// ── GPU/CPU parity test ─────────────────────────────────────────────

TEST(CVaRParity, GPUvsCPU) {
    // Generate scenarios, compute risk on both paths, compare.
    const Index n_assets = 3;
    const Index n_scenarios = 50000;

    VectorXd mu(3);
    mu << 0.03, 0.06, 0.09;

    VectorXd sigmas(3);
    sigmas << 0.15, 0.20, 0.25;

    MatrixXd R = MatrixXd::Identity(3, 3);
    R(0, 1) = R(1, 0) = 0.5;
    R(0, 2) = R(2, 0) = 0.3;
    R(1, 2) = R(2, 1) = 0.4;

    MatrixXd D = sigmas.asDiagonal();
    MatrixXd cov = D * R * D;
    auto chol = compute_cholesky(cov);

    // Generate scenarios on CPU (double).
    MonteCarloConfig mc_cfg;
    mc_cfg.n_scenarios = n_scenarios;
    mc_cfg.seed = 7777;
    MatrixXd cpu_scenarios = generate_scenarios_cpu(mu, chol, mc_cfg);

    // Upload to GPU as float.
    ScenarioMatrix gpu_scenarios(n_scenarios, n_assets);
    MatrixXs float_scenarios = cpu_scenarios.cast<float>();
    gpu_scenarios.from_host(float_scenarios);

    // Weights.
    VectorXd weights_d(3);
    weights_d << 0.4, 0.35, 0.25;
    VectorXs weights_f = weights_d.cast<float>();

    RiskConfig cfg;
    cfg.confidence_level = 0.95;

    auto gpu_result = compute_portfolio_risk_gpu(gpu_scenarios, weights_f, cfg);
    auto cpu_result = compute_portfolio_risk_cpu(cpu_scenarios, weights_d, cfg);

    // Float precision tolerance: ~1e-4 relative for VaR/CVaR.
    double var_tol = std::max(1e-4, 1e-3 * std::abs(cpu_result.var));
    double cvar_tol = std::max(1e-4, 1e-3 * std::abs(cpu_result.cvar));

    EXPECT_NEAR(gpu_result.var, cpu_result.var, var_tol)
        << "VaR GPU=" << gpu_result.var << " CPU=" << cpu_result.var;
    EXPECT_NEAR(gpu_result.cvar, cpu_result.cvar, cvar_tol)
        << "CVaR GPU=" << gpu_result.cvar << " CPU=" << cpu_result.cvar;
    EXPECT_NEAR(gpu_result.expected_return, cpu_result.expected_return, 1e-3)
        << "E[r] GPU=" << gpu_result.expected_return
        << " CPU=" << cpu_result.expected_return;
    EXPECT_NEAR(gpu_result.volatility, cpu_result.volatility, 1e-3)
        << "Vol GPU=" << gpu_result.volatility
        << " CPU=" << cpu_result.volatility;
}

// ── Statistics tests ────────────────────────────────────────────────

TEST(Statistics, ExpectedReturnAndVolatility) {
    // 5 deterministic losses: [-0.1, -0.05, 0, 0.05, 0.1]
    // mean_loss = 0, expected_return = 0
    // var(loss) = mean(loss^2) - mean(loss)^2 = (0.01+0.0025+0+0.0025+0.01)/5 = 0.005
    // vol = sqrt(0.005) = 0.07071
    std::vector<Scalar> losses_host = {-0.1f, -0.05f, 0.0f, 0.05f, 0.1f};
    DeviceVector<Scalar> d_losses(5);
    d_losses.from_host(losses_host);

    RiskConfig cfg;
    cfg.confidence_level = 0.50;
    auto result = compute_risk_gpu(d_losses, cfg);

    EXPECT_NEAR(result.expected_return, 0.0, 1e-5);
    EXPECT_NEAR(result.volatility, std::sqrt(0.005), 1e-4);
}

TEST(Statistics, SharpeAndSortino) {
    // Losses: all negative (all positive returns).
    // losses = [-0.10, -0.08, -0.06, -0.04, -0.02]
    // mean_loss = -0.06, expected_return = 0.06
    // loss^2 = [0.01, 0.0064, 0.0036, 0.0016, 0.0004]
    // var(loss) = mean(loss^2) - mean(loss)^2 = 0.0044 - 0.0036 = 0.0008
    // vol = sqrt(0.0008)
    // Sharpe = 0.06 / sqrt(0.0008)
    // downside: max(loss, 0) = all zeros (no negative returns)
    // downside_dev = 0, sortino = 0 (since we'd divide by zero, returns 0)
    std::vector<Scalar> losses_host = {-0.10f, -0.08f, -0.06f, -0.04f, -0.02f};
    DeviceVector<Scalar> d_losses(5);
    d_losses.from_host(losses_host);

    RiskConfig cfg;
    cfg.confidence_level = 0.50;
    auto result = compute_risk_gpu(d_losses, cfg);

    double expected_return = 0.06;
    double vol = std::sqrt(0.0008);
    EXPECT_NEAR(result.expected_return, expected_return, 1e-5);
    EXPECT_NEAR(result.volatility, vol, 1e-4);
    EXPECT_NEAR(result.sharpe_ratio, expected_return / vol, 1e-3);
    // No downside (all returns positive), so sortino should be 0.
    EXPECT_NEAR(result.sortino_ratio, 0.0, 1e-5);
}

// ── Edge cases ──────────────────────────────────────────────────────

TEST(CVaREdgeCases, SingleElement) {
    std::vector<Scalar> losses_host = {5.0f};
    DeviceVector<Scalar> d_losses(1);
    d_losses.from_host(losses_host);

    RiskConfig cfg;
    cfg.confidence_level = 0.50;
    auto result = compute_risk_gpu(d_losses, cfg);

    // With 1 element, VaR = CVaR = 5.0 regardless of alpha
    // (var_index = floor(0.5 * 1) = 0, tail = [5.0])
    EXPECT_NEAR(result.var, 5.0, 1e-4);
    EXPECT_NEAR(result.cvar, 5.0, 1e-4);
}

TEST(CVaREdgeCases, InvalidAlpha) {
    DeviceVector<Scalar> d_losses(10);
    std::vector<Scalar> dummy(10, 1.0f);
    d_losses.from_host(dummy);

    RiskConfig cfg;
    cfg.confidence_level = 0.0;
    EXPECT_THROW(compute_risk_gpu(d_losses, cfg), std::runtime_error);

    cfg.confidence_level = 1.0;
    EXPECT_THROW(compute_risk_gpu(d_losses, cfg), std::runtime_error);
}

// ── Full pipeline (convenience function) test ───────────────────────

TEST(FullPipeline, ConvenienceGPU) {
    // Single asset, known scenarios, verify the pipeline end-to-end.
    const Index n_scenarios = 5;
    const Index n_assets = 1;

    // Scenarios (returns): [0.10, -0.05, 0.03, -0.02, 0.07]
    // Column-major with 1 asset = same as flat.
    std::vector<Scalar> flat = {0.10f, -0.05f, 0.03f, -0.02f, 0.07f};
    ScenarioMatrix scenarios(n_scenarios, n_assets);
    scenarios.from_host(flat);

    VectorXs weights(1);
    weights << 1.0f;

    RiskConfig cfg;
    cfg.confidence_level = 0.80;

    auto result = compute_portfolio_risk_gpu(scenarios, weights, cfg);

    // Losses = [-0.10, 0.05, -0.03, 0.02, -0.07]
    // Sorted: [-0.10, -0.07, -0.03, 0.02, 0.05]
    // var_index = floor(0.8 * 5) = 4, VaR = sorted[4] = 0.05
    // CVaR = mean(sorted[4..4]) = 0.05
    EXPECT_NEAR(result.var, 0.05, 1e-4);
    EXPECT_NEAR(result.cvar, 0.05, 1e-4);
    EXPECT_EQ(result.n_scenarios, 5);
    EXPECT_NEAR(result.confidence_level, 0.80, 1e-12);
}
