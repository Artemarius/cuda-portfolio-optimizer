#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "constraints/constraint_set.h"
#include "optimizer/admm_kernels.h"
#include "optimizer/admm_solver.h"
#include "optimizer/efficient_frontier.h"
#include "optimizer/objective.h"
#include "simulation/cholesky_utils.h"
#include "simulation/monte_carlo.h"
#include "simulation/scenario_matrix.h"

using namespace cpo;

// ── Objective evaluation tests ──────────────────────────────────────

TEST(Objective, BasicEvaluation) {
    // 4 scenarios, 2 assets.
    MatrixXd scenarios(4, 2);
    scenarios << 0.10, 0.05,
                -0.05, 0.02,
                 0.03, -0.01,
                -0.02, -0.03;

    VectorXd w(2);
    w << 0.6, 0.4;

    // Losses = -scenarios * w = [-0.08, 0.022, -0.014, 0.024]
    // alpha = 0.05 (tail probability, 95% confidence)
    // zeta = 0 (arbitrary for this test)
    //
    // max(0, loss_i - 0):
    //   i=0: max(0, -0.08) = 0
    //   i=1: max(0, 0.022) = 0.022
    //   i=2: max(0, -0.014) = 0
    //   i=3: max(0, 0.024) = 0.024
    // sum_excess = 0.046
    // F = 0 + (1/(4*0.05)) * 0.046 = 0.046 / 0.2 = 0.23

    double alpha = 0.05;
    auto result = evaluate_objective_cpu(scenarios, w, 0.0, alpha);

    EXPECT_NEAR(result.value, 0.23, 1e-10);
    EXPECT_EQ(result.grad_w.size(), 2);
}

TEST(Objective, OptimalZeta) {
    // For equal-weight portfolio, find VaR.
    MatrixXd scenarios(4, 2);
    scenarios << 0.10, 0.05,
                -0.05, 0.02,
                 0.03, -0.01,
                -0.02, -0.03;

    VectorXd w(2);
    w << 0.6, 0.4;

    // Losses sorted: [-0.08, -0.014, 0.022, 0.024]
    // alpha = 0.05, confidence = 0.95
    // VaR index = floor(0.95 * 4) = 3, VaR = sorted[3] = 0.024
    double zeta = find_optimal_zeta(scenarios, w, 0.05);
    EXPECT_NEAR(zeta, 0.024, 1e-10);
}

TEST(Objective, InvalidAlpha) {
    MatrixXd scenarios(2, 1);
    scenarios << 0.1, -0.1;
    VectorXd w(1);
    w << 1.0;

    EXPECT_THROW(evaluate_objective_cpu(scenarios, w, 0.0, 0.0),
                 std::runtime_error);
    EXPECT_THROW(evaluate_objective_cpu(scenarios, w, 0.0, 1.0),
                 std::runtime_error);
}

// ── Single asset ADMM test ──────────────────────────────────────────

TEST(AdmmSolver, SingleAsset) {
    // Single asset: optimal w = [1.0], CVaR = scenario CVaR.
    const int n_scenarios = 1000;

    // Generate simple scenarios: uniform [-0.1, 0.2]
    MatrixXd scenarios(n_scenarios, 1);
    for (int i = 0; i < n_scenarios; ++i) {
        scenarios(i, 0) = -0.1 + 0.3 * i / (n_scenarios - 1.0);
    }

    VectorXd mu(1);
    mu << scenarios.col(0).mean();

    AdmmConfig config;
    config.confidence_level = 0.95;
    config.max_iter = 200;
    config.verbose = false;

    auto result = admm_solve(scenarios, mu, config);

    // Weight must be 1.0 (only one asset).
    EXPECT_NEAR(result.weights(0), 1.0, 1e-3);
    EXPECT_NEAR(result.weights.sum(), 1.0, 1e-6);
}

// ── 2-asset ADMM test ───────────────────────────────────────────────

TEST(AdmmSolver, TwoAssetConstraintSatisfaction) {
    // Verify basic constraint satisfaction: weights sum to 1, non-negative.
    const int n_scenarios = 10000;
    const int n_assets = 2;

    VectorXd mu(2);
    mu << 0.05, 0.10;

    MatrixXd cov(2, 2);
    cov << 0.04, 0.01,
           0.01, 0.09;
    auto chol = compute_cholesky(cov);

    MonteCarloConfig mc_cfg;
    mc_cfg.n_scenarios = n_scenarios;
    mc_cfg.seed = 42;
    MatrixXd scenarios = generate_scenarios_cpu(mu, chol, mc_cfg);

    AdmmConfig config;
    config.confidence_level = 0.95;
    config.max_iter = 300;

    auto result = admm_solve(scenarios, mu, config);

    // Weights must sum to 1.
    EXPECT_NEAR(result.weights.sum(), 1.0, 1e-4);
    // Weights must be non-negative.
    for (int i = 0; i < n_assets; ++i) {
        EXPECT_GE(result.weights(i), -1e-4)
            << "w(" << i << ") = " << result.weights(i);
    }
    // CVaR must be positive (losses in the tail).
    EXPECT_GT(result.cvar, 0.0);
}

// ── Box constraints test ────────────────────────────────────────────

TEST(AdmmSolver, BoxConstraints) {
    // 3 assets with max 50% per position.
    const int n_scenarios = 10000;
    const int n_assets = 3;

    VectorXd mu(3);
    mu << 0.03, 0.06, 0.12;

    MatrixXd cov = MatrixXd::Identity(3, 3);
    cov *= 0.04;
    cov(0, 1) = cov(1, 0) = 0.01;
    cov(1, 2) = cov(2, 1) = 0.02;
    auto chol = compute_cholesky(cov);

    MonteCarloConfig mc_cfg;
    mc_cfg.n_scenarios = n_scenarios;
    mc_cfg.seed = 123;
    MatrixXd scenarios = generate_scenarios_cpu(mu, chol, mc_cfg);

    AdmmConfig config;
    config.confidence_level = 0.95;
    config.max_iter = 300;
    config.constraints.has_position_limits = true;
    config.constraints.position_limits.w_min = VectorXd::Zero(3);
    config.constraints.position_limits.w_max = VectorXd::Constant(3, 0.5);

    auto result = admm_solve(scenarios, mu, config);

    EXPECT_NEAR(result.weights.sum(), 1.0, 1e-3);
    for (int i = 0; i < n_assets; ++i) {
        EXPECT_GE(result.weights(i), -1e-4)
            << "w(" << i << ") below lower bound";
        EXPECT_LE(result.weights(i), 0.5 + 1e-3)
            << "w(" << i << ") above upper bound";
    }
}

// ── Equal expected returns test ─────────────────────────────────────

TEST(AdmmSolver, EqualExpectedReturns) {
    // When all assets have same expected return, the optimizer should
    // find the minimum-risk portfolio (minimum CVaR).
    const int n_scenarios = 10000;
    const int n_assets = 3;

    VectorXd mu = VectorXd::Constant(3, 0.05);

    // Different volatilities but same mean.
    MatrixXd cov(3, 3);
    cov << 0.04, 0.005, 0.005,
           0.005, 0.09, 0.01,
           0.005, 0.01, 0.16;
    auto chol = compute_cholesky(cov);

    MonteCarloConfig mc_cfg;
    mc_cfg.n_scenarios = n_scenarios;
    mc_cfg.seed = 42;
    MatrixXd scenarios = generate_scenarios_cpu(mu, chol, mc_cfg);

    AdmmConfig config;
    config.confidence_level = 0.95;
    config.max_iter = 300;

    auto result = admm_solve(scenarios, mu, config);

    EXPECT_NEAR(result.weights.sum(), 1.0, 1e-4);

    // With equal means, lower-variance asset should get more weight.
    // Asset 0 has lowest variance (0.04) -> should get highest weight.
    EXPECT_GT(result.weights(0), result.weights(1));
    EXPECT_GT(result.weights(0), result.weights(2));
}

// ── GPU objective evaluation test ───────────────────────────────────

TEST(GpuObjective, MatchesCPU) {
    // Compare GPU and CPU objective evaluation.
    const int n_scenarios = 5000;
    const int n_assets = 3;

    VectorXd mu(3);
    mu << 0.03, 0.06, 0.09;

    MatrixXd cov = MatrixXd::Identity(3, 3) * 0.04;
    cov(0, 1) = cov(1, 0) = 0.01;
    auto chol = compute_cholesky(cov);

    MonteCarloConfig mc_cfg;
    mc_cfg.n_scenarios = n_scenarios;
    mc_cfg.seed = 42;
    MatrixXd cpu_scenarios = generate_scenarios_cpu(mu, chol, mc_cfg);

    // Upload to GPU.
    ScenarioMatrix gpu_scenarios(n_scenarios, n_assets);
    MatrixXs float_scenarios = cpu_scenarios.cast<float>();
    gpu_scenarios.from_host(float_scenarios);

    VectorXd w_d(3);
    w_d << 0.3, 0.4, 0.3;
    VectorXs w_f = w_d.cast<float>();
    double zeta = 0.1;
    double alpha = 0.05;

    // CPU evaluation.
    auto cpu_result = evaluate_objective_cpu(cpu_scenarios, w_d, zeta, alpha);

    // GPU evaluation.
    auto gpu_result = evaluate_objective_gpu(gpu_scenarios, w_f,
                                             static_cast<float>(zeta));

    // Assemble GPU objective value.
    double inv_n_alpha = 1.0 / (n_scenarios * alpha);
    double gpu_obj_value = zeta + inv_n_alpha * gpu_result.value;
    VectorXd gpu_grad = inv_n_alpha * gpu_result.grad_w;

    // Compare (float precision tolerance).
    EXPECT_NEAR(gpu_obj_value, cpu_result.value, 0.01)
        << "GPU=" << gpu_obj_value << " CPU=" << cpu_result.value;

    for (int j = 0; j < n_assets; ++j) {
        EXPECT_NEAR(gpu_grad(j), cpu_result.grad_w(j), 0.01)
            << "grad_w[" << j << "] GPU=" << gpu_grad(j)
            << " CPU=" << cpu_result.grad_w(j);
    }
}

// ── Efficient frontier tests ────────────────────────────────────────

TEST(EfficientFrontier, BasicFrontier) {
    // 2 assets, verify frontier is computed and has expected properties.
    const int n_scenarios = 10000;
    const int n_assets = 2;

    VectorXd mu(2);
    mu << 0.03, 0.10;

    MatrixXd cov(2, 2);
    cov << 0.04, 0.01,
           0.01, 0.09;
    auto chol = compute_cholesky(cov);

    MonteCarloConfig mc_cfg;
    mc_cfg.n_scenarios = n_scenarios;
    mc_cfg.seed = 42;
    MatrixXd scenarios = generate_scenarios_cpu(mu, chol, mc_cfg);

    FrontierConfig f_cfg;
    f_cfg.n_points = 5;
    f_cfg.admm_config.confidence_level = 0.95;
    f_cfg.admm_config.max_iter = 200;

    auto frontier = compute_efficient_frontier(scenarios, mu, f_cfg);

    ASSERT_EQ(frontier.size(), 5u);

    // All points should have valid weights.
    for (const auto& point : frontier) {
        EXPECT_NEAR(point.weights.sum(), 1.0, 1e-3)
            << "Weights don't sum to 1 at target=" << point.target_return;
        for (int i = 0; i < n_assets; ++i) {
            EXPECT_GE(point.weights(i), -1e-3)
                << "Negative weight at target=" << point.target_return;
        }
    }
}

TEST(EfficientFrontier, MonotonicRisk) {
    // Higher target return should generally lead to higher CVaR.
    // (Not strictly monotonic due to noise, but the trend should hold.)
    const int n_scenarios = 20000;

    VectorXd mu(3);
    mu << 0.02, 0.06, 0.12;

    MatrixXd cov(3, 3);
    cov << 0.01, 0.002, 0.003,
           0.002, 0.04, 0.01,
           0.003, 0.01, 0.16;
    auto chol = compute_cholesky(cov);

    MonteCarloConfig mc_cfg;
    mc_cfg.n_scenarios = n_scenarios;
    mc_cfg.seed = 42;
    MatrixXd scenarios = generate_scenarios_cpu(mu, chol, mc_cfg);

    FrontierConfig f_cfg;
    f_cfg.n_points = 5;
    f_cfg.admm_config.confidence_level = 0.95;
    f_cfg.admm_config.max_iter = 300;

    auto frontier = compute_efficient_frontier(scenarios, mu, f_cfg);

    // Check that the last point has higher CVaR than the first.
    // This is the overall trend of the efficient frontier.
    EXPECT_GT(frontier.back().cvar, frontier.front().cvar)
        << "Frontier should show increasing risk with return. "
        << "First CVaR=" << frontier.front().cvar
        << " Last CVaR=" << frontier.back().cvar;
}

// ── Turnover constraint test ────────────────────────────────────────

TEST(AdmmSolver, TurnoverConstraint) {
    // 3 assets with tight turnover limit from equal-weight.
    const int n_scenarios = 10000;
    const int n_assets = 3;

    VectorXd mu(3);
    mu << 0.03, 0.06, 0.12;

    MatrixXd cov = MatrixXd::Identity(3, 3) * 0.04;
    cov(0, 1) = cov(1, 0) = 0.01;
    cov(1, 2) = cov(2, 1) = 0.02;
    auto chol = compute_cholesky(cov);

    MonteCarloConfig mc_cfg;
    mc_cfg.n_scenarios = n_scenarios;
    mc_cfg.seed = 123;
    MatrixXd scenarios = generate_scenarios_cpu(mu, chol, mc_cfg);

    VectorXd w_prev = VectorXd::Constant(3, 1.0 / 3.0);

    AdmmConfig config;
    config.confidence_level = 0.95;
    config.max_iter = 300;
    config.constraints.has_turnover = true;
    config.constraints.turnover.w_prev = w_prev;
    config.constraints.turnover.tau = 0.3;

    auto result = admm_solve(scenarios, mu, config);

    EXPECT_NEAR(result.weights.sum(), 1.0, 1e-3);
    for (int i = 0; i < n_assets; ++i) {
        EXPECT_GE(result.weights(i), -1e-4)
            << "w(" << i << ") below 0";
    }

    double turnover = (result.weights - w_prev).lpNorm<1>();
    EXPECT_LE(turnover, 0.3 + 1e-2)
        << "Turnover " << turnover << " exceeds limit 0.3";
}

// ── Sector constraint test ──────────────────────────────────────────

TEST(AdmmSolver, SectorConstraints) {
    // 4 assets, sector {0,1} capped at 40%.
    const int n_scenarios = 10000;
    const int n_assets = 4;

    VectorXd mu(4);
    mu << 0.03, 0.06, 0.09, 0.12;

    MatrixXd cov = MatrixXd::Identity(4, 4) * 0.04;
    cov(0, 1) = cov(1, 0) = 0.01;
    cov(2, 3) = cov(3, 2) = 0.02;
    auto chol = compute_cholesky(cov);

    MonteCarloConfig mc_cfg;
    mc_cfg.n_scenarios = n_scenarios;
    mc_cfg.seed = 456;
    MatrixXd scenarios = generate_scenarios_cpu(mu, chol, mc_cfg);

    AdmmConfig config;
    config.confidence_level = 0.95;
    config.max_iter = 300;
    config.constraints.has_sector_constraints = true;
    SectorBound sb;
    sb.name = "SectorA";
    sb.assets = {0, 1};
    sb.max_exposure = 0.4;
    config.constraints.sector_constraints.sectors.push_back(sb);

    auto result = admm_solve(scenarios, mu, config);

    EXPECT_NEAR(result.weights.sum(), 1.0, 1e-3);
    for (int i = 0; i < n_assets; ++i) {
        EXPECT_GE(result.weights(i), -1e-4)
            << "w(" << i << ") below 0";
    }

    double sector_sum = result.weights(0) + result.weights(1);
    EXPECT_LE(sector_sum, 0.4 + 1e-2)
        << "Sector sum " << sector_sum << " exceeds limit 0.4";
}
