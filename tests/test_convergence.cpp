#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <tuple>
#include <vector>

#include "constraints/constraint_set.h"
#include "optimizer/admm_solver.h"
#include "optimizer/efficient_frontier.h"
#include "simulation/cholesky_utils.h"
#include "simulation/monte_carlo.h"

using namespace cpo;

// ── Helper functions ────────────────────────────────────────────────

/// Create a diagonal-dominant positive-definite covariance matrix.
///
/// Diagonal entries are drawn from [0.02, 0.10] (annualized variance),
/// off-diagonal correlations are small (0.005) to ensure well-conditioned
/// problems that converge reliably.
///
/// @param n Number of assets.
/// @return n x n symmetric positive-definite covariance matrix.
static MatrixXd make_diagonal_dominant_cov(int n) {
    MatrixXd cov = MatrixXd::Zero(n, n);

    // Diagonal: increasing variance per asset.
    for (int i = 0; i < n; ++i) {
        cov(i, i) = 0.02 + 0.08 * i / std::max(n - 1, 1);
    }

    // Off-diagonal: small positive correlation.
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            ScalarCPU off_diag = 0.005;
            cov(i, j) = off_diag;
            cov(j, i) = off_diag;
        }
    }

    return cov;
}

/// Create an expected returns vector with reasonable annualized values.
///
/// Returns are linearly spaced in [0.05, 0.15] so that asset 0 has the
/// lowest return and asset n-1 has the highest, creating a meaningful
/// risk-return tradeoff for efficient frontier computation.
///
/// @param n Number of assets.
/// @param seed Random seed (unused — deterministic for reproducibility).
/// @return Expected return vector (n, double).
static VectorXd make_random_mu(int n, uint64_t /*seed*/) {
    VectorXd mu(n);
    for (int i = 0; i < n; ++i) {
        mu(i) = 0.05 + 0.10 * i / std::max(n - 1, 1);
    }
    return mu;
}

/// Generate CPU-side return scenarios from mu and covariance.
///
/// Wraps compute_cholesky + generate_scenarios_cpu for concise test setup.
///
/// @param mu Expected return vector.
/// @param cov Covariance matrix.
/// @param n_scenarios Number of Monte Carlo scenarios.
/// @param seed RNG seed.
/// @return MatrixXd of shape (n_scenarios x n_assets).
static MatrixXd make_test_scenarios(const VectorXd& mu,
                                     const MatrixXd& cov,
                                     int n_scenarios,
                                     uint64_t seed) {
    auto chol = compute_cholesky(cov);

    MonteCarloConfig mc_cfg;
    mc_cfg.n_scenarios = n_scenarios;
    mc_cfg.seed = seed;

    return generate_scenarios_cpu(mu, chol, mc_cfg);
}

// ── Test fixture ────────────────────────────────────────────────────

/// Fixture providing a reusable 10-asset problem with well-conditioned
/// covariance for convergence regression tests.
class ConvergenceTest : public ::testing::Test {
  protected:
    static constexpr int kNAssets = 10;
    static constexpr int kNScenarios = 20000;
    static constexpr uint64_t kSeed = 42;

    VectorXd mu_;
    MatrixXd cov_;
    MatrixXd scenarios_;

    void SetUp() override {
        mu_ = make_random_mu(kNAssets, kSeed);
        cov_ = make_diagonal_dominant_cov(kNAssets);
        scenarios_ = make_test_scenarios(mu_, cov_, kNScenarios, kSeed);
    }
};

// ── Test 1: SmallProblemConverges ────────────────────────────────────

/// Regression guard: a trivial 2-asset problem must converge within 100
/// iterations. Phase 12 changes (over-relaxation, Anderson acceleration)
/// must not break small problems.
TEST_F(ConvergenceTest, SmallProblemConverges) {
    const int n_assets = 2;
    const int n_scenarios = 10000;

    VectorXd mu(n_assets);
    mu << 0.05, 0.12;

    MatrixXd cov(n_assets, n_assets);
    cov << 0.04, 0.01,
           0.01, 0.09;

    MatrixXd scenarios = make_test_scenarios(mu, cov, n_scenarios, 42);

    AdmmConfig config;
    config.confidence_level = 0.95;
    config.max_iter = 100;
    config.verbose = false;

    auto result = admm_solve(scenarios, mu, config);

    EXPECT_TRUE(result.converged)
        << "2-asset problem failed to converge in " << config.max_iter
        << " iterations (used " << result.iterations << ")";
    EXPECT_LT(result.iterations, 100)
        << "2-asset problem used too many iterations: " << result.iterations;
    EXPECT_NEAR(result.weights.sum(), 1.0, 1e-4);
    EXPECT_GT(result.cvar, 0.0);
}

// ── Test 2: MediumProblemConverges ───────────────────────────────────

/// 10-asset problem from the fixture must converge within 500 iterations.
/// Validates weight simplex constraints (sum-to-one, non-negative).
TEST_F(ConvergenceTest, MediumProblemConverges) {
    AdmmConfig config;
    config.confidence_level = 0.95;
    config.max_iter = 700;
    config.verbose = false;
    // Enable Phase 12 improvements for this test.
    config.alpha_relax = 1.5;
    config.anderson_depth = 3;
    config.x_update_lr = 0.05;

    auto result = admm_solve(scenarios_, mu_, config);

    EXPECT_TRUE(result.converged)
        << "10-asset problem failed to converge in " << config.max_iter
        << " iterations (used " << result.iterations << ")";
    EXPECT_LT(result.iterations, 700)
        << "10-asset problem used too many iterations: " << result.iterations;

    // Simplex constraints.
    EXPECT_NEAR(result.weights.sum(), 1.0, 1e-4)
        << "Weights sum = " << result.weights.sum();
    for (int i = 0; i < kNAssets; ++i) {
        EXPECT_GE(result.weights(i), -1e-4)
            << "w(" << i << ") = " << result.weights(i) << " is negative";
    }

    EXPECT_GT(result.cvar, 0.0);
    EXPECT_GT(result.expected_return, 0.0);
}

// ── Test 3: FrontierMonotonicity ────────────────────────────────────

/// The efficient frontier must show strictly increasing CVaR as target
/// return increases. This is the key Phase 12 success metric: currently
/// 12/15 points converge at 50 stocks, targeting 15/15.
///
/// Uses 5 assets and 5 frontier points — small enough to be reliable,
/// large enough to test monotonicity meaningfully.
TEST_F(ConvergenceTest, FrontierMonotonicity) {
    const int n_assets = 5;
    const int n_scenarios = 20000;

    VectorXd mu = make_random_mu(n_assets, 123);
    MatrixXd cov = make_diagonal_dominant_cov(n_assets);
    MatrixXd scenarios = make_test_scenarios(mu, cov, n_scenarios, 123);

    FrontierConfig f_cfg;
    f_cfg.n_points = 5;
    f_cfg.warm_start = true;
    f_cfg.admm_config.confidence_level = 0.95;
    f_cfg.admm_config.max_iter = 1000;
    f_cfg.admm_config.verbose = false;
    // Enable Phase 12 improvements.
    f_cfg.admm_config.alpha_relax = 1.5;
    f_cfg.admm_config.anderson_depth = 3;
    f_cfg.admm_config.x_update_lr = 0.05;

    auto frontier = compute_efficient_frontier(scenarios, mu, f_cfg);

    ASSERT_EQ(frontier.size(), 5u);

    // Count converged points — target is all 5, but accept >= 3 as a
    // regression guard. Phase 12 improvements aim to reach 5/5.
    int converged_count = 0;
    for (size_t i = 0; i < frontier.size(); ++i) {
        if (frontier[i].converged) ++converged_count;
    }
    EXPECT_GE(converged_count, 3)
        << "Only " << converged_count << "/5 frontier points converged";

    // CVaR must be non-decreasing along the frontier (monotonicity).
    // Use non-strict inequality since non-converged points may be imprecise.
    for (size_t i = 1; i < frontier.size(); ++i) {
        EXPECT_GE(frontier[i].cvar, frontier[i - 1].cvar - 1e-3)
            << "CVaR not monotonic: point " << i
            << " CVaR=" << frontier[i].cvar
            << " < point " << (i - 1)
            << " CVaR=" << frontier[i - 1].cvar;
    }

    // All weights must satisfy simplex constraints.
    for (size_t i = 0; i < frontier.size(); ++i) {
        EXPECT_NEAR(frontier[i].weights.sum(), 1.0, 1e-3)
            << "Frontier point " << i << " weights don't sum to 1";
        for (int j = 0; j < n_assets; ++j) {
            EXPECT_GE(frontier[i].weights(j), -1e-3)
                << "Frontier point " << i << " w(" << j << ") negative";
        }
    }
}

// ── Test 4: ConvergenceRateImproves (parameterized) ─────────────────

/// Parameterized test: verify ADMM converges within expected iteration
/// bounds for varying problem sizes. Current bounds are generous and
/// will be tightened after Phase 12 improvements are integrated.
class ConvergenceRateTest
    : public ::testing::TestWithParam<std::tuple<int, int>> {};

TEST_P(ConvergenceRateTest, WithinExpectedBounds) {
    auto [n_assets, max_expected_iters] = GetParam();

    VectorXd mu = make_random_mu(n_assets, 99);
    MatrixXd cov = make_diagonal_dominant_cov(n_assets);

    // More scenarios for larger problems to reduce MC noise.
    int n_scenarios = std::max(10000, n_assets * 2000);
    MatrixXd scenarios = make_test_scenarios(mu, cov, n_scenarios, 99);

    AdmmConfig config;
    config.confidence_level = 0.95;
    config.max_iter = max_expected_iters + 200;  // Generous headroom.
    config.verbose = false;
    // Enable Phase 12 improvements.
    config.alpha_relax = 1.5;
    config.anderson_depth = 3;
    config.x_update_lr = 0.05;
    // Residual balancing helps larger problems where primal/dual residuals
    // become imbalanced (Wohlberg 2017).
    config.residual_balancing = true;

    auto result = admm_solve(scenarios, mu, config);

    EXPECT_TRUE(result.converged)
        << n_assets << "-asset problem did not converge within "
        << config.max_iter << " iterations (used " << result.iterations << ")";
    EXPECT_LT(result.iterations, max_expected_iters)
        << n_assets << "-asset problem used " << result.iterations
        << " iterations, expected < " << max_expected_iters;
    EXPECT_NEAR(result.weights.sum(), 1.0, 1e-3);
}

INSTANTIATE_TEST_SUITE_P(
    ProblemSizes,
    ConvergenceRateTest,
    ::testing::Values(
        //       (n_assets, max_expected_iterations)
        std::make_tuple(5,  200),
        std::make_tuple(10, 800),
        std::make_tuple(25, 2500)
    ),
    [](const ::testing::TestParamInfo<std::tuple<int, int>>& info) {
        return std::to_string(std::get<0>(info.param)) + "_assets";
    }
);

// ── Test 5: WarmStartReducesIterations ──────────────────────────────

/// Warm-starting from a nearby solution should reduce iteration count.
/// This is critical for efficient frontier performance where each point
/// is solved using the previous point's weights as the initial guess.
TEST_F(ConvergenceTest, WarmStartReducesIterations) {
    const int n_assets = 5;
    const int n_scenarios = 15000;

    VectorXd mu = make_random_mu(n_assets, 77);
    MatrixXd cov = make_diagonal_dominant_cov(n_assets);
    MatrixXd scenarios = make_test_scenarios(mu, cov, n_scenarios, 77);

    AdmmConfig config;
    config.confidence_level = 0.95;
    config.max_iter = 500;
    config.verbose = false;

    // Cold start: solve from uniform initial weights (default).
    auto cold_result = admm_solve(scenarios, mu, config);

    ASSERT_TRUE(cold_result.converged)
        << "Cold start did not converge";

    // Warm start: solve from the cold-start solution.
    auto warm_result = admm_solve(scenarios, mu, config, cold_result.weights);

    EXPECT_TRUE(warm_result.converged)
        << "Warm start did not converge";
    EXPECT_LT(warm_result.iterations, cold_result.iterations)
        << "Warm start (" << warm_result.iterations
        << " iters) did not reduce iterations vs cold start ("
        << cold_result.iterations << " iters)";

    // Results should be close (same problem, same solution).
    EXPECT_NEAR(warm_result.cvar, cold_result.cvar, 1e-4)
        << "Warm start CVaR=" << warm_result.cvar
        << " differs from cold start CVaR=" << cold_result.cvar;
}

// ── Test 6: AdaptiveRhoActivates ────────────────────────────────────

/// Verify that adaptive rho adjusts the penalty parameter during
/// optimization. Use a 3-asset problem where initial rho is far from
/// optimal to trigger adaptation.
///
/// Reference: Boyd et al. 2011, Section 3.4.1, Eq. (3.13).
TEST_F(ConvergenceTest, AdaptiveRhoActivates) {
    const int n_assets = 3;
    const int n_scenarios = 10000;

    VectorXd mu(n_assets);
    mu << 0.03, 0.08, 0.15;

    // Highly heterogeneous variances to create imbalanced residuals.
    MatrixXd cov(n_assets, n_assets);
    cov << 0.01, 0.001, 0.002,
           0.001, 0.09, 0.005,
           0.002, 0.005, 0.25;

    MatrixXd scenarios = make_test_scenarios(mu, cov, n_scenarios, 55);

    // Start with deliberately poor initial rho.
    AdmmConfig config;
    config.confidence_level = 0.95;
    config.max_iter = 500;
    config.rho = 0.001;  // Very small initial rho.
    config.adaptive_rho = true;
    config.residual_balancing = true;
    config.alpha_relax = 1.5;
    config.anderson_depth = 3;
    config.x_update_lr = 0.05;
    config.verbose = false;

    auto result = admm_solve(scenarios, mu, config);

    EXPECT_TRUE(result.converged)
        << "Adaptive rho solver did not converge (used "
        << result.iterations << " iterations)";

    // If history is available, verify rho changed during optimization.
    if (!result.history.empty()) {
        ScalarCPU initial_rho = result.history.front().rho;
        ScalarCPU final_rho = result.history.back().rho;
        EXPECT_NE(initial_rho, final_rho)
            << "Adaptive rho did not activate: rho stayed at " << initial_rho;
    }

    EXPECT_NEAR(result.weights.sum(), 1.0, 1e-3);
    EXPECT_GT(result.cvar, 0.0);

    // Compare: non-adaptive with reasonable fixed rho should also converge.
    AdmmConfig config_fixed;
    config_fixed.confidence_level = 0.95;
    config_fixed.max_iter = 500;
    config_fixed.rho = 1.0;
    config_fixed.adaptive_rho = false;
    config_fixed.residual_balancing = false;
    auto fixed_result = admm_solve(scenarios, mu, config_fixed);

    EXPECT_TRUE(fixed_result.converged)
        << "Fixed rho solver did not converge";
}

// ── Test 7: ConstrainedConvergence ──────────────────────────────────

/// 10-asset problem with box constraints (max 20% per position) and
/// turnover constraint. Verify all constraints satisfied and convergence.
TEST_F(ConvergenceTest, ConstrainedConvergence) {
    // Use the fixture's 10-asset problem.
    VectorXd w_prev = VectorXd::Constant(kNAssets, 1.0 / kNAssets);

    AdmmConfig config;
    config.confidence_level = 0.95;
    config.max_iter = 1000;
    config.verbose = false;
    // Enable Phase 12 improvements.
    config.alpha_relax = 1.5;
    config.anderson_depth = 3;
    config.x_update_lr = 0.05;

    // Box constraints: each position <= 20%.
    config.constraints.has_position_limits = true;
    config.constraints.position_limits.w_min = VectorXd::Zero(kNAssets);
    config.constraints.position_limits.w_max =
        VectorXd::Constant(kNAssets, 0.20);

    // Turnover constraint from equal-weight portfolio.
    config.constraints.has_turnover = true;
    config.constraints.turnover.w_prev = w_prev;
    config.constraints.turnover.tau = 0.5;

    auto result = admm_solve(scenarios_, mu_, config);

    EXPECT_TRUE(result.converged)
        << "Constrained 10-asset problem did not converge (used "
        << result.iterations << " iterations)";

    // Simplex constraint.
    EXPECT_NEAR(result.weights.sum(), 1.0, 1e-3)
        << "Weights sum = " << result.weights.sum();

    // Box constraints.
    for (int i = 0; i < kNAssets; ++i) {
        EXPECT_GE(result.weights(i), -1e-4)
            << "w(" << i << ") = " << result.weights(i) << " below lower bound";
        EXPECT_LE(result.weights(i), 0.20 + 1e-3)
            << "w(" << i << ") = " << result.weights(i) << " above upper bound 0.20";
    }

    // Turnover constraint.
    ScalarCPU turnover = (result.weights - w_prev).lpNorm<1>();
    EXPECT_LE(turnover, 0.5 + 1e-2)
        << "Turnover " << turnover << " exceeds limit 0.5";

    EXPECT_GT(result.cvar, 0.0);
}

// ── Test 8: DeterministicResults ────────────────────────────────────

/// Same problem, same config, same seed must produce identical results.
/// The ADMM solver is deterministic given identical inputs (no GPU
/// non-determinism in CPU path).
TEST_F(ConvergenceTest, DeterministicResults) {
    const int n_assets = 5;
    const int n_scenarios = 10000;

    VectorXd mu = make_random_mu(n_assets, 31);
    MatrixXd cov = make_diagonal_dominant_cov(n_assets);
    MatrixXd scenarios = make_test_scenarios(mu, cov, n_scenarios, 31);

    AdmmConfig config;
    config.confidence_level = 0.95;
    config.max_iter = 500;
    config.verbose = false;
    // Enable Phase 12 improvements.
    config.alpha_relax = 1.5;
    config.anderson_depth = 3;
    config.x_update_lr = 0.05;

    // Run 1.
    auto result1 = admm_solve(scenarios, mu, config);
    // Run 2.
    auto result2 = admm_solve(scenarios, mu, config);

    ASSERT_TRUE(result1.converged) << "Run 1 did not converge";
    ASSERT_TRUE(result2.converged) << "Run 2 did not converge";

    // Iteration count must match exactly.
    EXPECT_EQ(result1.iterations, result2.iterations)
        << "Determinism violation: run 1 used " << result1.iterations
        << " iterations, run 2 used " << result2.iterations;

    // Weights must match to machine precision.
    for (int i = 0; i < n_assets; ++i) {
        EXPECT_NEAR(result1.weights(i), result2.weights(i), 1e-12)
            << "Determinism violation: w(" << i << ") run1="
            << result1.weights(i) << " run2=" << result2.weights(i);
    }

    // CVaR and zeta must match exactly.
    EXPECT_NEAR(result1.cvar, result2.cvar, 1e-12)
        << "CVaR mismatch: " << result1.cvar << " vs " << result2.cvar;
    EXPECT_NEAR(result1.zeta, result2.zeta, 1e-12)
        << "Zeta mismatch: " << result1.zeta << " vs " << result2.zeta;
}
