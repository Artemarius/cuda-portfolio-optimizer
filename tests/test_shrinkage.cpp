#include <gtest/gtest.h>

#include <Eigen/Eigenvalues>
#include <cmath>
#include <random>

#include "models/shrinkage_estimator.h"

namespace cpo {
namespace {

// Helper: generate a T x N return matrix from N(0, Sigma) with given seed.
MatrixXd generate_returns(const VectorXd& mu, const MatrixXd& sigma,
                          Index T, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> dist(0.0, 1.0);

    const Index N = static_cast<Index>(mu.size());

    // Cholesky of sigma.
    Eigen::LLT<MatrixXd> llt(sigma);
    MatrixXd L = llt.matrixL();

    MatrixXd returns(T, N);
    for (Index t = 0; t < T; ++t) {
        VectorXd z(N);
        for (Index i = 0; i < N; ++i) {
            z(i) = dist(rng);
        }
        returns.row(t) = (mu + L * z).transpose();
    }
    return returns;
}

// ── Intensity bounds ─────────────────────────────────────────────────

TEST(LedoitWolfTest, IntensityBounds) {
    // Any data should yield intensity in [0, 1].
    const Index N = 5;
    const Index T = 50;
    VectorXd mu = VectorXd::Zero(N);
    MatrixXd sigma = MatrixXd::Identity(N, N);
    MatrixXd returns = generate_returns(mu, sigma, T);

    auto result = ledoit_wolf_shrink(returns);
    EXPECT_GE(result.intensity, 0.0);
    EXPECT_LE(result.intensity, 1.0);
}

// ── Positive definite guarantee ──────────────────────────────────────

TEST(LedoitWolfTest, PositiveDefinite) {
    const Index N = 10;
    const Index T = 30;
    VectorXd mu = VectorXd::Zero(N);
    MatrixXd sigma = MatrixXd::Identity(N, N);
    MatrixXd returns = generate_returns(mu, sigma, T);

    auto result = ledoit_wolf_shrink(returns);

    // Check all eigenvalues are positive.
    Eigen::SelfAdjointEigenSolver<MatrixXd> solver(result.covariance);
    VectorXd eigenvalues = solver.eigenvalues();
    for (Index i = 0; i < N; ++i) {
        EXPECT_GT(eigenvalues(i), 0.0)
            << "Eigenvalue " << i << " is non-positive: " << eigenvalues(i);
    }
}

// ── Symmetry ─────────────────────────────────────────────────────────

TEST(LedoitWolfTest, Symmetric) {
    const Index N = 5;
    const Index T = 100;
    VectorXd mu = VectorXd::Zero(N);
    MatrixXd sigma = MatrixXd::Identity(N, N);
    MatrixXd returns = generate_returns(mu, sigma, T);

    auto result = ledoit_wolf_shrink(returns);
    double asym = (result.covariance - result.covariance.transpose()).norm();
    EXPECT_LT(asym, 1e-14);
}

// ── Identity covariance: intensity should be HIGH ────────────────────

TEST(LedoitWolfTest, IdentityCovarianceHighIntensity) {
    // When data is drawn from N(0, I), the true covariance IS the shrinkage
    // target (scaled identity). The LW estimator correctly detects this and
    // assigns HIGH intensity — "trust the target, it's correct."
    // This is the optimal behavior: the oracle intensity β²/δ² → ∞ (clamped
    // to 1) when δ² → 0 (target = truth).
    const Index N = 5;
    const Index T = 5000;
    VectorXd mu = VectorXd::Zero(N);
    MatrixXd sigma = MatrixXd::Identity(N, N);
    MatrixXd returns = generate_returns(mu, sigma, T);

    auto result = ledoit_wolf_shrink(returns);

    // True cov = target → intensity should be high (near 1).
    EXPECT_GT(result.intensity, 0.5)
        << "Expected high intensity when true cov = target, got "
        << result.intensity;
}

// ── T >> N: intensity approaches 0 ──────────────────────────────────

TEST(LedoitWolfTest, LargeSampleLowIntensity) {
    const Index N = 3;
    const Index T = 10000;
    VectorXd mu = VectorXd::Zero(N);

    // Non-trivial covariance.
    MatrixXd sigma(N, N);
    sigma << 1.0, 0.5, 0.3,
             0.5, 1.0, 0.4,
             0.3, 0.4, 1.0;
    MatrixXd returns = generate_returns(mu, sigma, T);

    auto result = ledoit_wolf_shrink(returns);
    EXPECT_LT(result.intensity, 0.05)
        << "Expected intensity near 0 for T >> N, got " << result.intensity;
}

// ── T = N regime: moderate to high shrinkage ─────────────────────────

TEST(LedoitWolfTest, SquareCaseModerateIntensity) {
    const Index N = 20;
    const Index T = 20;
    VectorXd mu = VectorXd::Zero(N);
    MatrixXd sigma = MatrixXd::Identity(N, N);
    MatrixXd returns = generate_returns(mu, sigma, T);

    auto result = ledoit_wolf_shrink(returns);

    // When T = N, sample covariance is poorly estimated; expect significant shrinkage.
    EXPECT_GT(result.intensity, 0.1)
        << "Expected moderate intensity for T = N, got " << result.intensity;
}

// ── T < N: high shrinkage (singular sample cov) ─────────────────────

TEST(LedoitWolfTest, UnderdeterminedHighIntensity) {
    const Index N = 30;
    const Index T = 10;
    VectorXd mu = VectorXd::Zero(N);
    MatrixXd sigma = MatrixXd::Identity(N, N);
    MatrixXd returns = generate_returns(mu, sigma, T);

    auto result = ledoit_wolf_shrink(returns);

    // When T < N, sample covariance is singular; intensity should be high.
    EXPECT_GT(result.intensity, 0.3)
        << "Expected high intensity for T < N, got " << result.intensity;

    // Even in the underdetermined case, result should be PD.
    Eigen::SelfAdjointEigenSolver<MatrixXd> solver(result.covariance);
    EXPECT_GT(solver.eigenvalues().minCoeff(), 0.0);
}

// ── Single factor: high correlation → shrinkage helps ────────────────

TEST(LedoitWolfTest, SingleFactorPositiveIntensity) {
    // All assets driven by one factor: high off-diagonal correlation.
    const Index N = 10;
    const Index T = 50;

    // Sigma = beta * beta' + D where beta is a single column.
    VectorXd beta = VectorXd::Ones(N) * 0.8;
    MatrixXd sigma = beta * beta.transpose()
                     + 0.2 * MatrixXd::Identity(N, N);

    VectorXd mu = VectorXd::Zero(N);
    MatrixXd returns = generate_returns(mu, sigma, T);

    auto result = ledoit_wolf_shrink(returns);

    // With structured covariance and moderate T/N, expect non-trivial shrinkage.
    EXPECT_GT(result.intensity, 0.0);
    EXPECT_LE(result.intensity, 1.0);
}

// ── Matches naive at same intensity ──────────────────────────────────

TEST(LedoitWolfTest, ConsistentWithNaiveShrinkage) {
    // The shrunk covariance should satisfy:
    //   S_shrunk = (1 - d) * S + d * (trace(S)/N) * I
    // where d = result.intensity and S = result.sample_cov.
    const Index N = 5;
    const Index T = 100;
    VectorXd mu = VectorXd::Zero(N);
    MatrixXd sigma = MatrixXd::Identity(N, N);
    MatrixXd returns = generate_returns(mu, sigma, T);

    auto result = ledoit_wolf_shrink(returns);

    double trace_over_n = result.sample_cov.trace() / static_cast<double>(N);
    MatrixXd expected = (1.0 - result.intensity) * result.sample_cov
                        + result.intensity * trace_over_n
                          * MatrixXd::Identity(N, N);

    double diff = (result.covariance - expected).norm();
    EXPECT_LT(diff, 1e-12)
        << "Shrunk covariance doesn't match naive formula at same intensity";
}

// ── Dimension checks ─────────────────────────────────────────────────

TEST(LedoitWolfTest, OutputDimensions) {
    const Index N = 7;
    const Index T = 50;
    VectorXd mu = VectorXd::Zero(N);
    MatrixXd sigma = MatrixXd::Identity(N, N);
    MatrixXd returns = generate_returns(mu, sigma, T);

    auto result = ledoit_wolf_shrink(returns);

    EXPECT_EQ(result.covariance.rows(), N);
    EXPECT_EQ(result.covariance.cols(), N);
    EXPECT_EQ(result.sample_cov.rows(), N);
    EXPECT_EQ(result.sample_cov.cols(), N);
}

// ── Error handling ───────────────────────────────────────────────────

TEST(LedoitWolfTest, TooFewObservations) {
    MatrixXd returns(1, 3);
    returns << 0.01, 0.02, 0.03;
    EXPECT_THROW(ledoit_wolf_shrink(returns), std::invalid_argument);
}

TEST(LedoitWolfTest, EmptyMatrix) {
    MatrixXd returns(0, 0);
    EXPECT_THROW(ledoit_wolf_shrink(returns), std::invalid_argument);
}

// ── 2-asset known structure ──────────────────────────────────────────

TEST(LedoitWolfTest, TwoAssetKnownStructure) {
    // With 2 assets, we can verify the formula manually.
    const Index N = 2;
    const Index T = 500;

    MatrixXd sigma(N, N);
    sigma << 0.04, 0.01,
             0.01, 0.09;

    VectorXd mu = VectorXd::Zero(N);
    MatrixXd returns = generate_returns(mu, sigma, T);

    auto result = ledoit_wolf_shrink(returns);

    // Basic sanity: intensity in [0,1], result is PD.
    EXPECT_GE(result.intensity, 0.0);
    EXPECT_LE(result.intensity, 1.0);

    Eigen::SelfAdjointEigenSolver<MatrixXd> solver(result.covariance);
    EXPECT_GT(solver.eigenvalues().minCoeff(), 0.0);

    // Shrunk cov should be closer to target than sample cov (in Frobenius norm)
    // or at least no worse (by construction).
    double mu_hat = result.sample_cov.trace() / 2.0;
    MatrixXd target = mu_hat * MatrixXd::Identity(N, N);
    double dist_sample = (result.sample_cov - target).squaredNorm();
    double dist_shrunk = (result.covariance - target).squaredNorm();
    // Shrunk should be between sample and target in distance.
    EXPECT_LE(dist_shrunk, dist_sample + 1e-12);
}

// ── Monotonicity: more data → less shrinkage ─────────────────────────

TEST(LedoitWolfTest, MoreDataLessShrinkage) {
    const Index N = 10;

    VectorXd mu = VectorXd::Zero(N);
    MatrixXd sigma = MatrixXd::Identity(N, N);
    // Add some off-diagonal structure.
    for (Index i = 0; i < N - 1; ++i) {
        sigma(i, i + 1) = 0.3;
        sigma(i + 1, i) = 0.3;
    }

    MatrixXd returns_small = generate_returns(mu, sigma, 30, 123);
    MatrixXd returns_large = generate_returns(mu, sigma, 3000, 123);

    auto result_small = ledoit_wolf_shrink(returns_small);
    auto result_large = ledoit_wolf_shrink(returns_large);

    // More data should generally yield lower intensity.
    EXPECT_GT(result_small.intensity, result_large.intensity)
        << "Expected more shrinkage with less data: small_T intensity="
        << result_small.intensity << ", large_T intensity=" << result_large.intensity;
}

// ── Single asset (degenerate case) ───────────────────────────────────

TEST(LedoitWolfTest, SingleAsset) {
    const Index N = 1;
    const Index T = 100;

    std::mt19937 rng(42);
    std::normal_distribution<double> dist(0.0, 0.1);
    MatrixXd returns(T, N);
    for (Index t = 0; t < T; ++t) {
        returns(t, 0) = dist(rng);
    }

    auto result = ledoit_wolf_shrink(returns);

    // For N=1, S is 1x1 and equals mu_hat * I already. Intensity = 0.
    EXPECT_NEAR(result.intensity, 0.0, 1e-10);
    EXPECT_EQ(result.covariance.rows(), 1);
    EXPECT_EQ(result.covariance.cols(), 1);
}

}  // anonymous namespace
}  // namespace cpo
