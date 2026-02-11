#include <gtest/gtest.h>

#include <cmath>

#include "simulation/cholesky_utils.h"
#include "simulation/monte_carlo.h"
#include "simulation/scenario_matrix.h"

using namespace cpo;

// ── Cholesky tests ─────────────────────────────────────────────────

TEST(Cholesky, Identity) {
    MatrixXd I = MatrixXd::Identity(3, 3);
    auto result = compute_cholesky(I);

    EXPECT_EQ(result.n, 3);
    // L should be identity for identity covariance.
    for (Index i = 0; i < 3; ++i) {
        for (Index j = 0; j < 3; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            EXPECT_NEAR(result.L_cpu(i, j), expected, 1e-12);
        }
    }
    EXPECT_TRUE(validate_cholesky(result, I));
}

TEST(Cholesky, TwoAsset) {
    // Sigma = [[1.0, 0.5], [0.5, 1.0]]
    MatrixXd cov(2, 2);
    cov << 1.0, 0.5, 0.5, 1.0;

    auto result = compute_cholesky(cov);

    EXPECT_EQ(result.n, 2);

    // L must be lower-triangular.
    EXPECT_NEAR(result.L_cpu(0, 1), 0.0, 1e-12);

    // Validate LLT = Sigma.
    EXPECT_TRUE(validate_cholesky(result, cov));

    // Check L_flat packing (row-major, 2x2).
    EXPECT_EQ(result.L_flat.size(), 4u);
    // L_flat[0] = L(0,0), L_flat[1] = L(0,1)=0, L_flat[2] = L(1,0), L_flat[3] = L(1,1)
    EXPECT_NEAR(result.L_flat[1], 0.0f, 1e-6f);
    EXPECT_NEAR(result.L_flat[0],
                static_cast<float>(result.L_cpu(0, 0)), 1e-6f);
    EXPECT_NEAR(result.L_flat[2],
                static_cast<float>(result.L_cpu(1, 0)), 1e-6f);
    EXPECT_NEAR(result.L_flat[3],
                static_cast<float>(result.L_cpu(1, 1)), 1e-6f);
}

TEST(Cholesky, NotPositiveDefiniteThrows) {
    // Eigenvalues: 1.5 and -0.5 → not PD.
    MatrixXd bad(2, 2);
    bad << 0.5, 1.0, 1.0, 0.5;
    EXPECT_THROW(compute_cholesky(bad), std::runtime_error);
}

TEST(Cholesky, NonSquareThrows) {
    MatrixXd rect(2, 3);
    rect.setZero();
    EXPECT_THROW(compute_cholesky(rect), std::runtime_error);
}

// ── ScenarioMatrix tests ───────────────────────────────────────────

TEST(ScenarioMatrix, AllocateAndTransfer) {
    const Index n_scenarios = 100;
    const Index n_assets = 5;

    ScenarioMatrix mat(n_scenarios, n_assets);
    EXPECT_EQ(mat.n_scenarios(), n_scenarios);
    EXPECT_EQ(mat.n_assets(), n_assets);
    EXPECT_NE(mat.device_ptr(), nullptr);

    // Create host data, upload, download, verify roundtrip.
    MatrixXs host = MatrixXs::Random(n_scenarios, n_assets);
    mat.from_host(host);

    MatrixXs roundtrip = mat.to_host();
    for (Index i = 0; i < n_scenarios; ++i) {
        for (Index j = 0; j < n_assets; ++j) {
            EXPECT_FLOAT_EQ(roundtrip(i, j), host(i, j));
        }
    }
}

TEST(ScenarioMatrix, MoveSemantics) {
    ScenarioMatrix a(50, 3);
    Scalar* ptr = a.device_ptr();

    // Move constructor.
    ScenarioMatrix b(std::move(a));
    EXPECT_EQ(b.device_ptr(), ptr);
    EXPECT_EQ(b.n_scenarios(), 50);
    EXPECT_EQ(b.n_assets(), 3);
    EXPECT_EQ(a.device_ptr(), nullptr);
    EXPECT_EQ(a.n_scenarios(), 0);

    // Move assignment.
    ScenarioMatrix c(10, 2);
    c = std::move(b);
    EXPECT_EQ(c.device_ptr(), ptr);
    EXPECT_EQ(b.device_ptr(), nullptr);
}

// ── Monte Carlo GPU tests ──────────────────────────────────────────

class MonteCarloGPUTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 2-asset setup: rho = 0.6
        mu2_.resize(2);
        mu2_ << 0.05, 0.10;

        cov2_.resize(2, 2);
        double sigma1 = 0.20, sigma2 = 0.30, rho = 0.6;
        cov2_ << sigma1 * sigma1, rho * sigma1 * sigma2,
                 rho * sigma1 * sigma2, sigma2 * sigma2;

        chol2_ = compute_cholesky(cov2_);
    }

    VectorXd mu2_;
    MatrixXd cov2_;
    CholeskyResult chol2_;
};

TEST_F(MonteCarloGPUTest, MeanConvergence2Asset) {
    MonteCarloConfig cfg;
    cfg.n_scenarios = 100000;
    cfg.seed = 12345;

    auto scenarios = generate_scenarios_gpu(mu2_, chol2_, cfg);
    MatrixXs host = scenarios.to_host();

    // Sample mean should converge to mu within 3 * sigma / sqrt(N).
    // sigma1 = 0.20, sigma2 = 0.30, N = 100000.
    for (Index j = 0; j < 2; ++j) {
        double sample_mean = 0.0;
        for (Index i = 0; i < cfg.n_scenarios; ++i) {
            sample_mean += static_cast<double>(host(i, j));
        }
        sample_mean /= cfg.n_scenarios;

        double sigma = std::sqrt(cov2_(j, j));
        double tol = 3.0 * sigma / std::sqrt(static_cast<double>(cfg.n_scenarios));
        EXPECT_NEAR(sample_mean, mu2_(j), tol)
            << "Asset " << j << ": sample_mean=" << sample_mean
            << " expected=" << mu2_(j) << " tol=" << tol;
    }
}

TEST_F(MonteCarloGPUTest, CovarianceConvergence2Asset) {
    MonteCarloConfig cfg;
    cfg.n_scenarios = 200000;
    cfg.seed = 99999;

    auto scenarios = generate_scenarios_gpu(mu2_, chol2_, cfg);
    MatrixXs host = scenarios.to_host();

    // Compute sample covariance.
    const int N = cfg.n_scenarios;
    VectorXd mean = VectorXd::Zero(2);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < 2; ++j) {
            mean(j) += static_cast<double>(host(i, j));
        }
    }
    mean /= N;

    MatrixXd sample_cov = MatrixXd::Zero(2, 2);
    for (int i = 0; i < N; ++i) {
        VectorXd diff(2);
        diff(0) = static_cast<double>(host(i, 0)) - mean(0);
        diff(1) = static_cast<double>(host(i, 1)) - mean(1);
        sample_cov += diff * diff.transpose();
    }
    sample_cov /= (N - 1);

    // 5% relative tolerance for float precision.
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            double expected = cov2_(i, j);
            double actual = sample_cov(i, j);
            double rel_err = std::abs(actual - expected) /
                             std::max(std::abs(expected), 1e-10);
            EXPECT_LT(rel_err, 0.05)
                << "Cov(" << i << "," << j << "): actual=" << actual
                << " expected=" << expected << " rel_err=" << rel_err;
        }
    }
}

TEST_F(MonteCarloGPUTest, CorrelationStructure) {
    // Use rho = 0.8 for clearer signal.
    MatrixXd cov_high(2, 2);
    double sigma1 = 0.20, sigma2 = 0.30, rho = 0.8;
    cov_high << sigma1 * sigma1, rho * sigma1 * sigma2,
                rho * sigma1 * sigma2, sigma2 * sigma2;
    auto chol_high = compute_cholesky(cov_high);

    VectorXd mu = VectorXd::Zero(2);
    MonteCarloConfig cfg;
    cfg.n_scenarios = 200000;
    cfg.seed = 55555;

    auto scenarios = generate_scenarios_gpu(mu, chol_high, cfg);
    MatrixXs host = scenarios.to_host();

    // Compute empirical correlation.
    const int N = cfg.n_scenarios;
    double sum_x = 0, sum_y = 0, sum_x2 = 0, sum_y2 = 0, sum_xy = 0;
    for (int i = 0; i < N; ++i) {
        double x = static_cast<double>(host(i, 0));
        double y = static_cast<double>(host(i, 1));
        sum_x += x;
        sum_y += y;
        sum_x2 += x * x;
        sum_y2 += y * y;
        sum_xy += x * y;
    }
    double mean_x = sum_x / N;
    double mean_y = sum_y / N;
    double var_x = sum_x2 / N - mean_x * mean_x;
    double var_y = sum_y2 / N - mean_y * mean_y;
    double cov_xy = sum_xy / N - mean_x * mean_y;
    double empirical_rho = cov_xy / std::sqrt(var_x * var_y);

    EXPECT_NEAR(empirical_rho, rho, 0.02)
        << "Empirical rho=" << empirical_rho << " expected=" << rho;
}

TEST_F(MonteCarloGPUTest, Reproducibility) {
    MonteCarloConfig cfg;
    cfg.n_scenarios = 10000;
    cfg.seed = 42;

    auto s1 = generate_scenarios_gpu(mu2_, chol2_, cfg);
    auto s2 = generate_scenarios_gpu(mu2_, chol2_, cfg);

    MatrixXs h1 = s1.to_host();
    MatrixXs h2 = s2.to_host();

    for (Index i = 0; i < cfg.n_scenarios; ++i) {
        for (Index j = 0; j < 2; ++j) {
            EXPECT_FLOAT_EQ(h1(i, j), h2(i, j))
                << "Mismatch at (" << i << "," << j << ")";
        }
    }
}

TEST_F(MonteCarloGPUTest, DifferentSeeds) {
    MonteCarloConfig cfg1;
    cfg1.n_scenarios = 1000;
    cfg1.seed = 42;

    MonteCarloConfig cfg2 = cfg1;
    cfg2.seed = 123;

    auto s1 = generate_scenarios_gpu(mu2_, chol2_, cfg1);
    auto s2 = generate_scenarios_gpu(mu2_, chol2_, cfg2);

    MatrixXs h1 = s1.to_host();
    MatrixXs h2 = s2.to_host();

    // Not all values should be equal.
    int diff_count = 0;
    for (Index i = 0; i < cfg1.n_scenarios; ++i) {
        for (Index j = 0; j < 2; ++j) {
            if (h1(i, j) != h2(i, j)) ++diff_count;
        }
    }
    EXPECT_GT(diff_count, cfg1.n_scenarios);  // Most values differ.
}

TEST_F(MonteCarloGPUTest, CurandStatesReuse) {
    MonteCarloConfig cfg;
    cfg.n_scenarios = 5000;
    cfg.seed = 77;

    auto states = create_curand_states(cfg.n_scenarios, cfg.seed);

    // First call with pre-allocated states.
    auto s1 = generate_scenarios_gpu(mu2_, chol2_, cfg, states.get());
    MatrixXs h1 = s1.to_host();

    // Second call with same states object (states should have advanced).
    auto s2 = generate_scenarios_gpu(mu2_, chol2_, cfg, states.get());
    MatrixXs h2 = s2.to_host();

    // Results should differ since RNG state advanced.
    int diff_count = 0;
    for (Index i = 0; i < cfg.n_scenarios; ++i) {
        for (Index j = 0; j < 2; ++j) {
            if (h1(i, j) != h2(i, j)) ++diff_count;
        }
    }
    EXPECT_GT(diff_count, cfg.n_scenarios);
}

TEST(MonteCarloGPU, FiveAssetConvergence) {
    // 5-asset with dense correlation structure.
    const Index n = 5;
    VectorXd mu(n);
    mu << 0.02, 0.05, 0.08, 0.03, 0.06;

    // Build a PD covariance: D * R * D where D = diag(sigmas), R = correlation.
    VectorXd sigmas(n);
    sigmas << 0.15, 0.20, 0.25, 0.18, 0.22;

    MatrixXd R = MatrixXd::Identity(n, n);
    // Fill off-diagonals with moderate correlations.
    double corrs[] = {0.5, 0.3, 0.2, 0.4, 0.6, 0.1, 0.3, 0.5, 0.2, 0.4};
    int k = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            R(i, j) = corrs[k];
            R(j, i) = corrs[k];
            ++k;
        }
    }

    MatrixXd D = sigmas.asDiagonal();
    MatrixXd cov = D * R * D;
    auto chol = compute_cholesky(cov);

    MonteCarloConfig cfg;
    cfg.n_scenarios = 200000;
    cfg.seed = 31415;

    auto scenarios = generate_scenarios_gpu(mu, chol, cfg);
    MatrixXs host = scenarios.to_host();

    // Mean convergence: 3 * sigma / sqrt(N).
    for (Index j = 0; j < n; ++j) {
        double sample_mean = 0.0;
        for (Index i = 0; i < cfg.n_scenarios; ++i) {
            sample_mean += static_cast<double>(host(i, j));
        }
        sample_mean /= cfg.n_scenarios;

        double tol = 3.0 * sigmas(j) / std::sqrt(
            static_cast<double>(cfg.n_scenarios));
        EXPECT_NEAR(sample_mean, mu(j), tol)
            << "Asset " << j << ": sample_mean=" << sample_mean
            << " expected=" << mu(j);
    }
}

// ── Monte Carlo CPU tests ──────────────────────────────────────────

TEST(MonteCarloCPU, MeanConvergence2Asset) {
    VectorXd mu(2);
    mu << 0.05, 0.10;

    MatrixXd cov(2, 2);
    double sigma1 = 0.20, sigma2 = 0.30, rho = 0.6;
    cov << sigma1 * sigma1, rho * sigma1 * sigma2,
           rho * sigma1 * sigma2, sigma2 * sigma2;
    auto chol = compute_cholesky(cov);

    MonteCarloConfig cfg;
    cfg.n_scenarios = 100000;
    cfg.seed = 12345;

    MatrixXd result = generate_scenarios_cpu(mu, chol, cfg);

    for (Index j = 0; j < 2; ++j) {
        double sample_mean = result.col(j).mean();
        double sigma = std::sqrt(cov(j, j));
        double tol = 3.0 * sigma / std::sqrt(
            static_cast<double>(cfg.n_scenarios));
        EXPECT_NEAR(sample_mean, mu(j), tol)
            << "Asset " << j << ": sample_mean=" << sample_mean
            << " expected=" << mu(j);
    }
}

TEST(MonteCarloCPU, CovarianceConvergence2Asset) {
    VectorXd mu(2);
    mu << 0.05, 0.10;

    MatrixXd cov(2, 2);
    double sigma1 = 0.20, sigma2 = 0.30, rho = 0.6;
    cov << sigma1 * sigma1, rho * sigma1 * sigma2,
           rho * sigma1 * sigma2, sigma2 * sigma2;
    auto chol = compute_cholesky(cov);

    MonteCarloConfig cfg;
    cfg.n_scenarios = 200000;
    cfg.seed = 54321;

    MatrixXd result = generate_scenarios_cpu(mu, chol, cfg);

    // Compute sample covariance.
    VectorXd mean = result.colwise().mean();
    MatrixXd centered = result.rowwise() - mean.transpose();
    MatrixXd sample_cov = (centered.transpose() * centered) /
                          (cfg.n_scenarios - 1);

    // 3% relative tolerance for double precision.
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            double expected = cov(i, j);
            double actual = sample_cov(i, j);
            double rel_err = std::abs(actual - expected) /
                             std::max(std::abs(expected), 1e-10);
            EXPECT_LT(rel_err, 0.03)
                << "Cov(" << i << "," << j << "): actual=" << actual
                << " expected=" << expected << " rel_err=" << rel_err;
        }
    }
}
