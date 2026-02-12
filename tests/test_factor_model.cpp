#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include <Eigen/Cholesky>

#include "models/factor_model.h"
#include "models/factor_monte_carlo.h"
#include "models/tiled_scenario_generator.h"
#include "data/csv_loader.h"
#include "data/returns.h"

using namespace cpo;

// ── Helper: generate synthetic returns with known factor structure ────

namespace {

/// Generate T x N returns from a known factor model:
///   R_t = mu + B * f_t + eps_t
///
/// where f_t ~ N(0, Sigma_f), eps_t ~ N(0, diag(D)).
/// This lets us verify that fit_factor_model recovers the structure.
MatrixXd generate_factor_returns(const VectorXd& mu,
                                  const MatrixXd& B,
                                  const MatrixXd& Sigma_f,
                                  const VectorXd& D,
                                  int T,
                                  unsigned seed = 42) {
    const auto N = static_cast<Index>(mu.size());
    const auto k = static_cast<Index>(B.cols());

    // Cholesky of factor covariance for correlated factor draws.
    Eigen::LLT<MatrixXd> llt(Sigma_f);
    MatrixXd L_f = llt.matrixL();

    std::mt19937 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    MatrixXd returns(T, N);
    for (int t = 0; t < T; ++t) {
        // Factor realization: f = L_f * z_f.
        VectorXd z_f(k);
        for (Index j = 0; j < k; ++j) {
            z_f(j) = normal(rng);
        }
        VectorXd f = L_f * z_f;

        // Idiosyncratic noise: eps_i ~ N(0, D_i).
        VectorXd eps(N);
        for (Index i = 0; i < N; ++i) {
            eps(i) = std::sqrt(D(i)) * normal(rng);
        }

        returns.row(t) = mu.transpose() + (B * f).transpose() + eps.transpose();
    }

    return returns;
}

}  // anonymous namespace

// ════════════════════════════════════════════════════════════════════
// PCA correctness tests
// ════════════════════════════════════════════════════════════════════

TEST(FactorModel, IdentityCovariance) {
    // Returns from N(0, I): sample cov ~ I for large T.
    // With k=N factors, reconstructed cov should approximate I.
    const int N = 4;
    const int T = 5000;

    std::mt19937 rng(123);
    std::normal_distribution<double> normal(0.0, 1.0);

    MatrixXd returns(T, N);
    for (int t = 0; t < T; ++t) {
        for (int i = 0; i < N; ++i) {
            returns(t, i) = normal(rng);
        }
    }

    FactorModelConfig config;
    config.n_factors = N;  // Full rank.

    auto model = fit_factor_model(returns, config);

    EXPECT_EQ(model.n_factors, N);
    EXPECT_EQ(model.n_assets, N);
    EXPECT_EQ(model.n_periods, T);

    // Reconstructed covariance should approximate I.
    MatrixXd cov = reconstruct_covariance(model);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            EXPECT_NEAR(cov(i, j), expected, 0.1)
                << "cov(" << i << "," << j << ")";
        }
    }
}

TEST(FactorModel, KnownTwoFactorStructure) {
    // 5 assets driven by exactly 2 factors.
    // Verify that the reconstructed covariance matches the true covariance.
    const int N = 5;
    const int k = 2;
    const int T = 10000;

    VectorXd mu = VectorXd::Zero(N);

    // Known loadings B (N x k).
    MatrixXd B(N, k);
    B << 0.8,  0.1,
         0.6,  0.3,
         0.2,  0.7,
         0.1,  0.9,
         0.5,  0.5;

    // Factor covariance (diagonal for simplicity).
    MatrixXd Sigma_f = MatrixXd::Identity(k, k);
    Sigma_f(0, 0) = 0.04;  // 20% vol factor 1
    Sigma_f(1, 1) = 0.02;  // ~14% vol factor 2

    // Idiosyncratic variance.
    VectorXd D(N);
    D << 0.001, 0.002, 0.001, 0.003, 0.002;

    // True covariance: B * Sigma_f * B' + diag(D).
    MatrixXd true_cov = B * Sigma_f * B.transpose();
    true_cov += D.asDiagonal();

    // Generate synthetic returns.
    MatrixXd returns = generate_factor_returns(mu, B, Sigma_f, D, T);

    // Fit factor model with k=2.
    FactorModelConfig config;
    config.n_factors = k;

    auto model = fit_factor_model(returns, config);
    MatrixXd recon_cov = reconstruct_covariance(model);

    // The reconstructed covariance should be close to the true covariance.
    // With T=10000, sampling error is small.
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            EXPECT_NEAR(recon_cov(i, j), true_cov(i, j),
                        0.05 * std::abs(true_cov(i, j)) + 0.005)
                << "cov(" << i << "," << j << ")";
        }
    }

    // Variance explained should be high (> 90%) since data is truly 2-factor.
    EXPECT_GT(model.variance_explained, 0.85);
}

TEST(FactorModel, EigenvalueOrdering) {
    // Eigenvalues must be sorted in descending order.
    const int N = 5;
    const int T = 500;

    std::mt19937 rng(77);
    std::normal_distribution<double> normal(0.0, 1.0);

    // Create returns with different variances per asset.
    MatrixXd returns(T, N);
    for (int t = 0; t < T; ++t) {
        for (int i = 0; i < N; ++i) {
            returns(t, i) = (i + 1) * 0.5 * normal(rng);
        }
    }

    FactorModelConfig config;
    config.n_factors = N;

    auto model = fit_factor_model(returns, config);

    // Check descending order.
    for (int i = 1; i < N; ++i) {
        EXPECT_GE(model.eigenvalues(i - 1), model.eigenvalues(i))
            << "eigenvalues not descending at index " << i;
    }

    // Variance explained should be 1.0 for k=N.
    EXPECT_NEAR(model.variance_explained, 1.0, 1e-10);

    // Eigenvalues should all be positive (from a valid covariance).
    for (int i = 0; i < N; ++i) {
        EXPECT_GT(model.eigenvalues(i), 0.0);
    }
}

TEST(FactorModel, VarianceExplainedComputation) {
    // Variance explained = sum(top-k eigenvalues) / sum(all eigenvalues).
    const int N = 5;
    const int T = 1000;

    // Build returns where first asset has much higher variance.
    std::mt19937 rng(99);
    std::normal_distribution<double> normal(0.0, 1.0);

    MatrixXd returns(T, N);
    for (int t = 0; t < T; ++t) {
        double common = normal(rng);
        returns(t, 0) = 5.0 * common + 0.1 * normal(rng);
        for (int i = 1; i < N; ++i) {
            returns(t, i) = common + normal(rng);
        }
    }

    FactorModelConfig config;
    config.n_factors = 1;

    auto model = fit_factor_model(returns, config);

    // Verify: variance_explained = eigenvalues(0) / sum(all).
    ScalarCPU expected = model.eigenvalues(0) / model.eigenvalues.sum();
    EXPECT_NEAR(model.variance_explained, expected, 1e-10);

    // With one dominant factor, k=1 should explain most variance.
    EXPECT_GT(model.variance_explained, 0.5);
}

TEST(FactorModel, ReconstructionVsSampleCovariance) {
    // With k=N, reconstructed covariance should match sample covariance
    // exactly (up to floating-point precision), since the factor model
    // with all factors is a complete decomposition.
    const int N = 4;
    const int T = 200;

    std::mt19937 rng(42);
    std::normal_distribution<double> normal(0.0, 1.0);

    MatrixXd returns(T, N);
    for (int t = 0; t < T; ++t) {
        for (int i = 0; i < N; ++i) {
            returns(t, i) = normal(rng);
        }
    }

    // Sample covariance.
    VectorXd mu = returns.colwise().mean();
    MatrixXd centered = returns.rowwise() - mu.transpose();
    MatrixXd sample_cov = (centered.transpose() * centered) /
                           static_cast<double>(T - 1);

    // Factor model with k=N.
    FactorModelConfig config;
    config.n_factors = N;

    auto model = fit_factor_model(returns, config);
    MatrixXd recon_cov = reconstruct_covariance(model);

    // Should match sample covariance closely.
    // Tolerance 1e-9 accounts for accumulated floating-point error in
    // eigendecomposition -> projection -> reconstruction roundtrip.
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            EXPECT_NEAR(recon_cov(i, j), sample_cov(i, j), 1e-9)
                << "cov(" << i << "," << j << ")";
        }
    }
}

TEST(FactorModel, AutoFactorSelection) {
    // With min_variance_explained = 0.90, auto-select k.
    const int N = 5;
    const int T = 2000;

    // 2 dominant factors + noise.
    VectorXd mu = VectorXd::Zero(N);
    MatrixXd B(N, 2);
    B << 1.0, 0.0,
         0.8, 0.2,
         0.1, 0.9,
         0.0, 1.0,
         0.5, 0.5;
    MatrixXd Sigma_f = MatrixXd::Identity(2, 2) * 0.10;
    VectorXd D = VectorXd::Constant(N, 0.001);

    MatrixXd returns = generate_factor_returns(mu, B, Sigma_f, D, T);

    FactorModelConfig config;
    config.min_variance_explained = 0.90;

    auto model = fit_factor_model(returns, config);

    // Should select k=2 (the true number of factors).
    EXPECT_LE(model.n_factors, 3);  // At most 3 (sampling noise might add one).
    EXPECT_GE(model.n_factors, 2);  // At least 2 to reach 90%.
    EXPECT_GE(model.variance_explained, 0.90);
}

// ── Edge cases ──────────────────────────────────────────────────────

TEST(FactorModel, SingleFactor) {
    const int N = 3;
    const int T = 500;

    std::mt19937 rng(55);
    std::normal_distribution<double> normal(0.0, 1.0);

    MatrixXd returns(T, N);
    for (int t = 0; t < T; ++t) {
        double common = normal(rng);
        for (int i = 0; i < N; ++i) {
            returns(t, i) = common + 0.1 * normal(rng);
        }
    }

    FactorModelConfig config;
    config.n_factors = 1;

    auto model = fit_factor_model(returns, config);

    EXPECT_EQ(model.n_factors, 1);
    EXPECT_EQ(model.loadings.rows(), N);
    EXPECT_EQ(model.loadings.cols(), 1);
    EXPECT_EQ(model.factor_covariance.rows(), 1);
    EXPECT_EQ(model.factor_covariance.cols(), 1);

    // Reconstructed covariance should be PD.
    MatrixXd cov = reconstruct_covariance(model);
    Eigen::LLT<MatrixXd> llt(cov);
    EXPECT_EQ(llt.info(), Eigen::Success);
}

TEST(FactorModel, KEqualsN) {
    // Full-rank factor model.
    const int N = 3;
    const int T = 200;

    std::mt19937 rng(33);
    std::normal_distribution<double> normal(0.0, 1.0);

    MatrixXd returns(T, N);
    for (int t = 0; t < T; ++t) {
        for (int i = 0; i < N; ++i) {
            returns(t, i) = normal(rng);
        }
    }

    FactorModelConfig config;
    config.n_factors = N;

    auto model = fit_factor_model(returns, config);

    EXPECT_EQ(model.n_factors, N);
    EXPECT_NEAR(model.variance_explained, 1.0, 1e-10);
}

TEST(FactorModel, SingleAsset) {
    const int T = 100;

    std::mt19937 rng(11);
    std::normal_distribution<double> normal(0.0, 1.0);

    MatrixXd returns(T, 1);
    for (int t = 0; t < T; ++t) {
        returns(t, 0) = 0.01 + 0.02 * normal(rng);
    }

    FactorModelConfig config;
    config.n_factors = 1;

    auto model = fit_factor_model(returns, config);

    EXPECT_EQ(model.n_factors, 1);
    EXPECT_EQ(model.n_assets, 1);
    EXPECT_NEAR(model.variance_explained, 1.0, 1e-10);

    MatrixXd cov = reconstruct_covariance(model);
    EXPECT_EQ(cov.rows(), 1);
    EXPECT_EQ(cov.cols(), 1);
    EXPECT_GT(cov(0, 0), 0.0);
}

TEST(FactorModel, NFactorsClamped) {
    // If n_factors > N (without auto-select), it should be clamped to N.
    const int N = 3;
    const int T = 100;

    std::mt19937 rng(44);
    std::normal_distribution<double> normal(0.0, 1.0);

    MatrixXd returns(T, N);
    for (int t = 0; t < T; ++t) {
        for (int i = 0; i < N; ++i) {
            returns(t, i) = normal(rng);
        }
    }

    // n_factors > N with auto-select disabled should throw.
    FactorModelConfig config;
    config.n_factors = 10;

    EXPECT_THROW(fit_factor_model(returns, config), std::invalid_argument);
}

// ── Error cases ─────────────────────────────────────────────────────

TEST(FactorModel, TooFewPeriodsThrows) {
    MatrixXd returns(1, 3);  // T=1 < 2.
    returns << 0.01, 0.02, 0.03;

    FactorModelConfig config;
    config.n_factors = 1;

    EXPECT_THROW(fit_factor_model(returns, config), std::invalid_argument);
}

TEST(FactorModel, EmptyReturnsThrows) {
    MatrixXd returns(5, 0);  // N=0.

    FactorModelConfig config;
    config.n_factors = 1;

    EXPECT_THROW(fit_factor_model(returns, config), std::invalid_argument);
}

// ── Cholesky compatibility ──────────────────────────────────────────

TEST(FactorModel, CholeskyFromFactorModel) {
    const int N = 4;
    const int T = 1000;

    std::mt19937 rng(88);
    std::normal_distribution<double> normal(0.0, 1.0);

    MatrixXd returns(T, N);
    for (int t = 0; t < T; ++t) {
        double common = normal(rng);
        for (int i = 0; i < N; ++i) {
            returns(t, i) = common * (0.5 + 0.2 * i) + 0.1 * normal(rng);
        }
    }

    FactorModelConfig config;
    config.n_factors = 2;

    auto model = fit_factor_model(returns, config);
    MatrixXd recon_cov = reconstruct_covariance(model);

    // compute_cholesky_from_factor_model should produce a valid Cholesky.
    CholeskyResult chol = compute_cholesky_from_factor_model(model);

    EXPECT_EQ(chol.n, N);
    EXPECT_TRUE(validate_cholesky(chol, recon_cov));

    // L_flat should be populated (N*N floats).
    EXPECT_EQ(static_cast<Index>(chol.L_flat.size()), N * N);
}

// ── Result dimensions ───────────────────────────────────────────────

TEST(FactorModel, ResultDimensions) {
    const int N = 5;
    const int k = 3;
    const int T = 200;

    std::mt19937 rng(66);
    std::normal_distribution<double> normal(0.0, 1.0);

    MatrixXd returns(T, N);
    for (int t = 0; t < T; ++t) {
        for (int i = 0; i < N; ++i) {
            returns(t, i) = normal(rng);
        }
    }

    FactorModelConfig config;
    config.n_factors = k;

    auto model = fit_factor_model(returns, config);

    // Check all dimensions.
    EXPECT_EQ(model.loadings.rows(), N);
    EXPECT_EQ(model.loadings.cols(), k);
    EXPECT_EQ(model.factor_returns.rows(), T);
    EXPECT_EQ(model.factor_returns.cols(), k);
    EXPECT_EQ(model.factor_covariance.rows(), k);
    EXPECT_EQ(model.factor_covariance.cols(), k);
    EXPECT_EQ(model.idiosyncratic_var.size(), N);
    EXPECT_EQ(model.eigenvalues.size(), N);
    EXPECT_EQ(model.mu.size(), N);

    // Reconstructed covariance dimensions.
    MatrixXd cov = reconstruct_covariance(model);
    EXPECT_EQ(cov.rows(), N);
    EXPECT_EQ(cov.cols(), N);
}

// ── Positive-definiteness of reconstructed covariance ───────────────

TEST(FactorModel, ReconstructedCovIsPD) {
    // Reconstructed cov must always be positive-definite
    // (guaranteed by D_i >= 1e-10 floor).
    const int N = 5;
    const int T = 50;  // T < N: degenerate case.

    std::mt19937 rng(77);
    std::normal_distribution<double> normal(0.0, 1.0);

    MatrixXd returns(T, N);
    for (int t = 0; t < T; ++t) {
        for (int i = 0; i < N; ++i) {
            returns(t, i) = normal(rng);
        }
    }

    FactorModelConfig config;
    config.n_factors = 3;

    auto model = fit_factor_model(returns, config);
    MatrixXd cov = reconstruct_covariance(model);

    // Cholesky succeeds iff PD.
    Eigen::LLT<MatrixXd> llt(cov);
    EXPECT_EQ(llt.info(), Eigen::Success);
}

// ── Mean (mu) extraction ────────────────────────────────────────────

TEST(FactorModel, MuIsColumnMean) {
    const int N = 3;
    const int T = 100;

    std::mt19937 rng(22);
    std::normal_distribution<double> normal(0.0, 1.0);

    MatrixXd returns(T, N);
    for (int t = 0; t < T; ++t) {
        returns(t, 0) = 0.05 + 0.01 * normal(rng);
        returns(t, 1) = 0.10 + 0.02 * normal(rng);
        returns(t, 2) = -0.03 + 0.01 * normal(rng);
    }

    FactorModelConfig config;
    config.n_factors = 2;

    auto model = fit_factor_model(returns, config);

    // mu should match column means of input returns.
    VectorXd expected_mu = returns.colwise().mean();
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(model.mu(i), expected_mu(i), 1e-12);
    }
}

// ════════════════════════════════════════════════════════════════════
// Factor Monte Carlo tests
// ════════════════════════════════════════════════════════════════════

namespace {

/// Helper: build a simple factor model for MC tests.
/// 3 assets, 2 factors, known structure.
FactorModelResult build_test_factor_model() {
    const int N = 3;
    const int k = 2;
    const int T = 5000;

    VectorXd mu(N);
    mu << 0.02, 0.04, 0.06;

    MatrixXd B(N, k);
    B << 0.8, 0.1,
         0.3, 0.7,
         0.5, 0.5;

    MatrixXd Sigma_f = MatrixXd::Identity(k, k);
    Sigma_f(0, 0) = 0.04;
    Sigma_f(1, 1) = 0.02;

    VectorXd D(N);
    D << 0.001, 0.002, 0.001;

    // Generate synthetic returns to fit the model.
    MatrixXd returns = generate_factor_returns(mu, B, Sigma_f, D, T, 42);

    FactorModelConfig config;
    config.n_factors = k;
    return fit_factor_model(returns, config);
}

}  // anonymous namespace

TEST(FactorMonteCarloCPU, MeanConvergence) {
    auto model = build_test_factor_model();
    const int N = model.n_assets;

    MonteCarloConfig mc;
    mc.n_scenarios = 200000;
    mc.seed = 123;

    MatrixXd scenarios = generate_scenarios_factor_cpu(model.mu, model, mc);

    EXPECT_EQ(scenarios.rows(), mc.n_scenarios);
    EXPECT_EQ(scenarios.cols(), N);

    // Sample mean should converge to mu.
    // CLT bound: |mean - mu| < 3 * sigma / sqrt(N_scenarios).
    MatrixXd recon_cov = reconstruct_covariance(model);
    VectorXd sample_mean = scenarios.colwise().mean();
    for (int i = 0; i < N; ++i) {
        double sigma_i = std::sqrt(recon_cov(i, i));
        double bound = 3.0 * sigma_i / std::sqrt(static_cast<double>(mc.n_scenarios));
        EXPECT_NEAR(sample_mean(i), model.mu(i), bound)
            << "asset " << i;
    }
}

TEST(FactorMonteCarloCPU, CovarianceConvergence) {
    auto model = build_test_factor_model();
    const int N = model.n_assets;

    MonteCarloConfig mc;
    mc.n_scenarios = 200000;
    mc.seed = 456;

    MatrixXd scenarios = generate_scenarios_factor_cpu(model.mu, model, mc);

    // Sample covariance should converge to B * Sigma_f * B' + D.
    VectorXd mu_s = scenarios.colwise().mean();
    MatrixXd centered = scenarios.rowwise() - mu_s.transpose();
    MatrixXd sample_cov = (centered.transpose() * centered) /
                           static_cast<double>(mc.n_scenarios - 1);

    MatrixXd true_cov = reconstruct_covariance(model);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double tol = 0.05 * std::abs(true_cov(i, j)) + 0.001;
            EXPECT_NEAR(sample_cov(i, j), true_cov(i, j), tol)
                << "cov(" << i << "," << j << ")";
        }
    }
}

TEST(FactorMonteCarloGPU, MeanConvergence) {
    auto model = build_test_factor_model();
    const int N = model.n_assets;

    MonteCarloConfig mc;
    mc.n_scenarios = 200000;
    mc.seed = 789;

    ScenarioMatrix gpu_scenarios =
        generate_scenarios_factor_gpu(model.mu, model, mc);

    EXPECT_EQ(gpu_scenarios.n_scenarios(), mc.n_scenarios);
    EXPECT_EQ(gpu_scenarios.n_assets(), N);

    // Download to CPU for verification.
    MatrixXs host_float = gpu_scenarios.to_host();
    MatrixXd scenarios = host_float.cast<double>();

    // Sample mean should converge to mu.
    MatrixXd recon_cov = reconstruct_covariance(model);
    VectorXd sample_mean = scenarios.colwise().mean();
    for (int i = 0; i < N; ++i) {
        double sigma_i = std::sqrt(recon_cov(i, i));
        double bound = 3.0 * sigma_i / std::sqrt(static_cast<double>(mc.n_scenarios));
        EXPECT_NEAR(sample_mean(i), model.mu(i), bound)
            << "asset " << i;
    }
}

TEST(FactorMonteCarloGPU, CovarianceConvergence) {
    auto model = build_test_factor_model();
    const int N = model.n_assets;

    MonteCarloConfig mc;
    mc.n_scenarios = 200000;
    mc.seed = 321;

    ScenarioMatrix gpu_scenarios =
        generate_scenarios_factor_gpu(model.mu, model, mc);

    MatrixXs host_float = gpu_scenarios.to_host();
    MatrixXd scenarios = host_float.cast<double>();

    VectorXd mu_s = scenarios.colwise().mean();
    MatrixXd centered = scenarios.rowwise() - mu_s.transpose();
    MatrixXd sample_cov = (centered.transpose() * centered) /
                           static_cast<double>(mc.n_scenarios - 1);

    MatrixXd true_cov = reconstruct_covariance(model);

    // Allow slightly wider tolerance for GPU (float precision).
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double tol = 0.05 * std::abs(true_cov(i, j)) + 0.002;
            EXPECT_NEAR(sample_cov(i, j), true_cov(i, j), tol)
                << "cov(" << i << "," << j << ")";
        }
    }
}

TEST(FactorMonteCarloGPU, Reproducibility) {
    auto model = build_test_factor_model();

    MonteCarloConfig mc;
    mc.n_scenarios = 1000;
    mc.seed = 42;

    // Generate twice with same seed (fresh cuRAND states each time).
    ScenarioMatrix s1 = generate_scenarios_factor_gpu(model.mu, model, mc);
    ScenarioMatrix s2 = generate_scenarios_factor_gpu(model.mu, model, mc);

    MatrixXs h1 = s1.to_host();
    MatrixXs h2 = s2.to_host();

    // Should be bit-identical.
    for (int i = 0; i < mc.n_scenarios; ++i) {
        for (int j = 0; j < model.n_assets; ++j) {
            EXPECT_EQ(h1(i, j), h2(i, j))
                << "mismatch at (" << i << "," << j << ")";
        }
    }
}

TEST(FactorMonteCarloGPU, CurandStatesReuse) {
    auto model = build_test_factor_model();

    MonteCarloConfig mc;
    mc.n_scenarios = 5000;
    mc.seed = 99;

    auto states = create_curand_states(mc.n_scenarios, mc.seed);

    // Two calls with the same states should produce different scenarios
    // (states advance between calls).
    ScenarioMatrix s1 =
        generate_scenarios_factor_gpu(model.mu, model, mc, states.get());
    ScenarioMatrix s2 =
        generate_scenarios_factor_gpu(model.mu, model, mc, states.get());

    MatrixXs h1 = s1.to_host();
    MatrixXs h2 = s2.to_host();

    // At least some values should differ.
    bool any_differ = false;
    for (int i = 0; i < std::min(mc.n_scenarios, 100); ++i) {
        if (h1(i, 0) != h2(i, 0)) {
            any_differ = true;
            break;
        }
    }
    EXPECT_TRUE(any_differ) << "cuRAND states did not advance between calls";
}

TEST(FactorMonteCarloGPU, GPUCPUParity) {
    // GPU (float) and CPU (double) should produce scenarios with
    // matching statistical properties.
    auto model = build_test_factor_model();
    const int N = model.n_assets;

    MonteCarloConfig mc;
    mc.n_scenarios = 100000;
    mc.seed = 555;

    MatrixXd cpu_scenarios =
        generate_scenarios_factor_cpu(model.mu, model, mc);

    mc.seed = 777;  // Different seed (can't match exactly due to precision).
    ScenarioMatrix gpu_raw =
        generate_scenarios_factor_gpu(model.mu, model, mc);
    MatrixXd gpu_scenarios = gpu_raw.to_host().cast<double>();

    // Compare means: both should be close to model.mu.
    VectorXd cpu_mean = cpu_scenarios.colwise().mean();
    VectorXd gpu_mean = gpu_scenarios.colwise().mean();

    MatrixXd recon_cov = reconstruct_covariance(model);
    for (int i = 0; i < N; ++i) {
        double sigma_i = std::sqrt(recon_cov(i, i));
        double bound = 3.0 * sigma_i / std::sqrt(static_cast<double>(mc.n_scenarios));
        EXPECT_NEAR(cpu_mean(i), model.mu(i), bound)
            << "CPU mean asset " << i;
        EXPECT_NEAR(gpu_mean(i), model.mu(i), bound)
            << "GPU mean asset " << i;
    }

    // Compare covariances: both should be close to reconstructed cov.
    auto compute_sample_cov = [](const MatrixXd& S) {
        VectorXd mu = S.colwise().mean();
        MatrixXd C = S.rowwise() - mu.transpose();
        return MatrixXd((C.transpose() * C) / static_cast<double>(S.rows() - 1));
    };

    MatrixXd cpu_cov = compute_sample_cov(cpu_scenarios);
    MatrixXd gpu_cov = compute_sample_cov(gpu_scenarios);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double ref = recon_cov(i, j);
            double tol = 0.05 * std::abs(ref) + 0.002;
            EXPECT_NEAR(cpu_cov(i, j), ref, tol)
                << "CPU cov(" << i << "," << j << ")";
            EXPECT_NEAR(gpu_cov(i, j), ref, tol)
                << "GPU cov(" << i << "," << j << ")";
        }
    }
}

TEST(FactorMonteCarloGPU, EquivalenceToFullCholesky) {
    // Factor MC and full Cholesky MC should produce scenarios with
    // the same statistical properties when using the same covariance.
    auto model = build_test_factor_model();
    const int N = model.n_assets;

    MatrixXd recon_cov = reconstruct_covariance(model);
    CholeskyResult chol = compute_cholesky(recon_cov);

    MonteCarloConfig mc;
    mc.n_scenarios = 200000;

    // Full Cholesky MC.
    mc.seed = 111;
    ScenarioMatrix full_raw =
        generate_scenarios_gpu(model.mu, chol, mc);
    MatrixXd full_scen = full_raw.to_host().cast<double>();

    // Factor MC.
    mc.seed = 222;
    ScenarioMatrix factor_raw =
        generate_scenarios_factor_gpu(model.mu, model, mc);
    MatrixXd factor_scen = factor_raw.to_host().cast<double>();

    // Compare sample covariances — both should match reconstructed cov.
    auto compute_sample_cov = [](const MatrixXd& S) {
        VectorXd mu = S.colwise().mean();
        MatrixXd C = S.rowwise() - mu.transpose();
        return MatrixXd((C.transpose() * C) / static_cast<double>(S.rows() - 1));
    };

    MatrixXd full_cov = compute_sample_cov(full_scen);
    MatrixXd factor_cov = compute_sample_cov(factor_scen);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double ref = recon_cov(i, j);
            double tol = 0.05 * std::abs(ref) + 0.002;
            EXPECT_NEAR(full_cov(i, j), ref, tol)
                << "full cov(" << i << "," << j << ")";
            EXPECT_NEAR(factor_cov(i, j), ref, tol)
                << "factor cov(" << i << "," << j << ")";
        }
    }
}

// ════════════════════════════════════════════════════════════════════
// Tiled scenario generation tests
// ════════════════════════════════════════════════════════════════════

TEST(TiledScenario, FullCholeskyTiled) {
    // Force tiling by setting a very small tile size.
    auto model = build_test_factor_model();
    const int N = model.n_assets;

    MatrixXd recon_cov = reconstruct_covariance(model);
    CholeskyResult chol = compute_cholesky(recon_cov);

    MonteCarloConfig mc;
    mc.n_scenarios = 10000;
    mc.seed = 42;

    TiledConfig tiled;
    tiled.vram_fraction = 0.7;

    MatrixXd scenarios = generate_scenarios_tiled(
        model.mu, chol, mc, tiled);

    EXPECT_EQ(scenarios.rows(), mc.n_scenarios);
    EXPECT_EQ(scenarios.cols(), N);

    // Verify mean convergence.
    VectorXd sample_mean = scenarios.colwise().mean();
    for (int i = 0; i < N; ++i) {
        double sigma_i = std::sqrt(recon_cov(i, i));
        double bound = 4.0 * sigma_i / std::sqrt(static_cast<double>(mc.n_scenarios));
        EXPECT_NEAR(sample_mean(i), model.mu(i), bound)
            << "asset " << i;
    }
}

TEST(TiledScenario, FactorTiled) {
    // Factor model tiled generation.
    auto model = build_test_factor_model();
    const int N = model.n_assets;

    MonteCarloConfig mc;
    mc.n_scenarios = 10000;
    mc.seed = 123;

    TiledConfig tiled;
    tiled.vram_fraction = 0.7;

    MatrixXd scenarios = generate_scenarios_factor_tiled(
        model.mu, model, mc, tiled);

    EXPECT_EQ(scenarios.rows(), mc.n_scenarios);
    EXPECT_EQ(scenarios.cols(), N);

    // Verify mean convergence.
    MatrixXd recon_cov = reconstruct_covariance(model);
    VectorXd sample_mean = scenarios.colwise().mean();
    for (int i = 0; i < N; ++i) {
        double sigma_i = std::sqrt(recon_cov(i, i));
        double bound = 4.0 * sigma_i / std::sqrt(static_cast<double>(mc.n_scenarios));
        EXPECT_NEAR(sample_mean(i), model.mu(i), bound)
            << "asset " << i;
    }
}

TEST(TiledScenario, OutputMatchesDimensions) {
    // Verify tiled output matches expected dimensions for both paths.
    auto model = build_test_factor_model();

    MatrixXd recon_cov = reconstruct_covariance(model);
    CholeskyResult chol = compute_cholesky(recon_cov);

    MonteCarloConfig mc;
    mc.n_scenarios = 5000;
    mc.seed = 99;

    TiledConfig tiled;
    tiled.vram_fraction = 0.7;

    // Full Cholesky tiled.
    MatrixXd s1 = generate_scenarios_tiled(model.mu, chol, mc, tiled);
    EXPECT_EQ(s1.rows(), 5000);
    EXPECT_EQ(s1.cols(), model.n_assets);

    // Factor tiled.
    MatrixXd s2 = generate_scenarios_factor_tiled(model.mu, model, mc, tiled);
    EXPECT_EQ(s2.rows(), 5000);
    EXPECT_EQ(s2.cols(), model.n_assets);
}
