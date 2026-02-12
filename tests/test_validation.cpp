#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "constraints/constraint_set.h"
#include "core/types.h"
#include "optimizer/admm_solver.h"
#include "simulation/cholesky_utils.h"
#include "simulation/monte_carlo.h"

using namespace cpo;
namespace fs = std::filesystem;

// ── Helper: load a validation JSON reference file ─────────────────

struct ValidationCase {
    std::string name;
    int n_assets;
    int n_scenarios;
    double alpha;
    uint64_t seed;
    VectorXd mu;
    MatrixXd covariance;

    // Constraints.
    bool has_box = false;
    double w_max = 1.0;
    bool has_target_return = false;
    double target_return = 0.0;

    // cvxpy reference result.
    VectorXd ref_weights;
    double ref_cvar = 0.0;
    double ref_expected_return = 0.0;
    std::string ref_status;
};

static bool load_validation_case(const std::string& path, ValidationCase& vc) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) return false;

    nlohmann::json j;
    try {
        j = nlohmann::json::parse(ifs);
    } catch (...) {
        return false;
    }

    vc.name = j.at("name").get<std::string>();
    vc.n_assets = j.at("n_assets").get<int>();
    vc.n_scenarios = j.at("n_scenarios").get<int>();
    vc.alpha = j.at("alpha").get<double>();
    vc.seed = j.at("seed").get<uint64_t>();

    // mu vector.
    auto mu_arr = j.at("mu").get<std::vector<double>>();
    vc.mu = Eigen::Map<VectorXd>(mu_arr.data(), mu_arr.size());

    // Covariance matrix.
    auto cov_arr = j.at("covariance").get<std::vector<std::vector<double>>>();
    vc.covariance.resize(vc.n_assets, vc.n_assets);
    for (int i = 0; i < vc.n_assets; ++i) {
        for (int k = 0; k < vc.n_assets; ++k) {
            vc.covariance(i, k) = cov_arr[i][k];
        }
    }

    // Constraints.
    if (j.contains("constraints")) {
        const auto& c = j["constraints"];
        if (c.contains("w_max")) {
            vc.has_box = true;
            vc.w_max = c["w_max"].get<double>();
        }
        if (c.contains("target_return")) {
            vc.has_target_return = true;
            vc.target_return = c["target_return"].get<double>();
        }
    }

    // cvxpy result.
    const auto& res = j.at("cvxpy_result");
    auto w_arr = res.at("weights").get<std::vector<double>>();
    vc.ref_weights = Eigen::Map<VectorXd>(w_arr.data(), w_arr.size());
    vc.ref_cvar = res.at("cvar").get<double>();
    vc.ref_expected_return = res.at("expected_return").get<double>();
    vc.ref_status = res.at("status").get<std::string>();

    return true;
}

// ── Collect all validation JSON files ─────────────────────────────

static std::vector<std::string> find_validation_files() {
    std::vector<std::string> files;
#ifdef TEST_DATA_DIR
    fs::path dir = fs::path(TEST_DATA_DIR) / "validation";
#else
    // Fallback: look relative to source.
    fs::path dir = fs::path(__FILE__).parent_path() / "data" / "validation";
#endif
    if (!fs::exists(dir) || !fs::is_directory(dir)) return files;

    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.path().extension() == ".json") {
            files.push_back(entry.path().string());
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

// ── Parameterized test fixture ────────────────────────────────────

class ValidationTest : public ::testing::TestWithParam<std::string> {};

TEST_P(ValidationTest, MatchesCvxpy) {
    ValidationCase vc;
    bool loaded = load_validation_case(GetParam(), vc);
    if (!loaded) {
        GTEST_SKIP() << "Could not load " << GetParam();
    }
    if (vc.ref_status != "optimal" && vc.ref_status != "optimal_inaccurate") {
        GTEST_SKIP() << "cvxpy status=" << vc.ref_status << " for " << vc.name;
    }

    // Generate scenarios from same mu/Sigma.
    // Note: the C++ CPU RNG (std::mt19937) differs from numpy's default_rng,
    // so scenarios will differ. We compare optimization quality, not scenario
    // identity. Both sides have enough scenarios for convergence.
    auto chol = compute_cholesky(vc.covariance);

    MonteCarloConfig mc_cfg;
    mc_cfg.n_scenarios = vc.n_scenarios;
    mc_cfg.seed = vc.seed;
    MatrixXd scenarios = generate_scenarios_cpu(vc.mu, chol, mc_cfg);

    // Configure ADMM.
    AdmmConfig admm_cfg;
    admm_cfg.confidence_level = 1.0 - vc.alpha;  // alpha=0.05 -> confidence=0.95
    admm_cfg.max_iter = 500;
    admm_cfg.abs_tol = 1e-7;
    admm_cfg.rel_tol = 1e-5;

    if (vc.has_box) {
        admm_cfg.constraints.has_position_limits = true;
        admm_cfg.constraints.position_limits.w_min = VectorXd::Zero(vc.n_assets);
        admm_cfg.constraints.position_limits.w_max =
            VectorXd::Constant(vc.n_assets, vc.w_max);
    }
    if (vc.has_target_return) {
        admm_cfg.has_target_return = true;
        admm_cfg.target_return = vc.target_return;
    }

    auto result = admm_solve(scenarios, vc.mu, admm_cfg);

    // ── Tolerances ────────────────────────────────────────────────
    // ADMM is a first-order method; scenarios differ between Python and C++.
    // Use loose but meaningful tolerances.
    double w_tol = (vc.n_assets <= 5) ? 0.05 : 0.10;
    double cvar_rel_tol = 0.10;

    // Weights sum to 1.
    EXPECT_NEAR(result.weights.sum(), 1.0, 1e-3)
        << vc.name << ": weights don't sum to 1";

    // Weights non-negative.
    for (int i = 0; i < vc.n_assets; ++i) {
        EXPECT_GE(result.weights(i), -1e-3)
            << vc.name << ": w(" << i << ") = " << result.weights(i);
    }

    // Box constraints satisfied.
    if (vc.has_box) {
        for (int i = 0; i < vc.n_assets; ++i) {
            EXPECT_LE(result.weights(i), vc.w_max + 1e-3)
                << vc.name << ": w(" << i << ") = " << result.weights(i)
                << " exceeds w_max=" << vc.w_max;
        }
    }

    // Target return satisfied.
    if (vc.has_target_return) {
        EXPECT_GE(result.expected_return, vc.target_return - 1e-3)
            << vc.name << ": expected_return=" << result.expected_return
            << " below target=" << vc.target_return;
    }

    // Weight comparison (L-inf norm).
    double w_diff = (result.weights - vc.ref_weights).lpNorm<Eigen::Infinity>();
    EXPECT_LT(w_diff, w_tol)
        << vc.name << ": weight L-inf diff=" << w_diff
        << " (tol=" << w_tol << ")"
        << "\n  ADMM:  " << result.weights.transpose()
        << "\n  cvxpy: " << vc.ref_weights.transpose();

    // CVaR comparison (relative).
    if (std::abs(vc.ref_cvar) > 1e-8) {
        double cvar_rel = std::abs(result.cvar - vc.ref_cvar) / std::abs(vc.ref_cvar);
        EXPECT_LT(cvar_rel, cvar_rel_tol)
            << vc.name << ": CVaR relative diff=" << cvar_rel
            << " (ADMM=" << result.cvar << ", cvxpy=" << vc.ref_cvar << ")";
    }
}

// Suppress GTest warning when no validation files exist.
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(ValidationTest);

INSTANTIATE_TEST_SUITE_P(
    CvxpyReference,
    ValidationTest,
    ::testing::ValuesIn(find_validation_files()),
    [](const ::testing::TestParamInfo<std::string>& info) {
        // Extract filename without extension for test name.
        fs::path p(info.param);
        return p.stem().string();
    }
);

// ── Fallback test when no validation files exist ──────────────────

TEST(ValidationSelfCheck, SkipIfNoFiles) {
    auto files = find_validation_files();
    if (files.empty()) {
        GTEST_SKIP() << "No validation JSON files found. "
                     << "Run: python scripts/validate_cvxpy.py";
    }
    // If files exist, the parameterized tests above cover them.
    SUCCEED() << "Found " << files.size() << " validation files";
}
