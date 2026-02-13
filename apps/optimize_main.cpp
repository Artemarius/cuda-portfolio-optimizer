#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <spdlog/spdlog.h>

#include "data/csv_loader.h"
#include "data/returns.h"
#include "models/factor_model.h"
#include "models/factor_monte_carlo.h"
#include "models/shrinkage_estimator.h"
#include "optimizer/admm_solver.h"
#include "optimizer/efficient_frontier.h"
#include "optimizer/optimize_config.h"
#include "reporting/report_writer.h"
#include "risk/component_cvar.h"
#include "simulation/cholesky_utils.h"
#include "simulation/monte_carlo.h"
#include "utils/cuda_utils.h"

namespace {

void print_usage() {
    spdlog::info("Usage: optimize --config <path> [--output <dir>]");
    spdlog::info("  --config <path>  JSON configuration file");
    spdlog::info("  --output <dir>   Output directory (overrides config)");
}

void print_result(const cpo::AdmmResult& result,
                  const cpo::VectorXd& mu,
                  const std::vector<std::string>& tickers) {
    spdlog::info("─── Optimization Result ───");
    spdlog::info("  Converged:        {}", result.converged ? "yes" : "no");
    spdlog::info("  Iterations:       {}", result.iterations);
    spdlog::info("  CVaR:             {:.6f}", result.cvar);
    spdlog::info("  Expected return:  {:.6f} ({:.2f}%)",
                 result.expected_return, result.expected_return * 100);
    spdlog::info("  VaR (zeta):       {:.6f}", result.zeta);
    spdlog::info("  Weights:");
    for (size_t i = 0; i < tickers.size(); ++i) {
        spdlog::info("    {:>8s}: {:.4f}", tickers[i],
                     result.weights(static_cast<cpo::Index>(i)));
    }
}

void print_frontier(const std::vector<cpo::FrontierPoint>& frontier) {
    spdlog::info("─── Efficient Frontier ({} points) ───", frontier.size());
    spdlog::info("  {:>12s} {:>12s} {:>12s} {:>6s} {:>5s}",
                 "Target Ret", "Achieved Ret", "CVaR", "Iters", "Conv");
    for (const auto& pt : frontier) {
        spdlog::info("  {:12.6f} {:12.6f} {:12.6f} {:6d} {:>5s}",
                     pt.target_return, pt.achieved_return, pt.cvar,
                     pt.iterations, pt.converged ? "yes" : "no");
    }
}

}  // anonymous namespace

int main(int argc, char* argv[]) {
    spdlog::info("cuda-portfolio-optimizer: optimize");
    spdlog::info("==================================");

    // Parse command-line arguments.
    std::string config_path;
    std::string output_dir;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            print_usage();
            return 0;
        }
    }

    if (config_path.empty()) {
        spdlog::error("Missing --config argument");
        print_usage();
        return 1;
    }

    // Load configuration.
    spdlog::info("Loading config from {}", config_path);
    cpo::OptimizeConfig cfg;
    try {
        cfg = cpo::load_optimize_config(config_path);
    } catch (const std::exception& e) {
        spdlog::error("Failed to load config: {}", e.what());
        return 1;
    }

    if (!output_dir.empty()) {
        cfg.output_dir = output_dir;
    }

    // GPU query.
    if (cfg.use_gpu) {
        cpo::device_query();
    }

    // Determine mu and covariance.
    cpo::VectorXd mu;
    cpo::MatrixXd cov;
    std::vector<std::string> tickers;

    // Optionally holds the fitted factor model (used by factor MC path).
    std::optional<cpo::FactorModelResult> factor_model;

    if (!cfg.price_csv_path.empty()) {
        // Load from CSV.
        spdlog::info("Loading prices from {}", cfg.price_csv_path);
        cpo::PriceData prices;
        try {
            if (cfg.tickers.empty()) {
                prices = cpo::load_csv_prices(cfg.price_csv_path);
            } else {
                prices = cpo::load_csv_prices(cfg.price_csv_path, cfg.tickers);
            }
        } catch (const std::exception& e) {
            spdlog::error("Failed to load prices: {}", e.what());
            return 1;
        }

        cpo::ReturnData returns = cpo::compute_returns(prices, cpo::ReturnType::kSimple);
        spdlog::info("Return matrix: {} periods x {} assets",
                     returns.num_periods(), returns.num_assets());
        tickers = returns.tickers;

        if (cfg.covariance_method == "factor") {
            // Factor model covariance estimation.
            spdlog::info("Fitting factor model (k={})...",
                         cfg.factor_config.n_factors);
            try {
                factor_model = cpo::fit_factor_model(
                    returns.returns, cfg.factor_config);
            } catch (const std::exception& e) {
                spdlog::error("Factor model failed: {}", e.what());
                return 1;
            }
            mu = factor_model->mu;
            cov = cpo::reconstruct_covariance(*factor_model);
            spdlog::info("Factor model: k={}, variance explained={:.1f}%",
                         factor_model->n_factors,
                         factor_model->variance_explained * 100.0);
        } else if (cfg.use_ledoit_wolf) {
            // Ledoit-Wolf optimal shrinkage covariance estimation.
            spdlog::info("Estimating covariance with Ledoit-Wolf shrinkage...");
            auto lw = cpo::ledoit_wolf_shrink(returns.returns);
            mu = returns.returns.colwise().mean().transpose();
            cov = std::move(lw.covariance);
            spdlog::info("Ledoit-Wolf shrinkage intensity: {:.4f}", lw.intensity);
        } else {
            // Sample covariance estimation (default).
            mu = returns.returns.colwise().mean().transpose();
            cpo::MatrixXd centered = returns.returns.rowwise() - mu.transpose();
            cov = (centered.transpose() * centered) /
                  static_cast<double>(returns.num_periods() - 1);
        }
    } else if (!cfg.mu_values.empty() && !cfg.cov_values.empty()) {
        // Use directly specified mu/covariance.
        int n = static_cast<int>(cfg.mu_values.size());
        mu = Eigen::Map<cpo::VectorXd>(cfg.mu_values.data(), n);
        cov.resize(n, n);
        for (int i = 0; i < n; ++i) {
            for (int k = 0; k < n; ++k) {
                cov(i, k) = cfg.cov_values[i][k];
            }
        }
        // Generate generic ticker names.
        for (int i = 0; i < n; ++i) {
            tickers.push_back("Asset" + std::to_string(i + 1));
        }
    } else {
        spdlog::error("Config must specify either price_csv_path or mu + covariance");
        return 1;
    }

    int n_assets = static_cast<int>(mu.size());

    // Construct position limit vectors from deferred scalar (fixes CSV-based configs).
    if (cfg.w_max_scalar > 0.0) {
        cfg.admm_config.constraints.has_position_limits = true;
        cfg.admm_config.constraints.position_limits.w_min =
            cpo::VectorXd::Zero(n_assets);
        cfg.admm_config.constraints.position_limits.w_max =
            cpo::VectorXd::Constant(n_assets, cfg.w_max_scalar);
    }

    spdlog::info("Optimizing {} assets, {} scenarios",
                 n_assets, cfg.mc_config.n_scenarios);

    // Generate scenarios.
    // When use_gpu: keep ScenarioMatrix alive for GPU ADMM path.
    std::optional<cpo::ScenarioMatrix> gpu_scenarios_holder;
    cpo::MatrixXd scenarios;
    if (cfg.use_factor_mc && factor_model.has_value()) {
        // Factor Monte Carlo path (optimized: O(Nk) vs O(N^2) per scenario).
        if (cfg.use_gpu) {
            spdlog::info("Generating factor MC scenarios on GPU...");
            auto curand_states = cpo::create_curand_states(
                cfg.mc_config.n_scenarios, cfg.mc_config.seed);
            gpu_scenarios_holder.emplace(cpo::generate_scenarios_factor_gpu(
                mu, *factor_model, cfg.mc_config, curand_states.get()));
        } else {
            spdlog::info("Generating factor MC scenarios on CPU...");
            scenarios = cpo::generate_scenarios_factor_cpu(
                mu, *factor_model, cfg.mc_config);
        }
    } else {
        // Full Cholesky Monte Carlo path (default).
        cpo::CholeskyResult chol;
        try {
            chol = cpo::compute_cholesky(cov);
        } catch (const std::exception& e) {
            spdlog::error("Cholesky failed: {}", e.what());
            return 1;
        }
        if (cfg.use_gpu) {
            spdlog::info("Generating scenarios on GPU...");
            auto curand_states = cpo::create_curand_states(
                cfg.mc_config.n_scenarios, cfg.mc_config.seed);
            gpu_scenarios_holder.emplace(cpo::generate_scenarios_gpu(
                mu, chol, cfg.mc_config, curand_states.get()));
        } else {
            spdlog::info("Generating scenarios on CPU...");
            scenarios = cpo::generate_scenarios_cpu(mu, chol, cfg.mc_config);
        }
    }

    // For CPU path, log scenario matrix size.
    // For GPU path, download only if needed (CPU solver falls back).
    if (gpu_scenarios_holder.has_value()) {
        spdlog::info("Scenario matrix (GPU): {} x {}",
                     gpu_scenarios_holder->n_scenarios(),
                     gpu_scenarios_holder->n_assets());
    } else {
        spdlog::info("Scenario matrix: {} x {}", scenarios.rows(), scenarios.cols());
    }

    // Create output directory.
    if (!cfg.output_dir.empty()) {
        std::filesystem::create_directories(cfg.output_dir);
    }

    if (cfg.frontier_mode) {
        // ── Efficient frontier mode ───────────────────────────────
        spdlog::info("Computing efficient frontier ({} points)...",
                     cfg.frontier_config.n_points);

        cfg.frontier_config.admm_config = cfg.admm_config;

        std::vector<cpo::FrontierPoint> frontier;
        if (gpu_scenarios_holder.has_value()) {
            frontier = cpo::compute_efficient_frontier(
                *gpu_scenarios_holder, mu, cfg.frontier_config);
        } else {
            frontier = cpo::compute_efficient_frontier(
                scenarios, mu, cfg.frontier_config);
        }

        print_frontier(frontier);

        // Compute component CVaR for the minimum-CVaR point.
        cpo::VectorXd frontier_component_cvar;
        if (!frontier.empty()) {
            const auto& best = frontier.front();
            cpo::RiskConfig risk_cfg;
            risk_cfg.confidence_level = cfg.admm_config.confidence_level;

            if (gpu_scenarios_holder.has_value()) {
                cpo::VectorXs wf = best.weights.cast<float>();
                auto [risk, comp] = cpo::compute_portfolio_risk_decomp_gpu(
                    *gpu_scenarios_holder, wf, risk_cfg);
                frontier_component_cvar = comp;
            } else {
                auto [risk, comp] = cpo::compute_portfolio_risk_decomp_cpu(
                    scenarios, best.weights, risk_cfg);
                frontier_component_cvar = comp;
            }

            spdlog::info("─── Risk Decomposition (min-CVaR point) ───");
            double total = frontier_component_cvar.sum();
            for (size_t i = 0; i < tickers.size(); ++i) {
                cpo::Index idx = static_cast<cpo::Index>(i);
                double pct = (std::abs(total) > 1e-15)
                                 ? frontier_component_cvar(idx) / total * 100.0
                                 : 0.0;
                spdlog::info("  {:>8s}: weight={:.4f}  CVaR_j={:.6f}  ({:.1f}%)",
                             tickers[i], best.weights(idx),
                             frontier_component_cvar(idx), pct);
            }
        }

        if (!cfg.output_dir.empty()) {
            std::string csv_path = cfg.output_dir + "/frontier.csv";
            std::string json_path = cfg.output_dir + "/frontier_result.json";
            cpo::write_frontier_csv(frontier, tickers, csv_path);
            spdlog::info("Wrote frontier CSV to {}", csv_path);

            // Write the minimum-CVaR point as the "result" JSON.
            if (!frontier.empty()) {
                const auto& best = frontier.front();
                cpo::AdmmResult best_result;
                best_result.weights = best.weights;
                best_result.cvar = best.cvar;
                best_result.expected_return = best.achieved_return;
                best_result.zeta = best.zeta;
                best_result.iterations = best.iterations;
                best_result.converged = best.converged;
                cpo::write_optimize_result_json(
                    best_result, mu, tickers, frontier_component_cvar,
                    json_path);
                spdlog::info("Wrote result JSON to {}", json_path);

                std::string decomp_path =
                    cfg.output_dir + "/risk_decomposition.csv";
                cpo::write_risk_decomposition_csv(
                    best.weights, frontier_component_cvar, tickers,
                    decomp_path);
                spdlog::info("Wrote risk decomposition to {}", decomp_path);
            }
        }
    } else {
        // ── Single-point optimization ─────────────────────────────
        spdlog::info("Running ADMM optimization...");

        cpo::AdmmResult result;
        if (gpu_scenarios_holder.has_value()) {
            result = cpo::admm_solve(*gpu_scenarios_holder, mu, cfg.admm_config);
        } else {
            result = cpo::admm_solve(scenarios, mu, cfg.admm_config);
        }
        print_result(result, mu, tickers);

        // Compute component CVaR.
        cpo::VectorXd component_cvar;
        {
            cpo::RiskConfig risk_cfg;
            risk_cfg.confidence_level = cfg.admm_config.confidence_level;

            if (gpu_scenarios_holder.has_value()) {
                cpo::VectorXs wf = result.weights.cast<float>();
                auto [risk, comp] = cpo::compute_portfolio_risk_decomp_gpu(
                    *gpu_scenarios_holder, wf, risk_cfg);
                component_cvar = comp;
            } else {
                auto [risk, comp] = cpo::compute_portfolio_risk_decomp_cpu(
                    scenarios, result.weights, risk_cfg);
                component_cvar = comp;
            }
        }

        spdlog::info("─── Risk Decomposition ───");
        double total = component_cvar.sum();
        for (size_t i = 0; i < tickers.size(); ++i) {
            cpo::Index idx = static_cast<cpo::Index>(i);
            double pct = (std::abs(total) > 1e-15)
                             ? component_cvar(idx) / total * 100.0
                             : 0.0;
            spdlog::info("  {:>8s}: weight={:.4f}  CVaR_j={:.6f}  ({:.1f}%)",
                         tickers[i], result.weights(idx),
                         component_cvar(idx), pct);
        }

        if (!cfg.output_dir.empty()) {
            std::string json_path = cfg.output_dir + "/optimize_result.json";
            cpo::write_optimize_result_json(
                result, mu, tickers, component_cvar, json_path);
            spdlog::info("Wrote result JSON to {}", json_path);

            std::string decomp_path =
                cfg.output_dir + "/risk_decomposition.csv";
            cpo::write_risk_decomposition_csv(
                result.weights, component_cvar, tickers, decomp_path);
            spdlog::info("Wrote risk decomposition to {}", decomp_path);
        }
    }

    spdlog::info("Optimization complete.");
    return 0;
}
