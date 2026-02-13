#include <filesystem>
#include <string>
#include <vector>

#include <spdlog/spdlog.h>

#include "backtest/backtest_config.h"
#include "backtest/backtest_engine.h"
#include "backtest/strategy.h"
#include "data/csv_loader.h"
#include "data/returns.h"
#include "reporting/report_writer.h"
#include "simulation/monte_carlo.h"
#include "utils/cuda_utils.h"

namespace {

void print_usage() {
    spdlog::info("Usage: backtest --config <path> [--output <dir>]");
}

void print_summary(const cpo::BacktestSummary& s) {
    spdlog::info("─── {} ───", s.strategy_name);
    spdlog::info("  Total return:       {:.4f} ({:.2f}%)", s.total_return, s.total_return * 100);
    spdlog::info("  Annualized return:  {:.4f} ({:.2f}%)", s.annualized_return, s.annualized_return * 100);
    spdlog::info("  Annualized vol:     {:.4f} ({:.2f}%)", s.annualized_volatility, s.annualized_volatility * 100);
    spdlog::info("  Sharpe ratio:       {:.4f}", s.sharpe_ratio);
    spdlog::info("  Sortino ratio:      {:.4f}", s.sortino_ratio);
    spdlog::info("  Max drawdown:       {:.4f} ({:.2f}%)", s.max_drawdown, s.max_drawdown * 100);
    spdlog::info("  Calmar ratio:       {:.4f}", s.calmar_ratio);
    spdlog::info("  Transaction costs:  {:.2f}", s.total_transaction_cost);
    spdlog::info("  Avg turnover:       {:.4f}", s.avg_turnover);
    spdlog::info("  Rebalances:         {}", s.n_rebalances);
    spdlog::info("  Trading days:       {}", s.n_days);
}

}  // anonymous namespace

int main(int argc, char* argv[]) {
    spdlog::info("cuda-portfolio-optimizer: backtest");
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
    cpo::BacktestConfig cfg;
    try {
        cfg = cpo::load_backtest_config(config_path);
    } catch (const std::exception& e) {
        spdlog::error("Failed to load config: {}", e.what());
        return 1;
    }

    // Override output dir if specified on command line.
    if (!output_dir.empty()) {
        cfg.output_dir = output_dir;
    }

    // GPU query.
    if (cfg.use_gpu) {
        cpo::device_query();
    }

    // Load price data.
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

    // Compute returns.
    spdlog::info("Computing returns ({} dates, {} assets)",
                 prices.num_dates(), prices.num_assets());
    cpo::ReturnData returns = cpo::compute_returns(prices, cfg.return_type);
    spdlog::info("Return matrix: {} periods x {} assets",
                 returns.num_periods(), returns.num_assets());

    // Determine strategies to run.
    std::vector<std::string> strategy_names;
    if (cfg.strategy_name == "all") {
        strategy_names = {"EqualWeight", "RiskParity", "MeanVariance", "MeanCVaR"};
    } else {
        strategy_names = {cfg.strategy_name};
    }

    // Create cuRAND states if GPU is enabled (shared across all MeanCVaR runs).
    cpo::CurandStatesGuard curand_guard;
    if (cfg.use_gpu) {
        spdlog::info("Initializing cuRAND states ({} scenarios)",
                     cfg.mc_config.n_scenarios);
        curand_guard = cpo::create_curand_states(
            cfg.mc_config.n_scenarios, cfg.mc_config.seed);
    }

    // Run backtests.
    std::vector<cpo::BacktestResult> all_results;
    for (const auto& sname : strategy_names) {
        spdlog::info("Running backtest: {}", sname);

        std::unique_ptr<cpo::Strategy> strategy;
        if (sname == "MeanCVaR") {
            cpo::MeanCVaRConfig mcfg;
            mcfg.admm_config = cfg.admm_config;
            mcfg.mc_config = cfg.mc_config;
            mcfg.use_gpu = cfg.use_gpu;
            mcfg.curand_states = curand_guard.get();
            mcfg.use_factor_model = cfg.use_factor_model;
            mcfg.factor_config = cfg.factor_config;
            mcfg.use_factor_mc = cfg.use_factor_mc;
            mcfg.use_ledoit_wolf = cfg.use_ledoit_wolf;
            strategy = std::make_unique<cpo::MeanCVaRStrategy>(mcfg);
        } else if (sname == "MeanVariance") {
            cpo::MeanVarianceConfig mvcfg;
            mvcfg.shrinkage_intensity = cfg.shrinkage_intensity;
            mvcfg.target_return = cfg.mv_target_return;
            mvcfg.has_target_return = cfg.mv_has_target_return;
            mvcfg.use_ledoit_wolf = cfg.use_ledoit_wolf;
            strategy = std::make_unique<cpo::MeanVarianceStrategy>(mvcfg);
        } else {
            strategy = cpo::create_strategy(sname);
        }

        try {
            auto result = cpo::run_backtest(returns, *strategy, cfg);
            print_summary(result.summary);
            all_results.push_back(std::move(result));
        } catch (const std::exception& e) {
            spdlog::error("Backtest failed for {}: {}", sname, e.what());
        }
    }

    // Write reports.
    if (!cfg.output_dir.empty() && !all_results.empty()) {
        std::filesystem::create_directories(cfg.output_dir);

        for (const auto& res : all_results) {
            std::string prefix = cfg.output_dir + "/" + res.summary.strategy_name;
            cpo::write_equity_curve_csv(res, prefix + "_equity.csv");
            cpo::write_weights_csv(res, prefix + "_weights.csv");
            cpo::write_summary_json(res.summary, prefix + "_summary.json");
            spdlog::info("Wrote reports for {} to {}", res.summary.strategy_name, cfg.output_dir);
        }

        if (all_results.size() > 1) {
            cpo::write_comparison_json(all_results, cfg.output_dir + "/comparison.json");
            cpo::write_comparison_csv(all_results, cfg.output_dir + "/comparison.csv");
            spdlog::info("Wrote comparison reports");
        }
    }

    spdlog::info("Backtest complete.");
    return 0;
}
