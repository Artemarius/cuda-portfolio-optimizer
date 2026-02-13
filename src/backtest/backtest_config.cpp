#include "backtest/backtest_config.h"

#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace cpo {

BacktestConfig load_backtest_config(const std::string& json_path) {
    std::ifstream ifs(json_path);
    if (!ifs.is_open()) {
        throw std::runtime_error(
            "load_backtest_config: cannot open " + json_path);
    }

    nlohmann::json j = nlohmann::json::parse(ifs);
    BacktestConfig cfg;

    // Required fields.
    cfg.price_csv_path = j.at("price_csv_path").get<std::string>();

    // Optional data fields.
    if (j.contains("tickers")) {
        cfg.tickers = j["tickers"].get<std::vector<std::string>>();
    }
    if (j.contains("start_date")) {
        cfg.start_date = j["start_date"].get<std::string>();
    }
    if (j.contains("end_date")) {
        cfg.end_date = j["end_date"].get<std::string>();
    }
    if (j.contains("return_type")) {
        std::string rt = j["return_type"].get<std::string>();
        cfg.return_type = (rt == "log") ? ReturnType::kLog : ReturnType::kSimple;
    }

    // Rolling window settings.
    if (j.contains("lookback_window")) {
        cfg.lookback_window = j["lookback_window"].get<Index>();
    }
    if (j.contains("rebalance_frequency")) {
        cfg.rebalance_frequency = j["rebalance_frequency"].get<Index>();
    }

    // Strategy.
    if (j.contains("strategy_name")) {
        cfg.strategy_name = j["strategy_name"].get<std::string>();
    }
    if (j.contains("initial_capital")) {
        cfg.initial_capital = j["initial_capital"].get<double>();
    }

    // GPU.
    if (j.contains("use_gpu")) {
        cfg.use_gpu = j["use_gpu"].get<bool>();
    }

    // ADMM config.
    if (j.contains("admm")) {
        const auto& a = j["admm"];
        if (a.contains("confidence_level"))
            cfg.admm_config.confidence_level = a["confidence_level"].get<double>();
        if (a.contains("target_return")) {
            cfg.admm_config.target_return = a["target_return"].get<double>();
            cfg.admm_config.has_target_return = true;
        }
        if (a.contains("rho"))
            cfg.admm_config.rho = a["rho"].get<double>();
        if (a.contains("max_iter"))
            cfg.admm_config.max_iter = a["max_iter"].get<int>();
        if (a.contains("abs_tol"))
            cfg.admm_config.abs_tol = a["abs_tol"].get<double>();
        if (a.contains("rel_tol"))
            cfg.admm_config.rel_tol = a["rel_tol"].get<double>();
        if (a.contains("verbose"))
            cfg.admm_config.verbose = a["verbose"].get<bool>();
    }

    // Monte Carlo config.
    if (j.contains("monte_carlo")) {
        const auto& mc = j["monte_carlo"];
        if (mc.contains("n_scenarios"))
            cfg.mc_config.n_scenarios = mc["n_scenarios"].get<Index>();
        if (mc.contains("seed"))
            cfg.mc_config.seed = mc["seed"].get<uint64_t>();
    }

    // Factor model settings.
    if (j.contains("factor_model")) {
        cfg.use_factor_model = true;
        const auto& fm = j["factor_model"];
        if (fm.contains("n_factors"))
            cfg.factor_config.n_factors = fm["n_factors"].get<int>();
        if (fm.contains("min_variance_explained"))
            cfg.factor_config.min_variance_explained = fm["min_variance_explained"].get<double>();
    }
    if (j.contains("use_factor_mc")) {
        cfg.use_factor_mc = j["use_factor_mc"].get<bool>();
    }

    // Mean-variance settings.
    if (j.contains("mean_variance")) {
        const auto& mv = j["mean_variance"];
        if (mv.contains("shrinkage_intensity"))
            cfg.shrinkage_intensity = mv["shrinkage_intensity"].get<double>();
        if (mv.contains("target_return")) {
            cfg.mv_target_return = mv["target_return"].get<double>();
            cfg.mv_has_target_return = true;
        }
        if (mv.contains("use_ledoit_wolf"))
            cfg.use_ledoit_wolf = mv["use_ledoit_wolf"].get<bool>();
    }

    // Top-level use_ledoit_wolf (applies to all strategies that use sample cov).
    if (j.contains("use_ledoit_wolf")) {
        cfg.use_ledoit_wolf = j["use_ledoit_wolf"].get<bool>();
    }

    // Transaction costs.
    if (j.contains("transaction_costs")) {
        const auto& tc = j["transaction_costs"];
        if (tc.contains("cost_rate"))
            cfg.transaction_costs.cost_rate = tc["cost_rate"].get<double>();
        if (tc.contains("min_trade_threshold"))
            cfg.transaction_costs.min_trade_threshold = tc["min_trade_threshold"].get<double>();
    }

    // Output.
    if (j.contains("output_dir")) {
        cfg.output_dir = j["output_dir"].get<std::string>();
    }
    if (j.contains("verbose")) {
        cfg.verbose = j["verbose"].get<bool>();
    }

    return cfg;
}

}  // namespace cpo
