#include "optimizer/optimize_config.h"

#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace cpo {

OptimizeConfig load_optimize_config(const std::string& json_path) {
    std::ifstream ifs(json_path);
    if (!ifs.is_open()) {
        throw std::runtime_error(
            "load_optimize_config: cannot open " + json_path);
    }

    nlohmann::json j = nlohmann::json::parse(ifs);
    OptimizeConfig cfg;

    // Data source: CSV or direct mu/covariance.
    if (j.contains("price_csv_path")) {
        cfg.price_csv_path = j["price_csv_path"].get<std::string>();
    }
    if (j.contains("tickers")) {
        cfg.tickers = j["tickers"].get<std::vector<std::string>>();
    }
    if (j.contains("start_date")) {
        cfg.start_date = j["start_date"].get<std::string>();
    }
    if (j.contains("end_date")) {
        cfg.end_date = j["end_date"].get<std::string>();
    }

    // Direct mu/covariance.
    if (j.contains("mu")) {
        cfg.mu_values = j["mu"].get<std::vector<double>>();
    }
    if (j.contains("covariance")) {
        cfg.cov_values = j["covariance"].get<std::vector<std::vector<double>>>();
    }

    // Covariance estimation method.
    if (j.contains("covariance_method")) {
        cfg.covariance_method = j["covariance_method"].get<std::string>();
    }
    if (j.contains("factor_model")) {
        const auto& fm = j["factor_model"];
        if (fm.contains("n_factors"))
            cfg.factor_config.n_factors = fm["n_factors"].get<int>();
        if (fm.contains("min_variance_explained"))
            cfg.factor_config.min_variance_explained = fm["min_variance_explained"].get<double>();
    }
    if (j.contains("use_factor_mc")) {
        cfg.use_factor_mc = j["use_factor_mc"].get<bool>();
    }

    // Mode.
    if (j.contains("frontier_mode")) {
        cfg.frontier_mode = j["frontier_mode"].get<bool>();
    }

    // GPU.
    if (j.contains("use_gpu")) {
        cfg.use_gpu = j["use_gpu"].get<bool>();
    }

    // Monte Carlo config.
    if (j.contains("monte_carlo")) {
        const auto& mc = j["monte_carlo"];
        if (mc.contains("n_scenarios"))
            cfg.mc_config.n_scenarios = mc["n_scenarios"].get<Index>();
        if (mc.contains("seed"))
            cfg.mc_config.seed = mc["seed"].get<uint64_t>();
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

    // Constraint parsing.
    if (j.contains("constraints")) {
        const auto& c = j["constraints"];
        if (c.contains("w_max")) {
            int n = 0;
            if (!cfg.mu_values.empty()) {
                n = static_cast<int>(cfg.mu_values.size());
            }
            if (n > 0) {
                cfg.admm_config.constraints.has_position_limits = true;
                cfg.admm_config.constraints.position_limits.w_min =
                    VectorXd::Zero(n);
                cfg.admm_config.constraints.position_limits.w_max =
                    VectorXd::Constant(n, c["w_max"].get<double>());
            }
        }
    }

    // Frontier config.
    if (j.contains("frontier")) {
        const auto& f = j["frontier"];
        if (f.contains("n_points"))
            cfg.frontier_config.n_points = f["n_points"].get<int>();
        if (f.contains("mu_min"))
            cfg.frontier_config.mu_min = f["mu_min"].get<double>();
        if (f.contains("mu_max"))
            cfg.frontier_config.mu_max = f["mu_max"].get<double>();
        if (f.contains("warm_start"))
            cfg.frontier_config.warm_start = f["warm_start"].get<bool>();
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
