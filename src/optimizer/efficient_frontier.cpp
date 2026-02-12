#include "optimizer/efficient_frontier.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <spdlog/spdlog.h>

namespace cpo {

std::vector<FrontierPoint> compute_efficient_frontier(
    const MatrixXd& scenarios,
    const VectorXd& mu,
    const FrontierConfig& config) {
    const int n_assets = static_cast<int>(mu.size());

    if (scenarios.cols() != n_assets) {
        throw std::runtime_error(
            "compute_efficient_frontier: scenarios cols (" +
            std::to_string(scenarios.cols()) + ") != mu size (" +
            std::to_string(n_assets) + ")");
    }
    if (config.n_points < 2) {
        throw std::runtime_error(
            "compute_efficient_frontier: n_points must be >= 2");
    }

    // Determine return range.
    double mu_lo = config.mu_min;
    double mu_hi = config.mu_max;

    if (mu_lo < -0.99) {
        mu_lo = mu.minCoeff();
    }
    if (mu_hi < -0.99) {
        mu_hi = mu.maxCoeff();
    }

    if (mu_lo >= mu_hi) {
        // All assets have same expected return — single point.
        mu_lo = mu.minCoeff();
        mu_hi = mu.maxCoeff();
        if (mu_lo >= mu_hi) {
            mu_hi = mu_lo + 1e-6;
        }
    }

    spdlog::info("Efficient frontier: {} points, mu=[{:.6f}, {:.6f}]",
                 config.n_points, mu_lo, mu_hi);

    std::vector<FrontierPoint> frontier;
    frontier.reserve(config.n_points);

    VectorXd w_prev;  // For warm starting.

    for (int i = 0; i < config.n_points; ++i) {
        double target = mu_lo + (mu_hi - mu_lo) * i / (config.n_points - 1);

        // Configure ADMM for this target return.
        AdmmConfig admm_cfg = config.admm_config;
        admm_cfg.target_return = target;
        admm_cfg.has_target_return = true;

        // Warm start from previous solution.
        VectorXd w_init;
        if (config.warm_start && w_prev.size() == n_assets) {
            w_init = w_prev;
        }

        spdlog::debug("Frontier point {}/{}: target_return={:.6f}",
                      i + 1, config.n_points, target);

        auto result = admm_solve(scenarios, mu, admm_cfg, w_init);

        FrontierPoint point;
        point.target_return = target;
        point.achieved_return = result.expected_return;
        point.cvar = result.cvar;
        point.zeta = result.zeta;
        point.weights = result.weights;
        point.iterations = result.iterations;
        point.converged = result.converged;
        frontier.push_back(point);

        w_prev = result.weights;
    }

    spdlog::info("Efficient frontier complete: {} points computed",
                 frontier.size());

    return frontier;
}

std::vector<FrontierPoint> compute_efficient_frontier(
    const ScenarioMatrix& scenarios_gpu,
    const VectorXd& mu,
    const FrontierConfig& config) {
    const int n_assets = static_cast<int>(mu.size());

    if (scenarios_gpu.n_assets() != n_assets) {
        throw std::runtime_error(
            "compute_efficient_frontier(GPU): scenarios n_assets (" +
            std::to_string(scenarios_gpu.n_assets()) + ") != mu size (" +
            std::to_string(n_assets) + ")");
    }
    if (config.n_points < 2) {
        throw std::runtime_error(
            "compute_efficient_frontier: n_points must be >= 2");
    }

    // Determine return range.
    double mu_lo = config.mu_min;
    double mu_hi = config.mu_max;

    if (mu_lo < -0.99) {
        mu_lo = mu.minCoeff();
    }
    if (mu_hi < -0.99) {
        mu_hi = mu.maxCoeff();
    }

    if (mu_lo >= mu_hi) {
        mu_lo = mu.minCoeff();
        mu_hi = mu.maxCoeff();
        if (mu_lo >= mu_hi) {
            mu_hi = mu_lo + 1e-6;
        }
    }

    spdlog::info("Efficient frontier (GPU): {} points, mu=[{:.6f}, {:.6f}]",
                 config.n_points, mu_lo, mu_hi);

    std::vector<FrontierPoint> frontier;
    frontier.reserve(config.n_points);

    VectorXd w_prev;

    for (int i = 0; i < config.n_points; ++i) {
        double target = mu_lo + (mu_hi - mu_lo) * i / (config.n_points - 1);

        AdmmConfig admm_cfg = config.admm_config;
        admm_cfg.target_return = target;
        admm_cfg.has_target_return = true;

        VectorXd w_init;
        if (config.warm_start && w_prev.size() == n_assets) {
            w_init = w_prev;
        }

        spdlog::debug("Frontier point {}/{}: target_return={:.6f}",
                      i + 1, config.n_points, target);

        auto result = admm_solve(scenarios_gpu, mu, admm_cfg, w_init);

        FrontierPoint point;
        point.target_return = target;
        point.achieved_return = result.expected_return;
        point.cvar = result.cvar;
        point.zeta = result.zeta;
        point.weights = result.weights;
        point.iterations = result.iterations;
        point.converged = result.converged;
        frontier.push_back(point);

        w_prev = result.weights;
    }

    spdlog::info("Efficient frontier (GPU) complete: {} points computed",
                 frontier.size());

    return frontier;
}

}  // namespace cpo
