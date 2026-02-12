#include "constraints/constraint_set.h"

#include <cmath>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace cpo {

int ConstraintSet::num_constraint_sets() const {
    int count = 1;  // Simplex is always active.
    if (has_position_limits) ++count;
    if (has_turnover) ++count;
    if (has_sector_constraints) {
        count += static_cast<int>(sector_constraints.sectors.size());
    }
    return count;
}

void ConstraintSet::validate(Index n_assets) const {
    if (has_position_limits) {
        const auto& pl = position_limits;
        if (pl.w_min.size() != n_assets) {
            throw std::runtime_error(
                "ConstraintSet: w_min size (" +
                std::to_string(pl.w_min.size()) +
                ") != n_assets (" + std::to_string(n_assets) + ")");
        }
        if (pl.w_max.size() != n_assets) {
            throw std::runtime_error(
                "ConstraintSet: w_max size (" +
                std::to_string(pl.w_max.size()) +
                ") != n_assets (" + std::to_string(n_assets) + ")");
        }
        for (Index i = 0; i < n_assets; ++i) {
            if (pl.w_min(i) > pl.w_max(i)) {
                throw std::runtime_error(
                    "ConstraintSet: w_min[" + std::to_string(i) +
                    "] (" + std::to_string(pl.w_min(i)) +
                    ") > w_max[" + std::to_string(i) +
                    "] (" + std::to_string(pl.w_max(i)) + ")");
            }
        }
        // Check simplex feasibility: sum(w_min) <= 1 <= sum(w_max).
        double sum_min = pl.w_min.sum();
        double sum_max = pl.w_max.sum();
        if (sum_min > 1.0 + 1e-10) {
            throw std::runtime_error(
                "ConstraintSet: sum(w_min) = " +
                std::to_string(sum_min) + " > 1 (infeasible)");
        }
        if (sum_max < 1.0 - 1e-10) {
            throw std::runtime_error(
                "ConstraintSet: sum(w_max) = " +
                std::to_string(sum_max) + " < 1 (infeasible)");
        }
    }

    if (has_turnover) {
        if (turnover.tau < 0.0) {
            throw std::runtime_error(
                "ConstraintSet: tau must be >= 0 (got " +
                std::to_string(turnover.tau) + ")");
        }
        if (turnover.w_prev.size() != n_assets) {
            throw std::runtime_error(
                "ConstraintSet: w_prev size (" +
                std::to_string(turnover.w_prev.size()) +
                ") != n_assets (" + std::to_string(n_assets) + ")");
        }
    }

    if (has_sector_constraints) {
        for (const auto& sector : sector_constraints.sectors) {
            for (Index idx : sector.assets) {
                if (idx < 0 || idx >= n_assets) {
                    throw std::runtime_error(
                        "ConstraintSet: sector '" + sector.name +
                        "' has invalid asset index " + std::to_string(idx) +
                        " (n_assets=" + std::to_string(n_assets) + ")");
                }
            }
            if (sector.min_exposure > sector.max_exposure) {
                throw std::runtime_error(
                    "ConstraintSet: sector '" + sector.name +
                    "' has min_exposure > max_exposure");
            }
        }
    }
}

bool ConstraintSet::is_feasible(const VectorXd& w, ScalarCPU tol) const {
    const Index n = static_cast<Index>(w.size());

    // Simplex: sum = 1, w >= 0.
    if (std::abs(w.sum() - 1.0) > tol) return false;
    for (Index i = 0; i < n; ++i) {
        if (w(i) < -tol) return false;
    }

    // Position limits.
    if (has_position_limits) {
        const auto& pl = position_limits;
        for (Index i = 0; i < n; ++i) {
            if (w(i) < pl.w_min(i) - tol) return false;
            if (w(i) > pl.w_max(i) + tol) return false;
        }
    }

    // Turnover.
    if (has_turnover) {
        double l1_norm = (w - turnover.w_prev).lpNorm<1>();
        if (l1_norm > turnover.tau + tol) return false;
    }

    // Sector constraints.
    if (has_sector_constraints) {
        for (const auto& sector : sector_constraints.sectors) {
            double sector_sum = 0.0;
            for (Index idx : sector.assets) {
                sector_sum += w(idx);
            }
            if (sector_sum < sector.min_exposure - tol) return false;
            if (sector_sum > sector.max_exposure + tol) return false;
        }
    }

    return true;
}

ConstraintSet parse_constraints(const nlohmann::json& j, Index n_assets) {
    ConstraintSet cs;

    if (j.contains("position_limits")) {
        const auto& pl_json = j["position_limits"];
        cs.has_position_limits = true;

        auto w_min_vec = pl_json.at("w_min").get<std::vector<double>>();
        auto w_max_vec = pl_json.at("w_max").get<std::vector<double>>();

        cs.position_limits.w_min = Eigen::Map<const VectorXd>(
            w_min_vec.data(), static_cast<Index>(w_min_vec.size()));
        cs.position_limits.w_max = Eigen::Map<const VectorXd>(
            w_max_vec.data(), static_cast<Index>(w_max_vec.size()));
    }

    if (j.contains("turnover")) {
        const auto& to_json = j["turnover"];
        cs.has_turnover = true;

        auto w_prev_vec = to_json.at("w_prev").get<std::vector<double>>();
        cs.turnover.w_prev = Eigen::Map<const VectorXd>(
            w_prev_vec.data(), static_cast<Index>(w_prev_vec.size()));

        if (to_json.contains("tau")) {
            cs.turnover.tau = to_json["tau"].get<double>();
        }
    }

    if (j.contains("sectors")) {
        cs.has_sector_constraints = true;
        for (const auto& s_json : j["sectors"]) {
            SectorBound sb;
            sb.name = s_json.at("name").get<std::string>();
            sb.assets = s_json.at("assets").get<std::vector<Index>>();

            if (s_json.contains("min")) {
                sb.min_exposure = s_json["min"].get<double>();
            }
            if (s_json.contains("max")) {
                sb.max_exposure = s_json["max"].get<double>();
            }
            cs.sector_constraints.sectors.push_back(std::move(sb));
        }
    }

    cs.validate(n_assets);
    return cs;
}

}  // namespace cpo
