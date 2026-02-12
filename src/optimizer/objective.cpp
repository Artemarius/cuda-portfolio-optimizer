#include "optimizer/objective.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace cpo {

ObjectiveResult evaluate_objective_cpu(const MatrixXd& scenarios,
                                        const VectorXd& w,
                                        ScalarCPU zeta,
                                        ScalarCPU alpha) {
    // Rockafellar & Uryasev, J. Risk 2000, Eq. (10):
    //   F(w, zeta) = zeta + (1/(N*alpha)) * sum_i max(0, -r_i'w - zeta)
    //
    // Subgradient w.r.t. w:
    //   dF/dw = -(1/(N*alpha)) * sum_{i: -r_i'w > zeta} r_i
    //
    // Subgradient w.r.t. zeta:
    //   dF/dzeta = 1 - (1/(N*alpha)) * |{i: -r_i'w > zeta}|

    const int n_scenarios = static_cast<int>(scenarios.rows());
    const int n_assets = static_cast<int>(scenarios.cols());

    if (w.size() != n_assets) {
        throw std::runtime_error(
            "evaluate_objective_cpu: w size (" +
            std::to_string(w.size()) + ") != n_assets (" +
            std::to_string(n_assets) + ")");
    }
    if (alpha <= 0.0 || alpha >= 1.0) {
        throw std::runtime_error(
            "evaluate_objective_cpu: alpha must be in (0,1), got " +
            std::to_string(alpha));
    }

    // Compute losses: loss_i = -r_i' w for each scenario.
    VectorXd losses = -(scenarios * w);

    // Accumulate objective and subgradients.
    double sum_excess = 0.0;
    VectorXd grad_w = VectorXd::Zero(n_assets);
    int count_exceeding = 0;

    const double inv_n_alpha = 1.0 / (n_scenarios * alpha);

    for (int i = 0; i < n_scenarios; ++i) {
        double excess = losses(i) - zeta;
        if (excess > 0.0) {
            sum_excess += excess;
            // Subgradient contribution: -r_i (row i of scenarios).
            grad_w -= scenarios.row(i).transpose();
            ++count_exceeding;
        }
    }

    ObjectiveResult result;
    result.value = zeta + inv_n_alpha * sum_excess;
    result.grad_w = inv_n_alpha * grad_w;
    result.grad_zeta = 1.0 - inv_n_alpha * count_exceeding;

    return result;
}

ScalarCPU find_optimal_zeta(const MatrixXd& scenarios,
                             const VectorXd& w,
                             ScalarCPU alpha) {
    const int n_scenarios = static_cast<int>(scenarios.rows());

    // Compute losses and sort to find the alpha-quantile (VaR).
    VectorXd losses = -(scenarios * w);
    std::vector<double> sorted(losses.data(), losses.data() + n_scenarios);
    std::sort(sorted.begin(), sorted.end());

    // VaR = sorted[floor((1-alpha) * N)] using the confidence convention.
    // Here alpha = tail probability, so VaR index = floor((1-alpha) * N).
    Index var_index = static_cast<Index>(
        std::floor((1.0 - alpha) * n_scenarios));
    var_index = std::clamp(var_index, 0, n_scenarios - 1);

    return sorted[var_index];
}

}  // namespace cpo
