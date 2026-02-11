#include "data/returns.h"

#include <cmath>
#include <stdexcept>

namespace cpo {

MatrixXd compute_returns(const MatrixXd& prices, ReturnType type) {
    const auto T = prices.rows();
    const auto N = prices.cols();

    if (T < 2) {
        throw std::invalid_argument(
            "compute_returns: need at least 2 price observations, got " +
            std::to_string(T));
    }

    MatrixXd returns(T - 1, N);

    if (type == ReturnType::kSimple) {
        // r_t = p_t / p_{t-1} - 1
        for (Eigen::Index t = 1; t < T; ++t) {
            returns.row(t - 1) =
                prices.row(t).array() / prices.row(t - 1).array() - 1.0;
        }
    } else {
        // r_t = ln(p_t / p_{t-1})
        for (Eigen::Index t = 1; t < T; ++t) {
            returns.row(t - 1) =
                (prices.row(t).array() / prices.row(t - 1).array()).log();
        }
    }

    return returns;
}

ReturnData compute_returns(const PriceData& prices, ReturnType type) {
    if (prices.num_dates() < 2) {
        throw std::invalid_argument(
            "compute_returns: need at least 2 price observations, got " +
            std::to_string(prices.num_dates()));
    }

    ReturnData result;
    result.tickers = prices.tickers;
    result.return_type = type;

    // End-of-period convention: return i uses date i+1 from prices.
    result.dates.reserve(prices.num_dates() - 1);
    for (Index i = 1; i < prices.num_dates(); ++i) {
        result.dates.push_back(prices.dates[i]);
    }

    result.returns = compute_returns(prices.prices, type);
    return result;
}

ReturnData compute_excess_returns(const ReturnData& returns,
                                  double risk_free_rate,
                                  double periods_per_year) {
    ReturnData result;
    result.dates = returns.dates;
    result.tickers = returns.tickers;
    result.return_type = returns.return_type;

    double per_period_rf = risk_free_rate / periods_per_year;
    result.returns = returns.returns.array() - per_period_rf;

    return result;
}

}  // namespace cpo
