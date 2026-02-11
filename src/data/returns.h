#pragma once

/// @file returns.h
/// @brief Compute asset returns from price data.

#include "data/market_data.h"

namespace cpo {

/// Compute returns from price data with full metadata.
///
/// Simple return: r_t = p_t / p_{t-1} - 1
/// Log return:    r_t = ln(p_t / p_{t-1})
///
/// Return dates use end-of-period convention: return at index i corresponds
/// to the price change from date i to date i+1 in the original PriceData,
/// and is labeled with date i+1.
///
/// @param prices     Input price data (T dates x N assets).
/// @param type       Return computation method.
/// @return ReturnData with (T-1) x N return matrix and metadata.
/// @throws std::invalid_argument if prices has fewer than 2 rows.
ReturnData compute_returns(const PriceData& prices, ReturnType type);

/// Compute returns from a raw price matrix (no metadata).
///
/// @param prices  Price matrix, T x N (CPU-precision).
/// @param type    Return computation method.
/// @return Return matrix, (T-1) x N.
/// @throws std::invalid_argument if prices has fewer than 2 rows.
MatrixXd compute_returns(const MatrixXd& prices, ReturnType type);

/// Compute excess returns by subtracting the annualized risk-free rate.
///
/// excess_r_t = r_t - rf / periods_per_year
///
/// @param returns          Input return data.
/// @param risk_free_rate   Annualized risk-free rate (e.g. 0.05 for 5%).
/// @param periods_per_year Number of return periods per year (e.g. 252 for daily).
/// @return ReturnData with excess returns and the same metadata.
ReturnData compute_excess_returns(const ReturnData& returns,
                                  double risk_free_rate,
                                  double periods_per_year);

}  // namespace cpo
