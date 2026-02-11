#pragma once

/// @file market_data.h
/// @brief Core data structures for price and return data.

#include <string>
#include <vector>

#include "core/types.h"

namespace cpo {

/// Policy for handling missing values in price data.
enum class MissingDataPolicy {
    kDropRows,     ///< Remove any row containing a missing value.
    kForwardFill,  ///< Fill missing values with the previous day's price.
};

/// Type of return computation.
enum class ReturnType {
    kSimple,  ///< Simple return: r_t = p_t / p_{t-1} - 1
    kLog,     ///< Log return:    r_t = ln(p_t / p_{t-1})
};

/// Time series of asset prices in wide format (T dates x N assets).
struct PriceData {
    std::vector<std::string> dates;    ///< ISO 8601 date strings, length T.
    std::vector<std::string> tickers;  ///< Asset ticker symbols, length N.
    MatrixXd prices;                   ///< Price matrix, T x N (CPU-precision).

    /// Number of time periods (rows).
    Index num_dates() const { return static_cast<Index>(dates.size()); }

    /// Number of assets (columns).
    Index num_assets() const { return static_cast<Index>(tickers.size()); }
};

/// Time series of asset returns in wide format ((T-1) dates x N assets).
struct ReturnData {
    std::vector<std::string> dates;    ///< End-of-period dates, length T-1.
    std::vector<std::string> tickers;  ///< Asset ticker symbols, length N.
    MatrixXd returns;                  ///< Return matrix, (T-1) x N (CPU-precision).
    ReturnType return_type;            ///< How returns were computed.

    /// Number of return periods (rows).
    Index num_periods() const { return static_cast<Index>(dates.size()); }

    /// Number of assets (columns).
    Index num_assets() const { return static_cast<Index>(tickers.size()); }
};

}  // namespace cpo
