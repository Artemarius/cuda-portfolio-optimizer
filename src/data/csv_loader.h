#pragma once

/// @file csv_loader.h
/// @brief Load price data from wide-format CSV files.
///
/// Expected CSV format (wide):
/// @code
/// Date,AAPL,MSFT,GOOG
/// 2023-01-02,100.0,200.0,150.0
/// 2023-01-03,102.0,198.0,152.0
/// @endcode

#include <string>

#include "data/market_data.h"
#include "data/universe.h"

namespace cpo {

/// Load all tickers from a wide-format CSV price file.
///
/// @param csv_path  Path to the CSV file.
/// @param policy    How to handle missing values (default: kDropRows).
/// @return Parsed price data with dates, tickers, and price matrix.
/// @throws std::runtime_error if the file cannot be read or has no data rows.
PriceData load_csv_prices(const std::string& csv_path,
                          MissingDataPolicy policy = MissingDataPolicy::kDropRows);

/// Load specific tickers from a wide-format CSV price file.
///
/// @param csv_path  Path to the CSV file.
/// @param tickers   Tickers to include (order is preserved from the argument).
/// @param policy    How to handle missing values (default: kDropRows).
/// @return Parsed price data filtered to the requested tickers.
/// @throws std::runtime_error if the file cannot be read, has no data, or a
///         requested ticker is not found in the header.
PriceData load_csv_prices(const std::string& csv_path,
                          const std::vector<std::string>& tickers,
                          MissingDataPolicy policy = MissingDataPolicy::kDropRows);

/// Load prices filtered by a Universe (tickers + date range).
///
/// @param csv_path  Path to the CSV file.
/// @param universe  Universe defining tickers and optional date range.
/// @param policy    How to handle missing values (default: kDropRows).
/// @return Parsed price data filtered to the universe.
/// @throws std::runtime_error if the file cannot be read, has no data, or a
///         requested ticker is not found in the header.
PriceData load_csv_prices(const std::string& csv_path,
                          const Universe& universe,
                          MissingDataPolicy policy = MissingDataPolicy::kDropRows);

}  // namespace cpo
