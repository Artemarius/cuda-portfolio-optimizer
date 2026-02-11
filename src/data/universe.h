#pragma once

/// @file universe.h
/// @brief Asset universe definition for filtering price data.

#include <string>
#include <vector>

namespace cpo {

/// Defines a subset of assets and a date range for analysis.
struct Universe {
    std::vector<std::string> tickers;  ///< Asset ticker symbols to include.
    std::string start_date;            ///< Inclusive start date (ISO 8601). Empty = no filter.
    std::string end_date;              ///< Inclusive end date (ISO 8601). Empty = no filter.
};

/// Load a universe definition from a JSON file.
///
/// Expected JSON format:
/// @code
/// {
///   "tickers": ["AAPL", "MSFT", "GOOG"],
///   "start_date": "2020-01-01",
///   "end_date": "2023-12-31"
/// }
/// @endcode
///
/// @param json_path Path to the JSON file.
/// @return The parsed Universe.
/// @throws std::runtime_error if the file cannot be read or parsed.
Universe load_universe(const std::string& json_path);

}  // namespace cpo
