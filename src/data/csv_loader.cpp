#include "data/csv_loader.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <vector>

#include <spdlog/spdlog.h>

namespace cpo {
namespace {

/// Split a string by a delimiter. Handles trailing delimiters correctly
/// (e.g. "a,,b," → ["a", "", "b", ""]).
std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> tokens;
    std::string::size_type start = 0;
    std::string::size_type end;
    while ((end = s.find(delim, start)) != std::string::npos) {
        tokens.push_back(s.substr(start, end - start));
        start = end + 1;
    }
    tokens.push_back(s.substr(start));
    return tokens;
}

/// Trim leading and trailing whitespace (including \r).
std::string trim(const std::string& s) {
    const char* ws = " \t\r\n";
    auto start = s.find_first_not_of(ws);
    if (start == std::string::npos) return "";
    auto end = s.find_last_not_of(ws);
    return s.substr(start, end - start + 1);
}

/// Check if a trimmed cell represents a missing value.
bool is_missing(const std::string& cell) {
    auto t = trim(cell);
    return t.empty() || t == "NA" || t == "N/A" || t == "NaN" || t == "nan" ||
           t == "#N/A";
}

/// Strip UTF-8 BOM if present at the start of a string.
void strip_bom(std::string& s) {
    if (s.size() >= 3 &&
        static_cast<unsigned char>(s[0]) == 0xEF &&
        static_cast<unsigned char>(s[1]) == 0xBB &&
        static_cast<unsigned char>(s[2]) == 0xBF) {
        s.erase(0, 3);
    }
}

/// Locale-safe string-to-double conversion using std::strtod.
/// @returns The parsed value, or NaN on failure.
double parse_double(const std::string& s) {
    const char* begin = s.c_str();
    char* end = nullptr;
    double val = std::strtod(begin, &end);
    if (end == begin) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return val;
}

/// Sentinel value marking a missing cell in the intermediate buffer.
constexpr double kMissing = std::numeric_limits<double>::quiet_NaN();

/// Internal representation of one row of parsed CSV data.
struct RawRow {
    std::string date;
    std::vector<double> values;  ///< One per header column; NaN = missing.
};

/// Parse the entire CSV into raw rows and header tickers.
/// Handles BOM, \r\n, missing cells, non-positive price warnings.
void parse_csv(const std::string& csv_path,
               std::vector<std::string>& out_tickers,
               std::vector<RawRow>& out_rows) {
    std::ifstream file(csv_path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open CSV file: " + csv_path);
    }

    // --- Header ---
    std::string header_line;
    if (!std::getline(file, header_line)) {
        throw std::runtime_error("CSV file is empty: " + csv_path);
    }
    strip_bom(header_line);
    auto header_tokens = split(header_line, ',');

    if (header_tokens.size() < 2) {
        throw std::runtime_error(
            "CSV header must have at least 2 columns (date + 1 ticker): " +
            csv_path);
    }

    const Index num_cols = static_cast<Index>(header_tokens.size()) - 1;
    out_tickers.clear();
    out_tickers.reserve(num_cols);
    for (Index i = 1; i <= num_cols; ++i) {
        out_tickers.push_back(trim(header_tokens[i]));
    }

    // --- Data rows ---
    std::string line;
    int line_num = 1;  // header was line 1
    while (std::getline(file, line)) {
        ++line_num;
        auto trimmed = trim(line);
        if (trimmed.empty()) continue;  // skip blank lines

        auto tokens = split(trimmed, ',');
        if (static_cast<Index>(tokens.size()) != num_cols + 1) {
            spdlog::warn("{}:{}: expected {} columns, got {}; skipping row",
                         csv_path, line_num, num_cols + 1, tokens.size());
            continue;
        }

        RawRow row;
        row.date = trim(tokens[0]);
        row.values.resize(num_cols);

        for (Index j = 0; j < num_cols; ++j) {
            if (is_missing(tokens[j + 1])) {
                row.values[j] = kMissing;
            } else {
                double val = parse_double(trim(tokens[j + 1]));
                if (std::isnan(val)) {
                    spdlog::warn("{}:{}: could not parse '{}' for {}; treating as missing",
                                 csv_path, line_num, trim(tokens[j + 1]),
                                 out_tickers[j]);
                    row.values[j] = kMissing;
                } else if (val <= 0.0) {
                    spdlog::warn("{}:{}: non-positive price {:.6f} for {}",
                                 csv_path, line_num, val, out_tickers[j]);
                    row.values[j] = val;
                } else {
                    row.values[j] = val;
                }
            }
        }
        out_rows.push_back(std::move(row));
    }
}

/// Apply missing data policy to raw rows, writing into a PriceData struct.
/// Only writes columns at the given indices.
PriceData apply_policy(const std::vector<std::string>& all_tickers,
                       std::vector<RawRow>& rows,
                       const std::vector<Index>& col_indices,
                       const std::vector<std::string>& out_ticker_names,
                       MissingDataPolicy policy) {
    const Index n_assets = static_cast<Index>(col_indices.size());

    if (policy == MissingDataPolicy::kForwardFill) {
        // Forward-fill: for each selected column, replace NaN with previous
        // row's value. First-row NaN cannot be filled — those rows will still
        // contain NaN and will be silently kept (caller can check).
        for (Index ci = 0; ci < n_assets; ++ci) {
            Index src_col = col_indices[ci];
            for (size_t r = 1; r < rows.size(); ++r) {
                if (std::isnan(rows[r].values[src_col]) &&
                    !std::isnan(rows[r - 1].values[src_col])) {
                    rows[r].values[src_col] = rows[r - 1].values[src_col];
                }
            }
        }
    }

    // Collect rows that have no NaN in the selected columns.
    std::vector<size_t> good_row_indices;
    good_row_indices.reserve(rows.size());
    for (size_t r = 0; r < rows.size(); ++r) {
        bool has_nan = false;
        for (Index ci = 0; ci < n_assets; ++ci) {
            if (std::isnan(rows[r].values[col_indices[ci]])) {
                has_nan = true;
                break;
            }
        }
        if (!has_nan) {
            good_row_indices.push_back(r);
        }
    }

    if (good_row_indices.empty()) {
        throw std::runtime_error("No complete rows remain after applying missing data policy");
    }

    // Build PriceData
    PriceData result;
    result.tickers = out_ticker_names;

    const Index n_rows = static_cast<Index>(good_row_indices.size());
    result.dates.reserve(n_rows);
    result.prices.resize(n_rows, n_assets);

    for (Index i = 0; i < n_rows; ++i) {
        size_t r = good_row_indices[i];
        result.dates.push_back(rows[r].date);
        for (Index j = 0; j < n_assets; ++j) {
            result.prices(i, j) = rows[r].values[col_indices[j]];
        }
    }

    return result;
}

/// Find column indices for requested tickers in the header.
std::vector<Index> resolve_ticker_indices(
    const std::vector<std::string>& header_tickers,
    const std::vector<std::string>& requested,
    const std::string& csv_path) {
    std::vector<Index> indices;
    indices.reserve(requested.size());
    for (const auto& ticker : requested) {
        auto it = std::find(header_tickers.begin(), header_tickers.end(), ticker);
        if (it == header_tickers.end()) {
            throw std::runtime_error("Ticker '" + ticker +
                                     "' not found in CSV header: " + csv_path);
        }
        indices.push_back(static_cast<Index>(it - header_tickers.begin()));
    }
    return indices;
}

/// Filter raw rows by date range [start, end] (inclusive, lexicographic).
void filter_date_range(std::vector<RawRow>& rows,
                       const std::string& start_date,
                       const std::string& end_date) {
    rows.erase(
        std::remove_if(rows.begin(), rows.end(),
                       [&](const RawRow& row) {
                           if (!start_date.empty() && row.date < start_date)
                               return true;
                           if (!end_date.empty() && row.date > end_date)
                               return true;
                           return false;
                       }),
        rows.end());
}

}  // namespace

PriceData load_csv_prices(const std::string& csv_path,
                          MissingDataPolicy policy) {
    std::vector<std::string> tickers;
    std::vector<RawRow> rows;
    parse_csv(csv_path, tickers, rows);

    if (rows.empty()) {
        throw std::runtime_error("CSV file has no data rows: " + csv_path);
    }

    // Use all columns.
    const Index n = static_cast<Index>(tickers.size());
    std::vector<Index> all_cols(n);
    for (Index i = 0; i < n; ++i) all_cols[i] = i;

    return apply_policy(tickers, rows, all_cols, tickers, policy);
}

PriceData load_csv_prices(const std::string& csv_path,
                          const std::vector<std::string>& tickers,
                          MissingDataPolicy policy) {
    std::vector<std::string> header_tickers;
    std::vector<RawRow> rows;
    parse_csv(csv_path, header_tickers, rows);

    if (rows.empty()) {
        throw std::runtime_error("CSV file has no data rows: " + csv_path);
    }

    auto col_indices = resolve_ticker_indices(header_tickers, tickers, csv_path);
    return apply_policy(header_tickers, rows, col_indices, tickers, policy);
}

PriceData load_csv_prices(const std::string& csv_path,
                          const Universe& universe,
                          MissingDataPolicy policy) {
    std::vector<std::string> header_tickers;
    std::vector<RawRow> rows;
    parse_csv(csv_path, header_tickers, rows);

    if (rows.empty()) {
        throw std::runtime_error("CSV file has no data rows: " + csv_path);
    }

    // Filter dates.
    filter_date_range(rows, universe.start_date, universe.end_date);

    if (rows.empty()) {
        throw std::runtime_error(
            "No rows remain after date filtering [" + universe.start_date +
            ", " + universe.end_date + "] in: " + csv_path);
    }

    auto col_indices =
        resolve_ticker_indices(header_tickers, universe.tickers, csv_path);
    return apply_policy(header_tickers, rows, col_indices, universe.tickers,
                        policy);
}

}  // namespace cpo
