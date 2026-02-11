#include <cmath>
#include <fstream>
#include <string>

#include <gtest/gtest.h>

#include "data/csv_loader.h"
#include "data/market_data.h"
#include "data/returns.h"
#include "data/universe.h"

// TEST_DATA_DIR is defined via compile definition in CMakeLists.txt.
#ifndef TEST_DATA_DIR
#error "TEST_DATA_DIR must be defined"
#endif

namespace cpo {
namespace {

const std::string kDataDir = TEST_DATA_DIR;

std::string data_path(const std::string& filename) {
    return kDataDir + "/" + filename;
}

// ═══════════════════════════════════════════════════════════════════
// CSV Loader Tests
// ═══════════════════════════════════════════════════════════════════

TEST(CsvLoader, LoadAll2Asset) {
    auto pd = load_csv_prices(data_path("prices_2asset.csv"));

    EXPECT_EQ(pd.num_dates(), 6);
    EXPECT_EQ(pd.num_assets(), 2);
    EXPECT_EQ(pd.tickers[0], "AAPL");
    EXPECT_EQ(pd.tickers[1], "MSFT");
    EXPECT_EQ(pd.dates[0], "2023-01-02");
    EXPECT_EQ(pd.dates[5], "2023-01-09");

    // Spot-check values.
    EXPECT_DOUBLE_EQ(pd.prices(0, 0), 100.0);
    EXPECT_DOUBLE_EQ(pd.prices(0, 1), 200.0);
    EXPECT_DOUBLE_EQ(pd.prices(5, 0), 110.0);
    EXPECT_DOUBLE_EQ(pd.prices(5, 1), 208.0);
}

TEST(CsvLoader, LoadAll5Asset) {
    auto pd = load_csv_prices(data_path("prices_5asset.csv"));

    EXPECT_EQ(pd.num_dates(), 11);
    EXPECT_EQ(pd.num_assets(), 5);
    EXPECT_EQ(pd.tickers[2], "GOOG");
    EXPECT_EQ(pd.tickers[4], "TSLA");
}

TEST(CsvLoader, FilterTickers) {
    std::vector<std::string> tickers = {"MSFT", "AAPL"};
    auto pd = load_csv_prices(data_path("prices_2asset.csv"), tickers);

    EXPECT_EQ(pd.num_assets(), 2);
    // Order preserved from the argument, not the CSV header.
    EXPECT_EQ(pd.tickers[0], "MSFT");
    EXPECT_EQ(pd.tickers[1], "AAPL");

    // First row: MSFT=200, AAPL=100.
    EXPECT_DOUBLE_EQ(pd.prices(0, 0), 200.0);
    EXPECT_DOUBLE_EQ(pd.prices(0, 1), 100.0);
}

TEST(CsvLoader, FilterByUniverse) {
    auto universe = load_universe(data_path("universe_test.json"));
    auto pd = load_csv_prices(data_path("prices_5asset.csv"), universe);

    // Universe: tickers [AAPL, GOOG, TSLA], dates [2023-01-04, 2023-01-10].
    EXPECT_EQ(pd.num_assets(), 3);
    EXPECT_EQ(pd.tickers[0], "AAPL");
    EXPECT_EQ(pd.tickers[1], "GOOG");
    EXPECT_EQ(pd.tickers[2], "TSLA");

    // Date range: 2023-01-04 through 2023-01-10 inclusive.
    // From the 5-asset CSV: rows 2023-01-04, 05, 06, 09, 10 → 5 rows.
    EXPECT_EQ(pd.num_dates(), 5);
    EXPECT_EQ(pd.dates.front(), "2023-01-04");
    EXPECT_EQ(pd.dates.back(), "2023-01-10");
}

TEST(CsvLoader, MissingDataDropRows) {
    auto pd = load_csv_prices(data_path("prices_missing.csv"),
                              MissingDataPolicy::kDropRows);

    // prices_missing.csv has 5 rows. Rows 2023-01-03 (BBB missing) and
    // 2023-01-05 (AAA + CCC missing) should be dropped → 3 rows remain.
    EXPECT_EQ(pd.num_dates(), 3);
    EXPECT_EQ(pd.dates[0], "2023-01-02");
    EXPECT_EQ(pd.dates[1], "2023-01-04");
    EXPECT_EQ(pd.dates[2], "2023-01-06");
}

TEST(CsvLoader, MissingDataForwardFill) {
    auto pd = load_csv_prices(data_path("prices_missing.csv"),
                              MissingDataPolicy::kForwardFill);

    // Forward fill: row 2023-01-03 BBB gets 200.0 (from 01-02).
    // Row 2023-01-05: AAA gets 105.0 (from 01-04), CCC gets 310.0 (from 01-04).
    // All rows complete → 5 rows.
    EXPECT_EQ(pd.num_dates(), 5);

    // BBB column (index 1): 200, 200(filled), 198, 201, 205.
    EXPECT_DOUBLE_EQ(pd.prices(1, 1), 200.0);

    // AAA column (index 0): row 2023-01-05 → 105.0 (filled).
    EXPECT_DOUBLE_EQ(pd.prices(3, 0), 105.0);

    // CCC column (index 2): row 2023-01-05 → 310.0 (filled).
    EXPECT_DOUBLE_EQ(pd.prices(3, 2), 310.0);
}

TEST(CsvLoader, NonexistentFileThrows) {
    EXPECT_THROW(load_csv_prices("no_such_file.csv"), std::runtime_error);
}

TEST(CsvLoader, MissingTickerThrows) {
    std::vector<std::string> tickers = {"AAPL", "NOPE"};
    EXPECT_THROW(load_csv_prices(data_path("prices_2asset.csv"), tickers),
                 std::runtime_error);
}

// ═══════════════════════════════════════════════════════════════════
// Return Computation Tests
// ═══════════════════════════════════════════════════════════════════

TEST(Returns, SimpleReturns2Asset) {
    auto pd = load_csv_prices(data_path("prices_2asset.csv"));
    auto rd = compute_returns(pd, ReturnType::kSimple);

    EXPECT_EQ(rd.num_periods(), 5);
    EXPECT_EQ(rd.num_assets(), 2);
    EXPECT_EQ(rd.return_type, ReturnType::kSimple);

    // End-of-period dates: first return date = second price date.
    EXPECT_EQ(rd.dates[0], "2023-01-03");
    EXPECT_EQ(rd.dates[4], "2023-01-09");

    // Hand-computed simple returns for AAPL:
    // 102/100 - 1 = 0.02
    // 105/102 - 1 ≈ 0.02941176...
    // 103/105 - 1 ≈ -0.01904762...
    // 107/103 - 1 ≈ 0.03883495...
    // 110/107 - 1 ≈ 0.02803738...
    EXPECT_NEAR(rd.returns(0, 0), 0.02, 1e-10);
    EXPECT_NEAR(rd.returns(1, 0), 105.0 / 102.0 - 1.0, 1e-10);
    EXPECT_NEAR(rd.returns(2, 0), 103.0 / 105.0 - 1.0, 1e-10);
    EXPECT_NEAR(rd.returns(3, 0), 107.0 / 103.0 - 1.0, 1e-10);
    EXPECT_NEAR(rd.returns(4, 0), 110.0 / 107.0 - 1.0, 1e-10);

    // Hand-computed simple returns for MSFT:
    // 198/200 - 1 = -0.01
    EXPECT_NEAR(rd.returns(0, 1), -0.01, 1e-10);
    EXPECT_NEAR(rd.returns(1, 1), 201.0 / 198.0 - 1.0, 1e-10);
}

TEST(Returns, LogReturns2Asset) {
    auto pd = load_csv_prices(data_path("prices_2asset.csv"));
    auto rd = compute_returns(pd, ReturnType::kLog);

    EXPECT_EQ(rd.num_periods(), 5);
    EXPECT_EQ(rd.return_type, ReturnType::kLog);

    // Hand-computed log returns for AAPL:
    // ln(102/100) ≈ 0.01980263...
    // ln(105/102) ≈ 0.02898753...
    EXPECT_NEAR(rd.returns(0, 0), std::log(102.0 / 100.0), 1e-10);
    EXPECT_NEAR(rd.returns(1, 0), std::log(105.0 / 102.0), 1e-10);
    EXPECT_NEAR(rd.returns(2, 0), std::log(103.0 / 105.0), 1e-10);
    EXPECT_NEAR(rd.returns(3, 0), std::log(107.0 / 103.0), 1e-10);
    EXPECT_NEAR(rd.returns(4, 0), std::log(110.0 / 107.0), 1e-10);

    // MSFT: ln(198/200) ≈ -0.01005034...
    EXPECT_NEAR(rd.returns(0, 1), std::log(198.0 / 200.0), 1e-10);
}

TEST(Returns, MatrixOverload) {
    MatrixXd prices(3, 2);
    prices << 100.0, 200.0,
              110.0, 190.0,
              105.0, 210.0;

    auto ret = compute_returns(prices, ReturnType::kSimple);
    EXPECT_EQ(ret.rows(), 2);
    EXPECT_EQ(ret.cols(), 2);

    EXPECT_NEAR(ret(0, 0), 0.1, 1e-10);
    EXPECT_NEAR(ret(0, 1), -0.05, 1e-10);
    EXPECT_NEAR(ret(1, 0), 105.0 / 110.0 - 1.0, 1e-10);
    EXPECT_NEAR(ret(1, 1), 210.0 / 190.0 - 1.0, 1e-10);
}

TEST(Returns, ExcessReturns) {
    auto pd = load_csv_prices(data_path("prices_2asset.csv"));
    auto rd = compute_returns(pd, ReturnType::kSimple);

    double rf = 0.05;         // 5% annualized
    double periods = 252.0;   // daily
    auto excess = compute_excess_returns(rd, rf, periods);

    double per_period_rf = rf / periods;
    EXPECT_EQ(excess.num_periods(), rd.num_periods());
    EXPECT_EQ(excess.num_assets(), rd.num_assets());

    // Each excess return = original return - rf/252.
    for (Index i = 0; i < excess.num_periods(); ++i) {
        for (Index j = 0; j < excess.num_assets(); ++j) {
            EXPECT_NEAR(excess.returns(i, j),
                        rd.returns(i, j) - per_period_rf, 1e-10);
        }
    }
}

TEST(Returns, SingleRowThrows) {
    MatrixXd prices(1, 2);
    prices << 100.0, 200.0;
    EXPECT_THROW(compute_returns(prices, ReturnType::kSimple),
                 std::invalid_argument);
}

TEST(Returns, TwoRowsGivesOneReturn) {
    MatrixXd prices(2, 1);
    prices << 100.0, 110.0;
    auto ret = compute_returns(prices, ReturnType::kSimple);
    EXPECT_EQ(ret.rows(), 1);
    EXPECT_NEAR(ret(0, 0), 0.1, 1e-10);
}

// ═══════════════════════════════════════════════════════════════════
// Universe Tests
// ═══════════════════════════════════════════════════════════════════

TEST(Universe, LoadFromJson) {
    auto u = load_universe(data_path("universe_test.json"));

    EXPECT_EQ(u.tickers.size(), 3u);
    EXPECT_EQ(u.tickers[0], "AAPL");
    EXPECT_EQ(u.tickers[1], "GOOG");
    EXPECT_EQ(u.tickers[2], "TSLA");
    EXPECT_EQ(u.start_date, "2023-01-04");
    EXPECT_EQ(u.end_date, "2023-01-10");
}

TEST(Universe, NonexistentFileThrows) {
    EXPECT_THROW(load_universe("no_such_file.json"), std::runtime_error);
}

}  // namespace
}  // namespace cpo
