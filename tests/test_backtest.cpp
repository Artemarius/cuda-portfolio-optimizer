#define _USE_MATH_DEFINES
#include <cmath>

#include <gtest/gtest.h>

#include <fstream>
#include <random>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "backtest/backtest_config.h"
#include "backtest/backtest_engine.h"
#include "backtest/strategy.h"
#include "backtest/transaction_costs.h"
#include "reporting/report_writer.h"

using namespace cpo;

// ══════════════════════════════════════════════════════════════════════
// Transaction costs tests
// ══════════════════════════════════════════════════════════════════════

TEST(TransactionCosts, ZeroTurnover) {
    VectorXd w(3);
    w << 0.3, 0.3, 0.4;
    TransactionCostConfig cfg;
    cfg.cost_rate = 0.001;

    auto result = compute_transaction_costs(w, w, 1000000.0, cfg);

    EXPECT_NEAR(result.total_cost, 0.0, 1e-10);
    EXPECT_NEAR(result.turnover, 0.0, 1e-10);
    EXPECT_NEAR(result.cost_as_fraction, 0.0, 1e-10);
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(result.effective_weights(i), w(i), 1e-10);
    }
}

TEST(TransactionCosts, FullRebalance) {
    // From [1, 0, 0] to [0, 0, 1]: turnover = 2.0.
    VectorXd w_old(3), w_new(3);
    w_old << 1.0, 0.0, 0.0;
    w_new << 0.0, 0.0, 1.0;
    TransactionCostConfig cfg;
    cfg.cost_rate = 0.001;

    auto result = compute_transaction_costs(w_new, w_old, 1000000.0, cfg);

    EXPECT_NEAR(result.turnover, 2.0, 1e-10);
    EXPECT_NEAR(result.total_cost, 0.001 * 2.0 * 1000000.0, 1e-6);
    EXPECT_NEAR(result.cost_as_fraction, 0.002, 1e-10);
}

TEST(TransactionCosts, ProportionalCost) {
    VectorXd w_old(2), w_new(2);
    w_old << 0.5, 0.5;
    w_new << 0.7, 0.3;
    TransactionCostConfig cfg;
    cfg.cost_rate = 0.002;

    auto result = compute_transaction_costs(w_new, w_old, 500000.0, cfg);

    // Turnover = |0.2| + |-0.2| = 0.4.
    EXPECT_NEAR(result.turnover, 0.4, 1e-10);
    EXPECT_NEAR(result.total_cost, 0.002 * 0.4 * 500000.0, 1e-6);
}

TEST(TransactionCosts, ThresholdSuppression) {
    VectorXd w_old(3), w_new(3);
    w_old << 0.333, 0.333, 0.334;
    w_new << 0.334, 0.332, 0.334;  // Very small changes.
    TransactionCostConfig cfg;
    cfg.cost_rate = 0.001;
    cfg.min_trade_threshold = 0.01;  // Suppress trades < 1%.

    auto result = compute_transaction_costs(w_new, w_old, 1000000.0, cfg);

    // All trades are below threshold -> suppressed -> effective = old.
    EXPECT_NEAR(result.turnover, 0.0, 1e-10);
    EXPECT_NEAR(result.total_cost, 0.0, 1e-6);
}

TEST(TransactionCosts, DimensionMismatch) {
    VectorXd w_old(2), w_new(3);
    w_old << 0.5, 0.5;
    w_new << 0.3, 0.3, 0.4;
    TransactionCostConfig cfg;

    EXPECT_THROW(compute_transaction_costs(w_new, w_old, 100.0, cfg),
                 std::runtime_error);
}

// ══════════════════════════════════════════════════════════════════════
// Strategy tests
// ══════════════════════════════════════════════════════════════════════

namespace {
/// Build a simple synthetic return matrix for testing.
/// T rows, N columns, with known statistical properties.
MatrixXd make_test_returns(int T, int N, double base_return = 0.001) {
    MatrixXd ret(T, N);
    // Deterministic returns for reproducibility.
    for (int t = 0; t < T; ++t) {
        for (int j = 0; j < N; ++j) {
            // Asset j has return base_return * (j+1) + small variation.
            double variation = 0.001 * std::sin(static_cast<double>(t * (j + 1)));
            ret(t, j) = base_return * static_cast<double>(j + 1) + variation;
        }
    }
    return ret;
}
}  // anonymous namespace

TEST(EqualWeight, SumToOne) {
    MatrixXd returns = make_test_returns(60, 5);
    VectorXd w_prev = VectorXd::Constant(5, 0.2);

    EqualWeightStrategy strategy;
    auto result = strategy.allocate(returns, w_prev);

    EXPECT_TRUE(result.success);
    EXPECT_NEAR(result.weights.sum(), 1.0, 1e-12);
    EXPECT_EQ(result.weights.size(), 5);
    for (int i = 0; i < 5; ++i) {
        EXPECT_NEAR(result.weights(i), 0.2, 1e-12);
    }
}

TEST(EqualWeight, Name) {
    EqualWeightStrategy s;
    EXPECT_EQ(s.name(), "EqualWeight");
}

TEST(RiskParity, InverseVolWeighting) {
    // 2 assets: asset 0 has 2x the volatility of asset 1.
    // Risk parity: w_0 = (1/sigma_0) / (1/sigma_0 + 1/sigma_1)
    //            = (1/2sigma) / (1/2sigma + 1/sigma)
    //            = 0.5 / 1.5 = 1/3
    const int T = 252;
    MatrixXd returns(T, 2);
    for (int t = 0; t < T; ++t) {
        double angle = 2.0 * M_PI * t / 252.0;
        returns(t, 0) = 0.02 * std::sin(angle);  // Higher vol.
        returns(t, 1) = 0.01 * std::sin(angle);  // Lower vol.
    }

    RiskParityStrategy strategy;
    VectorXd w_prev = VectorXd::Constant(2, 0.5);
    auto result = strategy.allocate(returns, w_prev);

    EXPECT_TRUE(result.success);
    EXPECT_NEAR(result.weights.sum(), 1.0, 1e-10);
    // Asset 0 has 2x vol, so 1/vol_0 = 0.5 * (1/vol_1).
    // w_0 / w_1 should be ~0.5 (inverse vol ratio).
    EXPECT_NEAR(result.weights(0) / result.weights(1), 0.5, 1e-6);
}

TEST(RiskParity, EqualVolGivesEqualWeight) {
    const int T = 252;
    MatrixXd returns(T, 3);
    for (int t = 0; t < T; ++t) {
        double angle = 2.0 * M_PI * t / 252.0;
        for (int j = 0; j < 3; ++j) {
            // Same vol, different phase.
            returns(t, j) = 0.01 * std::sin(angle + j * 0.5);
        }
    }

    RiskParityStrategy strategy;
    VectorXd w_prev = VectorXd::Constant(3, 1.0 / 3.0);
    auto result = strategy.allocate(returns, w_prev);

    EXPECT_TRUE(result.success);
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(result.weights(i), 1.0 / 3.0, 0.02);
    }
}

TEST(MeanVariance, MinVarianceWeightsSumToOne) {
    MatrixXd returns = make_test_returns(120, 3);
    VectorXd w_prev = VectorXd::Constant(3, 1.0 / 3.0);

    MeanVarianceStrategy strategy;
    auto result = strategy.allocate(returns, w_prev);

    EXPECT_TRUE(result.success);
    EXPECT_NEAR(result.weights.sum(), 1.0, 1e-10);
    for (int i = 0; i < 3; ++i) {
        EXPECT_GE(result.weights(i), -1e-10);  // Long-only after clamping.
    }
}

TEST(MeanVariance, TwoAssetAnalytical) {
    // 2 uncorrelated assets with known variances.
    // Min-variance: w_1 = var_2 / (var_1 + var_2), w_2 = var_1 / (var_1 + var_2).
    const int T = 10000;
    MatrixXd returns(T, 2);
    // Asset 0: high vol.
    // Asset 1: low vol.
    std::mt19937 rng(42);
    std::normal_distribution<double> d0(0.0, 0.02);  // 2% daily vol
    std::normal_distribution<double> d1(0.0, 0.01);  // 1% daily vol
    for (int t = 0; t < T; ++t) {
        returns(t, 0) = d0(rng);
        returns(t, 1) = d1(rng);
    }

    MeanVarianceStrategy strategy;
    VectorXd w_prev = VectorXd::Constant(2, 0.5);
    auto result = strategy.allocate(returns, w_prev);

    EXPECT_TRUE(result.success);
    EXPECT_NEAR(result.weights.sum(), 1.0, 1e-10);
    // Expected: w_0 ≈ var_1 / (var_0 + var_1) = 0.0001 / 0.0005 = 0.2
    // Expected: w_1 ≈ var_0 / (var_0 + var_1) = 0.0004 / 0.0005 = 0.8
    EXPECT_NEAR(result.weights(0), 0.2, 0.05);
    EXPECT_NEAR(result.weights(1), 0.8, 0.05);
}

TEST(MeanCVaR, ProducesValidWeights) {
    MatrixXd returns = make_test_returns(120, 3, 0.001);
    VectorXd w_prev = VectorXd::Constant(3, 1.0 / 3.0);

    MeanCVaRConfig cfg;
    cfg.mc_config.n_scenarios = 1000;
    cfg.mc_config.seed = 42;
    cfg.use_gpu = false;
    cfg.admm_config.max_iter = 200;
    cfg.admm_config.verbose = false;

    MeanCVaRStrategy strategy(cfg);
    auto result = strategy.allocate(returns, w_prev);

    // Should produce valid weights regardless of convergence.
    EXPECT_NEAR(result.weights.sum(), 1.0, 1e-4);
    for (int i = 0; i < 3; ++i) {
        EXPECT_GE(result.weights(i), -1e-6);
    }
}

TEST(StrategyFactory, KnownNames) {
    EXPECT_NO_THROW(create_strategy("EqualWeight"));
    EXPECT_NO_THROW(create_strategy("RiskParity"));
    EXPECT_NO_THROW(create_strategy("MeanVariance"));
    EXPECT_NO_THROW(create_strategy("MeanCVaR"));
}

TEST(StrategyFactory, UnknownName) {
    EXPECT_THROW(create_strategy("InvalidStrategy"), std::runtime_error);
}

// ══════════════════════════════════════════════════════════════════════
// BacktestConfig tests
// ══════════════════════════════════════════════════════════════════════

TEST(BacktestConfig, LoadFromJson) {
    std::string path = std::string(TEST_DATA_DIR) + "/backtest_config.json";
    BacktestConfig cfg = load_backtest_config(path);

    EXPECT_EQ(cfg.price_csv_path, "test_prices.csv");
    EXPECT_EQ(cfg.tickers.size(), 3u);
    EXPECT_EQ(cfg.tickers[0], "AAPL");
    EXPECT_EQ(cfg.lookback_window, 60);
    EXPECT_EQ(cfg.rebalance_frequency, 21);
    EXPECT_EQ(cfg.strategy_name, "EqualWeight");
    EXPECT_NEAR(cfg.initial_capital, 100000.0, 1e-6);
    EXPECT_FALSE(cfg.use_gpu);
    EXPECT_NEAR(cfg.admm_config.confidence_level, 0.95, 1e-10);
    EXPECT_EQ(cfg.mc_config.n_scenarios, 5000);
    EXPECT_NEAR(cfg.transaction_costs.cost_rate, 0.001, 1e-10);
    EXPECT_NEAR(cfg.transaction_costs.min_trade_threshold, 0.001, 1e-10);
    EXPECT_EQ(cfg.output_dir, "results/backtest");
}

TEST(BacktestConfig, MissingFile) {
    EXPECT_THROW(load_backtest_config("nonexistent.json"), std::runtime_error);
}

// ══════════════════════════════════════════════════════════════════════
// Backtest engine tests
// ══════════════════════════════════════════════════════════════════════

namespace {

/// Create a synthetic ReturnData for testing the backtest engine.
ReturnData make_test_return_data(int T, int N) {
    ReturnData rd;
    rd.return_type = ReturnType::kSimple;

    // Generate dates.
    for (int t = 0; t < T; ++t) {
        rd.dates.push_back("2023-01-" + std::to_string(t + 1));
    }

    // Generate tickers.
    for (int j = 0; j < N; ++j) {
        rd.tickers.push_back("ASSET" + std::to_string(j));
    }

    // Deterministic returns.
    rd.returns = MatrixXd(T, N);
    for (int t = 0; t < T; ++t) {
        for (int j = 0; j < N; ++j) {
            rd.returns(t, j) = 0.001 * (j + 1)
                + 0.005 * std::sin(2.0 * M_PI * t / 20.0 + j);
        }
    }

    return rd;
}

}  // anonymous namespace

TEST(BacktestEngine, EqualWeightKnownReturns) {
    // 2 assets, 15 days of returns, lookback=5, rebalance every 3 days.
    ReturnData rd;
    rd.return_type = ReturnType::kSimple;
    rd.tickers = {"A", "B"};
    for (int t = 0; t < 15; ++t) {
        rd.dates.push_back("day" + std::to_string(t));
    }
    rd.returns = MatrixXd(15, 2);
    // Constant returns for easy hand-computation.
    for (int t = 0; t < 15; ++t) {
        rd.returns(t, 0) = 0.01;   // Asset A: +1% daily.
        rd.returns(t, 1) = 0.005;  // Asset B: +0.5% daily.
    }

    BacktestConfig cfg;
    cfg.lookback_window = 5;
    cfg.rebalance_frequency = 3;
    cfg.initial_capital = 10000.0;
    cfg.transaction_costs.cost_rate = 0.0;  // No costs for this test.

    EqualWeightStrategy strategy;
    auto result = run_backtest(rd, strategy, cfg);

    // Should have 10 days of snapshots (t=5..14).
    EXPECT_EQ(result.snapshots.size(), 10u);

    // First day should be a rebalance.
    EXPECT_TRUE(result.snapshots[0].is_rebalance_date);

    // Portfolio value should always increase (positive returns).
    for (size_t i = 1; i < result.snapshots.size(); ++i) {
        EXPECT_GT(result.snapshots[i].portfolio_value,
                  result.snapshots[i - 1].portfolio_value - 1e-6);
    }

    // Check total return is positive.
    EXPECT_GT(result.summary.total_return, 0.0);
}

TEST(BacktestEngine, TransactionCostsReduceValue) {
    ReturnData rd = make_test_return_data(100, 3);

    BacktestConfig cfg_no_cost;
    cfg_no_cost.lookback_window = 20;
    cfg_no_cost.rebalance_frequency = 10;
    cfg_no_cost.initial_capital = 100000.0;
    cfg_no_cost.transaction_costs.cost_rate = 0.0;

    BacktestConfig cfg_with_cost = cfg_no_cost;
    cfg_with_cost.transaction_costs.cost_rate = 0.01;  // 100 bps.

    EqualWeightStrategy strategy;
    auto result_no_cost = run_backtest(rd, strategy, cfg_no_cost);
    auto result_with_cost = run_backtest(rd, strategy, cfg_with_cost);

    // Portfolio value with costs should be lower.
    EXPECT_LT(result_with_cost.snapshots.back().portfolio_value,
              result_no_cost.snapshots.back().portfolio_value);

    // Total transaction cost should be positive.
    EXPECT_GT(result_with_cost.summary.total_transaction_cost, 0.0);
}

TEST(BacktestEngine, RebalanceCount) {
    ReturnData rd = make_test_return_data(100, 3);

    BacktestConfig cfg;
    cfg.lookback_window = 20;
    cfg.rebalance_frequency = 10;
    cfg.initial_capital = 100000.0;
    cfg.transaction_costs.cost_rate = 0.0;

    EqualWeightStrategy strategy;
    auto result = run_backtest(rd, strategy, cfg);

    // Count rebalance events.
    int rebal_count = 0;
    for (const auto& snap : result.snapshots) {
        if (snap.is_rebalance_date) ++rebal_count;
    }
    EXPECT_EQ(rebal_count, result.summary.n_rebalances);
    EXPECT_GT(rebal_count, 0);

    // First snapshot should always be a rebalance.
    EXPECT_TRUE(result.snapshots[0].is_rebalance_date);
}

TEST(BacktestEngine, WeightDrift) {
    // With no rebalancing after initial allocation, weights should drift.
    ReturnData rd;
    rd.return_type = ReturnType::kSimple;
    rd.tickers = {"A", "B"};
    for (int t = 0; t < 25; ++t) {
        rd.dates.push_back("day" + std::to_string(t));
    }
    rd.returns = MatrixXd(25, 2);
    // Asset A goes up, Asset B flat.
    for (int t = 0; t < 25; ++t) {
        rd.returns(t, 0) = 0.02;  // +2% daily.
        rd.returns(t, 1) = 0.0;   // Flat.
    }

    BacktestConfig cfg;
    cfg.lookback_window = 5;
    cfg.rebalance_frequency = 1000;  // Never rebalance after initial.
    cfg.initial_capital = 10000.0;
    cfg.transaction_costs.cost_rate = 0.0;

    EqualWeightStrategy strategy;
    auto result = run_backtest(rd, strategy, cfg);

    // After some days, weight of Asset A should drift above 0.5
    // (since it's appreciating while B is flat).
    const auto& last_snap = result.snapshots.back();
    EXPECT_GT(last_snap.weights(0), 0.5);
    EXPECT_LT(last_snap.weights(1), 0.5);
    EXPECT_NEAR(last_snap.weights.sum(), 1.0, 1e-10);
}

TEST(BacktestEngine, InsufficientDataThrows) {
    ReturnData rd = make_test_return_data(10, 3);

    BacktestConfig cfg;
    cfg.lookback_window = 20;  // Need 21 periods, only have 10.
    cfg.rebalance_frequency = 5;
    cfg.initial_capital = 100000.0;

    EqualWeightStrategy strategy;
    EXPECT_THROW(run_backtest(rd, strategy, cfg), std::runtime_error);
}

TEST(BacktestSummary, MaxDrawdownKnownCurve) {
    // Equity curve: [100, 110, 105, 120, 90, 100]
    // Peak=120 at index 3, trough=90 at index 4.
    // Max drawdown = (120 - 90) / 120 = 0.25.
    std::vector<PortfolioSnapshot> snaps;
    std::vector<double> values = {100.0, 110.0, 105.0, 120.0, 90.0, 100.0};
    for (size_t i = 0; i < values.size(); ++i) {
        PortfolioSnapshot s;
        s.date = "day" + std::to_string(i);
        s.portfolio_value = values[i];
        s.transaction_cost = 0.0;
        s.turnover = 0.0;
        s.daily_return = (i == 0) ? 0.0 : (values[i] / values[i - 1] - 1.0);
        s.weights = VectorXd::Constant(1, 1.0);
        s.is_rebalance_date = false;
        snaps.push_back(std::move(s));
    }

    auto summary = compute_backtest_summary(snaps, "Test");

    EXPECT_NEAR(summary.max_drawdown, 0.25, 1e-10);
    EXPECT_EQ(summary.n_days, 6);
    EXPECT_EQ(summary.strategy_name, "Test");
}

TEST(BacktestSummary, SharpePositive) {
    // Positive mean daily returns with some variance -> positive Sharpe.
    std::vector<PortfolioSnapshot> snaps;
    double value = 1000.0;
    for (int i = 0; i < 252; ++i) {
        PortfolioSnapshot s;
        s.date = "day" + std::to_string(i);
        // Positive mean with variation to ensure non-zero vol.
        double ret = 0.001 + 0.002 * std::sin(static_cast<double>(i));
        value *= (1.0 + ret);
        s.portfolio_value = value;
        s.daily_return = ret;
        s.transaction_cost = 0.0;
        s.turnover = 0.0;
        s.weights = VectorXd::Constant(1, 1.0);
        s.is_rebalance_date = false;
        snaps.push_back(std::move(s));
    }

    auto summary = compute_backtest_summary(snaps, "PositiveReturn");
    EXPECT_GT(summary.sharpe_ratio, 0.0);
    EXPECT_GT(summary.annualized_return, 0.0);
    EXPECT_GT(summary.annualized_volatility, 0.0);
}

// ══════════════════════════════════════════════════════════════════════
// Reporting tests
// ══════════════════════════════════════════════════════════════════════

TEST(Reporting, EquityCurveCSV) {
    BacktestResult result;
    result.tickers = {"A", "B"};

    for (int i = 0; i < 5; ++i) {
        PortfolioSnapshot s;
        s.date = "2023-01-0" + std::to_string(i + 1);
        s.portfolio_value = 10000.0 + i * 100.0;
        s.daily_return = 0.01;
        s.transaction_cost = (i == 0) ? 5.0 : 0.0;
        s.turnover = (i == 0) ? 0.5 : 0.0;
        s.weights = VectorXd::Constant(2, 0.5);
        s.is_rebalance_date = (i == 0);
        result.snapshots.push_back(std::move(s));
    }
    result.summary = compute_backtest_summary(result.snapshots, "Test");

    std::string path = "test_equity_curve.csv";
    write_equity_curve_csv(result, path);

    // Read back and verify.
    std::ifstream ifs(path);
    ASSERT_TRUE(ifs.is_open());
    std::string header;
    std::getline(ifs, header);
    EXPECT_EQ(header, "date,portfolio_value,daily_return,transaction_cost,is_rebalance");

    int line_count = 0;
    std::string line;
    while (std::getline(ifs, line)) ++line_count;
    EXPECT_EQ(line_count, 5);

    // Cleanup.
    std::remove(path.c_str());
}

TEST(Reporting, SummaryJSON) {
    BacktestSummary summary{};
    summary.strategy_name = "TestStrategy";
    summary.total_return = 0.15;
    summary.annualized_return = 0.12;
    summary.annualized_volatility = 0.08;
    summary.sharpe_ratio = 1.5;
    summary.sortino_ratio = 2.0;
    summary.max_drawdown = 0.05;
    summary.calmar_ratio = 2.4;
    summary.total_transaction_cost = 500.0;
    summary.avg_turnover = 0.3;
    summary.n_rebalances = 12;
    summary.n_days = 252;

    std::string path = "test_summary.json";
    write_summary_json(summary, path);

    // Read back and verify JSON structure.
    std::ifstream ifs(path);
    ASSERT_TRUE(ifs.is_open());
    nlohmann::json j = nlohmann::json::parse(ifs);

    EXPECT_EQ(j["strategy_name"], "TestStrategy");
    EXPECT_NEAR(j["total_return"].get<double>(), 0.15, 1e-10);
    EXPECT_NEAR(j["sharpe_ratio"].get<double>(), 1.5, 1e-10);
    EXPECT_EQ(j["n_rebalances"].get<int>(), 12);
    EXPECT_EQ(j["n_days"].get<int>(), 252);

    // Cleanup.
    std::remove(path.c_str());
}

TEST(Reporting, WeightsCSV) {
    BacktestResult result;
    result.tickers = {"X", "Y"};

    for (int i = 0; i < 5; ++i) {
        PortfolioSnapshot s;
        s.date = "day" + std::to_string(i);
        s.portfolio_value = 10000.0;
        s.daily_return = 0.0;
        s.transaction_cost = 0.0;
        s.turnover = 0.0;
        VectorXd w(2);
        w << 0.5 + i * 0.01, 0.5 - i * 0.01;
        s.weights = w;
        s.is_rebalance_date = (i % 2 == 0);  // Rebalance on even days.
        result.snapshots.push_back(std::move(s));
    }
    result.summary = compute_backtest_summary(result.snapshots, "Test");

    std::string path = "test_weights.csv";
    write_weights_csv(result, path);

    // Read back: should only have rebalance rows (days 0, 2, 4 = 3 data rows).
    std::ifstream ifs(path);
    ASSERT_TRUE(ifs.is_open());
    std::string header;
    std::getline(ifs, header);
    EXPECT_EQ(header, "date,X,Y");

    int line_count = 0;
    std::string line;
    while (std::getline(ifs, line)) ++line_count;
    EXPECT_EQ(line_count, 3);

    std::remove(path.c_str());
}

TEST(Reporting, ComparisonCSV) {
    // Build two mock results.
    std::vector<BacktestResult> results(2);
    for (int s = 0; s < 2; ++s) {
        results[s].tickers = {"A"};
        results[s].summary.strategy_name = (s == 0) ? "Strat1" : "Strat2";
        results[s].summary.total_return = 0.1 * (s + 1);
        results[s].summary.annualized_return = 0.08 * (s + 1);
        results[s].summary.annualized_volatility = 0.05;
        results[s].summary.sharpe_ratio = 1.0 + s;
        results[s].summary.sortino_ratio = 1.5;
        results[s].summary.max_drawdown = 0.03;
        results[s].summary.calmar_ratio = 2.0;
        results[s].summary.total_transaction_cost = 100.0;
        results[s].summary.avg_turnover = 0.2;
        results[s].summary.n_rebalances = 12;
        results[s].summary.n_days = 252;
    }

    std::string path = "test_comparison.csv";
    write_comparison_csv(results, path);

    std::ifstream ifs(path);
    ASSERT_TRUE(ifs.is_open());
    std::string header;
    std::getline(ifs, header);
    // Verify header has expected columns.
    EXPECT_NE(header.find("strategy"), std::string::npos);
    EXPECT_NE(header.find("sharpe"), std::string::npos);

    int line_count = 0;
    std::string line;
    while (std::getline(ifs, line)) ++line_count;
    EXPECT_EQ(line_count, 2);

    std::remove(path.c_str());
}

TEST(Reporting, ComparisonJSON) {
    std::vector<BacktestResult> results(2);
    for (int s = 0; s < 2; ++s) {
        results[s].summary.strategy_name = "S" + std::to_string(s);
        results[s].summary.total_return = 0.1;
        results[s].summary.n_days = 100;
    }

    std::string path = "test_comparison.json";
    write_comparison_json(results, path);

    std::ifstream ifs(path);
    ASSERT_TRUE(ifs.is_open());
    nlohmann::json j = nlohmann::json::parse(ifs);
    EXPECT_TRUE(j.is_array());
    EXPECT_EQ(j.size(), 2u);
    EXPECT_EQ(j[0]["strategy_name"], "S0");
    EXPECT_EQ(j[1]["strategy_name"], "S1");

    std::remove(path.c_str());
}
