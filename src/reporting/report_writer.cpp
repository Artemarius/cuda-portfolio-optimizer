#include "reporting/report_writer.h"

#include <fstream>
#include <iomanip>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace cpo {

void write_equity_curve_csv(const BacktestResult& result,
                            const std::string& path) {
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        throw std::runtime_error("write_equity_curve_csv: cannot open " + path);
    }

    ofs << std::fixed << std::setprecision(6);
    ofs << "date,portfolio_value,daily_return,transaction_cost,is_rebalance\n";

    for (const auto& snap : result.snapshots) {
        ofs << snap.date << ","
            << snap.portfolio_value << ","
            << snap.daily_return << ","
            << snap.transaction_cost << ","
            << (snap.is_rebalance_date ? 1 : 0) << "\n";
    }
}

void write_weights_csv(const BacktestResult& result,
                       const std::string& path) {
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        throw std::runtime_error("write_weights_csv: cannot open " + path);
    }

    ofs << std::fixed << std::setprecision(6);

    // Header.
    ofs << "date";
    for (const auto& ticker : result.tickers) {
        ofs << "," << ticker;
    }
    ofs << "\n";

    // Only write rebalance-day rows.
    for (const auto& snap : result.snapshots) {
        if (!snap.is_rebalance_date) continue;
        ofs << snap.date;
        for (Index i = 0; i < static_cast<Index>(snap.weights.size()); ++i) {
            ofs << "," << snap.weights(i);
        }
        ofs << "\n";
    }
}

namespace {

nlohmann::json summary_to_json(const BacktestSummary& summary) {
    return {
        {"strategy_name", summary.strategy_name},
        {"total_return", summary.total_return},
        {"annualized_return", summary.annualized_return},
        {"annualized_volatility", summary.annualized_volatility},
        {"sharpe_ratio", summary.sharpe_ratio},
        {"sortino_ratio", summary.sortino_ratio},
        {"max_drawdown", summary.max_drawdown},
        {"calmar_ratio", summary.calmar_ratio},
        {"total_transaction_cost", summary.total_transaction_cost},
        {"avg_turnover", summary.avg_turnover},
        {"n_rebalances", summary.n_rebalances},
        {"n_days", summary.n_days}
    };
}

}  // anonymous namespace

void write_summary_json(const BacktestSummary& summary,
                        const std::string& path) {
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        throw std::runtime_error("write_summary_json: cannot open " + path);
    }
    ofs << summary_to_json(summary).dump(2) << "\n";
}

void write_comparison_json(const std::vector<BacktestResult>& results,
                           const std::string& path) {
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        throw std::runtime_error("write_comparison_json: cannot open " + path);
    }

    nlohmann::json j = nlohmann::json::array();
    for (const auto& res : results) {
        j.push_back(summary_to_json(res.summary));
    }
    ofs << j.dump(2) << "\n";
}

void write_comparison_csv(const std::vector<BacktestResult>& results,
                          const std::string& path) {
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        throw std::runtime_error("write_comparison_csv: cannot open " + path);
    }

    ofs << std::fixed << std::setprecision(6);
    ofs << "strategy,total_return,annualized_return,annualized_vol,"
        << "sharpe,sortino,max_drawdown,calmar,"
        << "total_txn_cost,avg_turnover,n_rebalances,n_days\n";

    for (const auto& res : results) {
        const auto& s = res.summary;
        ofs << s.strategy_name << ","
            << s.total_return << ","
            << s.annualized_return << ","
            << s.annualized_volatility << ","
            << s.sharpe_ratio << ","
            << s.sortino_ratio << ","
            << s.max_drawdown << ","
            << s.calmar_ratio << ","
            << s.total_transaction_cost << ","
            << s.avg_turnover << ","
            << s.n_rebalances << ","
            << s.n_days << "\n";
    }
}

// ── Optimization reporting ─────────────────────────────────────────

void write_frontier_csv(const std::vector<FrontierPoint>& frontier,
                        const std::vector<std::string>& tickers,
                        const std::string& path) {
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        throw std::runtime_error("write_frontier_csv: cannot open " + path);
    }

    ofs << std::fixed << std::setprecision(8);

    // Header.
    ofs << "target_return,achieved_return,cvar,zeta,converged,iterations";
    for (const auto& t : tickers) {
        ofs << "," << t;
    }
    ofs << "\n";

    for (const auto& pt : frontier) {
        ofs << pt.target_return << ","
            << pt.achieved_return << ","
            << pt.cvar << ","
            << pt.zeta << ","
            << (pt.converged ? 1 : 0) << ","
            << pt.iterations;
        for (Index i = 0; i < static_cast<Index>(pt.weights.size()); ++i) {
            ofs << "," << pt.weights(i);
        }
        ofs << "\n";
    }
}

void write_optimize_result_json(const AdmmResult& result,
                                const VectorXd& mu,
                                const std::vector<std::string>& tickers,
                                const std::string& path) {
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        throw std::runtime_error("write_optimize_result_json: cannot open " + path);
    }

    nlohmann::json j;
    j["converged"] = result.converged;
    j["iterations"] = result.iterations;
    j["cvar"] = result.cvar;
    j["expected_return"] = result.expected_return;
    j["zeta"] = result.zeta;

    // Weights as object {ticker: weight}.
    nlohmann::json w_obj = nlohmann::json::object();
    for (size_t i = 0; i < tickers.size(); ++i) {
        w_obj[tickers[i]] = result.weights(static_cast<Index>(i));
    }
    j["weights"] = w_obj;

    // Weights as array (for easier parsing).
    std::vector<double> w_arr(result.weights.data(),
                               result.weights.data() + result.weights.size());
    j["weights_array"] = w_arr;

    // Input mu.
    std::vector<double> mu_arr(mu.data(), mu.data() + mu.size());
    j["mu"] = mu_arr;
    j["tickers"] = tickers;

    ofs << j.dump(2) << "\n";
}

void write_optimize_result_json(const AdmmResult& result,
                                const VectorXd& mu,
                                const std::vector<std::string>& tickers,
                                const VectorXd& component_cvar,
                                const std::string& path) {
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        throw std::runtime_error("write_optimize_result_json: cannot open " + path);
    }

    nlohmann::json j;
    j["converged"] = result.converged;
    j["iterations"] = result.iterations;
    j["cvar"] = result.cvar;
    j["expected_return"] = result.expected_return;
    j["zeta"] = result.zeta;

    // Weights as object {ticker: weight}.
    nlohmann::json w_obj = nlohmann::json::object();
    for (size_t i = 0; i < tickers.size(); ++i) {
        w_obj[tickers[i]] = result.weights(static_cast<Index>(i));
    }
    j["weights"] = w_obj;

    // Weights as array.
    std::vector<double> w_arr(result.weights.data(),
                               result.weights.data() + result.weights.size());
    j["weights_array"] = w_arr;

    // Input mu.
    std::vector<double> mu_arr(mu.data(), mu.data() + mu.size());
    j["mu"] = mu_arr;
    j["tickers"] = tickers;

    // Component CVaR as object {ticker: value}.
    nlohmann::json cc_obj = nlohmann::json::object();
    for (size_t i = 0; i < tickers.size(); ++i) {
        cc_obj[tickers[i]] = component_cvar(static_cast<Index>(i));
    }
    j["component_cvar"] = cc_obj;

    // Component CVaR as array.
    std::vector<double> cc_arr(component_cvar.data(),
                                component_cvar.data() + component_cvar.size());
    j["component_cvar_array"] = cc_arr;

    ofs << j.dump(2) << "\n";
}

void write_risk_decomposition_csv(const VectorXd& weights,
                                   const VectorXd& component_cvar,
                                   const std::vector<std::string>& tickers,
                                   const std::string& path) {
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        throw std::runtime_error("write_risk_decomposition_csv: cannot open " + path);
    }

    ofs << std::fixed << std::setprecision(8);
    ofs << "ticker,weight,component_cvar,pct_contribution\n";

    double total_cvar = component_cvar.sum();

    for (size_t i = 0; i < tickers.size(); ++i) {
        Index idx = static_cast<Index>(i);
        double pct = (std::abs(total_cvar) > 1e-15)
                         ? component_cvar(idx) / total_cvar * 100.0
                         : 0.0;
        ofs << tickers[i] << ","
            << weights(idx) << ","
            << component_cvar(idx) << ","
            << pct << "\n";
    }
}

}  // namespace cpo
