#pragma once

/// @file risk_result.h
/// @brief Configuration and result structs for risk computation.
///
/// All result fields use ScalarCPU (double) per the dual-precision convention:
///   GPU computes in float, results are promoted to double for reporting.

#include "core/types.h"

namespace cpo {

/// Configuration for risk computation (CVaR, VaR, statistics).
struct RiskConfig {
    ScalarCPU confidence_level = 0.95;  ///< Alpha for VaR/CVaR (e.g., 0.95 = 95%).
    int threads_per_block = 256;
};

/// Portfolio risk metrics computed from a loss distribution.
///
/// VaR and CVaR follow the Rockafellar-Uryasev convention:
///   VaR_alpha = the alpha-quantile of the loss distribution
///   CVaR_alpha = E[L | L >= VaR_alpha]  (average of worst (1-alpha) fraction)
///
/// Reference: Rockafellar & Uryasev, "Optimization of Conditional
///   Value-at-Risk", J. Risk 2000.
struct RiskResult {
    ScalarCPU var = 0.0;               ///< Value-at-Risk at alpha.
    ScalarCPU cvar = 0.0;              ///< Conditional Value-at-Risk at alpha.
    ScalarCPU expected_return = 0.0;   ///< E[w'r] = mean of portfolio returns.
    ScalarCPU volatility = 0.0;        ///< std(w'r) — portfolio return volatility.
    ScalarCPU sharpe_ratio = 0.0;      ///< expected_return / volatility.
    ScalarCPU sortino_ratio = 0.0;     ///< expected_return / downside_deviation.
    ScalarCPU confidence_level = 0.0;  ///< Alpha used for VaR/CVaR.
    Index n_scenarios = 0;             ///< Number of scenarios used.
};

}  // namespace cpo
