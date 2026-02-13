#include "models/shrinkage_estimator.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <spdlog/spdlog.h>

namespace cpo {

ShrinkageResult ledoit_wolf_shrink(const MatrixXd& returns) {
    const Index T = static_cast<Index>(returns.rows());
    const Index N = static_cast<Index>(returns.cols());

    if (T < 2) {
        throw std::invalid_argument(
            "ledoit_wolf_shrink: need T >= 2, got T=" + std::to_string(T));
    }
    if (N < 1) {
        throw std::invalid_argument(
            "ledoit_wolf_shrink: need N >= 1, got N=" + std::to_string(N));
    }

    // Center returns: X = returns - mean.
    VectorXd mu = returns.colwise().mean();
    MatrixXd X = returns.rowwise() - mu.transpose();

    // Sample covariance: S = (1/(T-1)) * X' * X.
    const double n_obs = static_cast<double>(T);
    const double denom = n_obs - 1.0;
    MatrixXd S = (X.transpose() * X) / denom;

    // Shrinkage target: mu_hat * I, where mu_hat = trace(S) / N.
    // Ledoit & Wolf (2004), Eq. (2).
    const double mu_hat = S.trace() / static_cast<double>(N);

    // delta: squared Frobenius distance from S to target, normalized by N.
    // delta = ||S - mu_hat * I||_F^2 / N
    // Ledoit & Wolf (2004), Lemma 3.1.
    MatrixXd S_minus_target = S - mu_hat * MatrixXd::Identity(N, N);
    const double delta = S_minus_target.squaredNorm() / static_cast<double>(N);

    // beta_bar: sample variance of the entries of S.
    // beta_bar = (1/(T*N)) * sum_t ||x_t * x_t' - S||_F^2
    // where x_t is the t-th centered observation (row of X).
    // Ledoit & Wolf (2004), Lemma 3.1, consistent estimator of beta.
    //
    // Efficient computation: for each observation t, compute x_t * x_t'
    // (rank-1 outer product), subtract S, and accumulate the squared norm.
    double beta_sum = 0.0;
    for (Index t = 0; t < T; ++t) {
        // x_t is a column vector (N x 1).
        VectorXd x_t = X.row(t).transpose();
        // Outer product minus sample covariance.
        MatrixXd diff = x_t * x_t.transpose() - S;
        beta_sum += diff.squaredNorm();
    }
    const double beta_bar = beta_sum / (n_obs * n_obs * static_cast<double>(N));

    // Optimal shrinkage intensity: clamp beta_bar to [0, delta].
    // intensity = min(beta_bar, delta) / delta.
    // Ledoit & Wolf (2004), Theorem 1.
    double intensity;
    if (delta < 1e-30) {
        // S is already (close to) a scaled identity — no shrinkage needed.
        intensity = 0.0;
    } else {
        const double beta = std::min(beta_bar, delta);
        intensity = beta / delta;
    }

    // Clamp to [0, 1] for numerical safety.
    intensity = std::clamp(intensity, 0.0, 1.0);

    // Shrunk covariance: S_shrunk = intensity * mu_hat * I + (1 - intensity) * S.
    MatrixXd shrunk = (1.0 - intensity) * S
                      + intensity * mu_hat * MatrixXd::Identity(N, N);

    spdlog::debug("Ledoit-Wolf shrinkage: T={}, N={}, intensity={:.4f}, "
                  "mu_hat={:.6f}, delta={:.6f}, beta_bar={:.6f}",
                  T, N, intensity, mu_hat, delta, beta_bar);

    return ShrinkageResult{std::move(shrunk), intensity, std::move(S)};
}

}  // namespace cpo
