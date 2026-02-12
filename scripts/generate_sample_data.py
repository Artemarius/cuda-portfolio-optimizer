#!/usr/bin/env python3
"""Generate synthetic price data for demo/testing purposes.

Creates a 252-day x 5-asset price CSV using Geometric Brownian Motion (GBM):
    dS/S = mu*dt + sigma*dW

Usage:
    python scripts/generate_sample_data.py
"""

import os
from pathlib import Path

import numpy as np

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_prices.csv"

# Asset definitions.
TICKERS = ["ALPHA", "BETA", "GAMMA", "DELTA", "EPSILON"]
N_DAYS = 252  # One trading year.
SEED = 42

# Annualized parameters (realistic but synthetic).
MU = np.array([0.08, 0.10, 0.12, 0.06, 0.14])      # Expected returns.
SIGMA = np.array([0.15, 0.20, 0.25, 0.12, 0.30])    # Volatilities.

# Correlation matrix.
CORR = np.array([
    [1.00, 0.40, 0.30, 0.20, 0.25],
    [0.40, 1.00, 0.50, 0.15, 0.35],
    [0.30, 0.50, 1.00, 0.10, 0.45],
    [0.20, 0.15, 0.10, 1.00, 0.05],
    [0.25, 0.35, 0.45, 0.05, 1.00],
])

INITIAL_PRICES = np.array([100.0, 200.0, 150.0, 50.0, 75.0])


def generate_gbm_prices(n_days, mu, sigma, corr, initial_prices, seed):
    """Generate correlated GBM price paths.

    GBM discrete approximation:
        S_{t+1} = S_t * exp((mu - sigma^2/2)*dt + sigma*sqrt(dt)*Z_t)
    where Z_t ~ N(0, Sigma) with correlation structure.
    """
    rng = np.random.default_rng(seed)
    n_assets = len(mu)
    dt = 1.0 / 252.0  # Daily.

    # Cholesky for correlated normals.
    D = np.diag(sigma)
    cov = D @ corr @ D
    L = np.linalg.cholesky(cov)

    prices = np.zeros((n_days + 1, n_assets))
    prices[0] = initial_prices

    for t in range(n_days):
        z = rng.standard_normal(n_assets)
        corr_z = L @ z
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = corr_z * np.sqrt(dt)
        prices[t + 1] = prices[t] * np.exp(drift + diffusion)

    return prices


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    prices = generate_gbm_prices(N_DAYS, MU, SIGMA, CORR, INITIAL_PRICES, SEED)

    # Generate business dates (skip weekends).
    from datetime import date, timedelta
    start = date(2024, 1, 2)  # First trading day of 2024.
    dates = []
    d = start
    while len(dates) < len(prices):
        if d.weekday() < 5:  # Mon-Fri.
            dates.append(d.isoformat())
        d += timedelta(days=1)

    with open(OUTPUT_PATH, "w") as f:
        f.write("Date," + ",".join(TICKERS) + "\n")
        for i, row in enumerate(prices):
            f.write(dates[i])
            for p in row:
                f.write(f",{p:.4f}")
            f.write("\n")

    print(f"Generated {len(prices)} rows x {len(TICKERS)} assets")
    print(f"Date range: {dates[0]} to {dates[-1]}")
    print(f"Written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
