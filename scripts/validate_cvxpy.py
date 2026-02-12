#!/usr/bin/env python3
"""Cross-validation of ADMM solver against cvxpy.

Generates test cases with known mu/Sigma, solves the Rockafellar-Uryasev
CVaR minimization with cvxpy + ECOS, and writes reference JSON files to
tests/data/validation/ for the C++ test_validation to compare against.

Usage:
    python scripts/validate_cvxpy.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

try:
    import cvxpy as cp
except ImportError:
    print("ERROR: cvxpy not installed. Run: pip install -r scripts/requirements.txt")
    sys.exit(1)


OUTPUT_DIR = Path(__file__).resolve().parent.parent / "tests" / "data" / "validation"


def generate_scenarios(mu, cov, n_scenarios, seed):
    """Generate Monte Carlo scenarios: r = mu + L @ z, z ~ N(0,I)."""
    rng = np.random.default_rng(seed)
    n_assets = len(mu)
    L = np.linalg.cholesky(cov)
    Z = rng.standard_normal((n_scenarios, n_assets))
    scenarios = Z @ L.T + mu[np.newaxis, :]
    return scenarios


def solve_cvar_cvxpy(scenarios, mu_vec, alpha, w_min=None, w_max=None,
                     target_return=None):
    """Solve Mean-CVaR via Rockafellar-Uryasev formulation with cvxpy.

    min_{w, zeta}  zeta + 1/(N*alpha) * sum_i max(0, -r_i'w - zeta)
    s.t.  1'w = 1, w >= 0
          w_min <= w <= w_max  (optional)
          mu'w >= target_return (optional)

    References:
        Rockafellar & Uryasev, "Optimization of Conditional Value-at-Risk",
        Journal of Risk, 2000 -- Eq. (9)-(10).
    """
    n_scenarios, n_assets = scenarios.shape

    w = cp.Variable(n_assets)
    zeta = cp.Variable()

    # Portfolio losses per scenario: loss_i = -r_i' w
    losses = -scenarios @ w  # (n_scenarios,)

    # R-U auxiliary: u_i = max(0, loss_i - zeta)
    excess = cp.maximum(losses - zeta, 0)

    objective = cp.Minimize(zeta + (1.0 / (n_scenarios * alpha)) * cp.sum(excess))

    constraints = [
        cp.sum(w) == 1,
        w >= 0,
    ]

    if w_min is not None:
        constraints.append(w >= w_min)
    if w_max is not None:
        constraints.append(w <= w_max)
    if target_return is not None:
        constraints.append(mu_vec @ w >= target_return)

    prob = cp.Problem(objective, constraints)

    # Try ECOS first (exact for SOC), fall back to SCS.
    try:
        prob.solve(solver=cp.ECOS, max_iters=500, abstol=1e-8, reltol=1e-8)
    except (cp.SolverError, Exception):
        prob.solve(solver=cp.SCS, max_iters=10000)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        return None

    return {
        "weights": w.value.tolist(),
        "cvar": float(prob.value),
        "zeta": float(zeta.value),
        "expected_return": float(mu_vec @ w.value),
        "status": prob.status,
    }


def make_test_case(name, n_assets, n_scenarios, alpha, seed, w_max=None,
                   target_return=None):
    """Build one test case: generate data, solve, write JSON."""
    # Synthetic mu: linearly spaced returns.
    mu = np.linspace(0.02, 0.02 + 0.03 * (n_assets - 1), n_assets)

    # Synthetic covariance: constant correlation 0.3, vol = 0.20.
    vol = 0.20
    corr = np.full((n_assets, n_assets), 0.3)
    np.fill_diagonal(corr, 1.0)
    D = np.diag(np.full(n_assets, vol))
    cov = D @ corr @ D

    scenarios = generate_scenarios(mu, cov, n_scenarios, seed)

    w_min_arr = None
    w_max_arr = None
    if w_max is not None:
        w_min_arr = np.zeros(n_assets)
        w_max_arr = np.full(n_assets, w_max)

    result = solve_cvar_cvxpy(
        scenarios, mu, alpha,
        w_min=w_min_arr, w_max=w_max_arr,
        target_return=target_return,
    )

    if result is None:
        print(f"  SKIPPED {name}: solver failed")
        return None

    # Build constraint description for JSON.
    constraints = {}
    if w_max is not None:
        constraints["w_max"] = w_max
    if target_return is not None:
        constraints["target_return"] = target_return

    case = {
        "name": name,
        "n_assets": n_assets,
        "n_scenarios": n_scenarios,
        "alpha": alpha,
        "seed": seed,
        "mu": mu.tolist(),
        "covariance": cov.tolist(),
        "constraints": constraints,
        "cvxpy_result": result,
    }

    return case


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    test_cases = [
        # Unconstrained (simplex + non-negativity only).
        ("2asset_unconstrained", 2, 20000, 0.05, 42, None, None),
        ("5asset_unconstrained", 5, 20000, 0.05, 42, None, None),
        ("10asset_unconstrained", 10, 30000, 0.05, 42, None, None),

        # Box constraints (40% max per position).
        ("5asset_box40", 5, 20000, 0.05, 42, 0.40, None),
        ("10asset_box40", 10, 30000, 0.05, 42, 0.40, None),

        # Target return constraint.
        ("5asset_target", 5, 20000, 0.05, 42, None, 0.04),
        ("10asset_target", 10, 30000, 0.05, 42, None, 0.04),

        # Combined: box + target return.
        ("5asset_combined", 5, 20000, 0.05, 42, 0.40, 0.04),
        ("10asset_combined", 10, 30000, 0.05, 42, 0.40, 0.04),
    ]

    for name, n_assets, n_scenarios, alpha, seed, w_max, target_ret in test_cases:
        print(f"Solving {name} ({n_assets} assets, {n_scenarios} scenarios)...")
        case = make_test_case(name, n_assets, n_scenarios, alpha, seed,
                              w_max=w_max, target_return=target_ret)
        if case is not None:
            out_path = OUTPUT_DIR / f"{name}.json"
            with open(out_path, "w") as f:
                json.dump(case, f, indent=2)
            w = case["cvxpy_result"]["weights"]
            cvar = case["cvxpy_result"]["cvar"]
            print(f"  -> CVaR={cvar:.6f}, weights={[f'{x:.4f}' for x in w]}")
            print(f"  -> Wrote {out_path}")

    print(f"\nDone. {len(test_cases)} reference files in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
