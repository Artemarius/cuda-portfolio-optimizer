# cuda-portfolio-optimizer

GPU-accelerated portfolio optimization using Monte Carlo simulation and Mean-CVaR, with a custom ADMM solver implemented in C++17/CUDA and a rolling-window backtesting engine.

## Motivation

Classical portfolio optimization (Markowitz mean-variance) has well-known limitations: sensitivity to estimation error, Gaussian return assumptions, and variance as a poor proxy for tail risk. CVaR (Conditional Value-at-Risk) addresses the tail risk problem, but the scenario-based formulation is computationally expensive — you're solving a linear program over tens of thousands of Monte Carlo scenarios. This maps naturally to GPU parallelism: scenario generation is embarrassingly parallel, and the ADMM solver's expensive per-iteration step is a matrix-vector product across all scenarios.

This project implements the full pipeline from scratch: return estimation → correlated Monte Carlo simulation on GPU → CVaR optimization via ADMM → backtesting with transaction costs.

## What's Implemented

**Monte Carlo Scenario Generation (CUDA)**
- Covariance estimation from historical returns (sample + Ledoit-Wolf shrinkage)
- Cholesky decomposition (CPU) → correlated sample generation on GPU via cuRAND
- Column-major scenario matrix layout for coalesced GPU memory access
- Optional factor model decomposition (R = Bf + ε) for large universes

**Risk Computation (CUDA)**
- Portfolio loss computation: one CUDA thread per scenario
- VaR and CVaR via GPU-accelerated sort (CUB) + reduction
- Additional metrics: volatility, Sharpe, Sortino, maximum drawdown

**Convex Optimizer (C++/CUDA)**
- Rockafellar-Uryasev formulation: reformulates Mean-CVaR as a linear program
  ```
  min_{w,ζ}  ζ + 1/(Nα) Σᵢ max(0, -rᵢᵀw - ζ)
  s.t.  μᵀw ≥ target_return,  1ᵀw = 1,  w_min ≤ w ≤ w_max,  ‖w - w_prev‖₁ ≤ τ
  ```
- ADMM (Alternating Direction Method of Multipliers) solver with GPU-accelerated x-update
- Constraint handling: position limits, leverage, turnover, sector exposure

**Backtesting Engine**
- Rolling-window rebalancing on historical data
- Transaction cost model (proportional, with minimum trade threshold)
- Strategy comparison: Mean-Variance, Mean-CVaR, Equal-Weight (1/N), Risk Parity
- Output: equity curves, drawdown, rolling Sharpe, turnover, risk decomposition

## The Math

**CVaR** measures the average loss in the worst α% of outcomes. Unlike VaR (a single quantile), CVaR is coherent, convex, and captures tail risk. For a portfolio with weights w and scenario returns rᵢ:

- VaR_α = inf{ ζ : P(-rᵀw ≤ ζ) ≥ α }
- CVaR_α = E[-rᵀw | -rᵀw ≥ VaR_α]

The Rockafellar-Uryasev insight is that CVaR minimization can be reformulated as a linear program by introducing auxiliary variables, making it tractable for convex solvers.

**ADMM** decomposes the constrained problem into alternating steps:
1. x-update: minimize augmented Lagrangian (involves scenario matrix — GPU-accelerated)
2. z-update: project onto constraint set (box constraints, simplex — cheap, CPU)
3. Dual update: gradient ascent on dual variables

**Correlated simulation**: given covariance Σ = LLᵀ (Cholesky), generate scenarios as r = μ + Lz where z ~ N(0,I). Each scenario is independent → embarrassingly parallel on GPU.

## Performance

*Targets:*

| Operation | GPU | CPU | Speedup |
|---|---|---|---|
| Monte Carlo (100K × 500 assets) | < 100 ms | ~10 s | ~100x |
| CVaR (100K scenarios) | < 10 ms | ~500 ms | ~50x |
| Full optimization (single solve) | < 1 s | — | — |
| Full backtest (10yr, monthly) | < 5 min | — | — |

## Building

### Prerequisites
- NVIDIA GPU with compute capability 7.0+
- CUDA Toolkit 12.0+
- C++17 compiler (GCC 9+, Clang 10+)
- CMake 3.20+

### Build
```bash
git clone https://github.com/<username>/cuda-portfolio-optimizer.git
cd cuda-portfolio-optimizer

cmake -B build -DCMAKE_BUILD_TYPE=Release -G Ninja
cmake --build build -- -j$(nproc)
```

### Run
```bash
# Portfolio optimization
./build/optimize --config config/sp500_cvar.json --output results/

# Backtest with strategy comparison
./build/backtest --config config/backtest_rolling.json --output results/backtest/

# GPU vs CPU benchmark
./build/bench_montecarlo
```

## Project Structure

```
src/
  data/        — Market data loader, return computation
  models/      — Return distributions: historical, factor, shrinkage estimators
  simulation/  — CUDA Monte Carlo scenario generator
  risk/        — CVaR/VaR computation (CUDA), volatility, drawdown
  optimizer/   — ADMM solver (C++/CUDA), constraint handling
  backtest/    — Rolling-window backtester, transaction costs
  reporting/   — Efficient frontier, risk decomposition, CSV/JSON output
  utils/       — Timer, logging, matrix helpers
tests/         — Google Test (validated against cvxpy/scipy)
benchmarks/    — GPU vs CPU comparison
```

## Dependencies

| Library | Purpose |
|---|---|
| CUDA 12+ | Compute kernels, cuRAND |
| CUB | GPU sort and reduction primitives |
| Eigen3 | CPU linear algebra, Cholesky, covariance |
| nlohmann/json | Config, output |
| Google Test | Unit tests |
| Google Benchmark | CPU vs GPU benchmarks |
| spdlog | Logging |

## Validation

Optimization results are validated against known solutions:
- 2-asset analytical case (closed-form efficient frontier)
- Small problems (5-10 assets) cross-checked against Python cvxpy + scipy
- Efficient frontier monotonicity and constraint satisfaction checks
- Monte Carlo convergence tests (increasing N should reduce variance)

## References

1. Rockafellar & Uryasev, [Optimization of Conditional Value-at-Risk](https://doi.org/10.21314/JOR.2000.038), Journal of Risk 2000
2. Boyd et al., [Multi-Period Trading via Convex Optimization](https://web.stanford.edu/~boyd/papers/cvx_portfolio.html), Found. & Trends in Optimization 2017
3. Boyd et al., [Distributed Optimization and Statistical Learning via ADMM](https://stanford.edu/~boyd/admm.html), Found. & Trends in ML 2011
4. Ledoit & Wolf, [Honey, I Shrunk the Sample Covariance Matrix](http://www.ledoit.net/honey.pdf), 2004
5. Markowitz, Portfolio Selection, Journal of Finance 1952

## License

MIT
