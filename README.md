# cuda-portfolio-optimizer

GPU-accelerated portfolio optimization using Monte Carlo simulation and Mean-CVaR, with a custom ADMM solver implemented in C++17/CUDA and a rolling-window backtesting engine.

## Motivation

Classical portfolio optimization (Markowitz mean-variance) has well-known limitations: sensitivity to estimation error, Gaussian return assumptions, and variance as a poor proxy for tail risk. CVaR (Conditional Value-at-Risk) addresses the tail risk problem, but the scenario-based formulation is computationally expensive -- you're solving over tens of thousands of Monte Carlo scenarios. This maps naturally to GPU parallelism: scenario generation is embarrassingly parallel, and the ADMM solver's per-iteration bottleneck is a matrix-vector product across all scenarios.

This project implements the full pipeline from scratch: return estimation, correlated Monte Carlo simulation on GPU, CVaR optimization via ADMM, efficient frontier computation, and backtesting with transaction costs.

## Architecture

```
                 ┌─────────────┐
                 │  Price CSV   │
                 └──────┬───────┘
                        │
                 ┌──────▼───────┐
                 │  Returns &   │
                 │  Covariance  │
                 └──────┬───────┘
                        │
                 ┌──────▼───────┐     ┌─────────────────┐
                 │   Cholesky   │────►│  L (lower tri)   │
                 │ Decomposition│     └────────┬─────────┘
                 └──────────────┘              │
                                        ┌──────▼───────┐
                                        │  Monte Carlo  │  GPU: cuRAND + custom kernel
                                        │  Scenarios    │  r = mu + L*z, z ~ N(0,I)
                                        └──────┬───────┘
                                               │
                         ┌─────────────────────┬┘
                         │                     │
                  ┌──────▼───────┐     ┌───────▼──────┐
                  │ ADMM Solver  │     │  Backtester   │
                  │ (Mean-CVaR)  │     │ (rolling win) │
                  └──────┬───────┘     └───────┬───────┘
                         │                     │
                  ┌──────▼───────┐     ┌───────▼──────┐
                  │  Efficient   │     │  Equity Curve │
                  │  Frontier    │     │  + Comparison │
                  └──────┬───────┘     └───────┬───────┘
                         │                     │
                  ┌──────▼─────────────────────▼──────┐
                  │        CSV / JSON Reports          │
                  └────────────────────────────────────┘
```

```
src/
  core/          Fundamental types, config, portfolio result structs
  data/          Market data loader (CSV), return computation, universe definition
  simulation/    GPU Monte Carlo scenario generator (correlated returns via Cholesky + cuRAND)
  risk/          CVaR computation (CUDA), VaR, volatility, drawdown metrics
  optimizer/     ADMM solver (C++/CUDA), projections, efficient frontier
  constraints/   Portfolio constraints: position limits, leverage, turnover, sector
  backtest/      Rolling-window backtesting engine with transaction costs
  reporting/     Efficient frontier, risk decomposition, strategy comparison (CSV/JSON)
  utils/         Timer, logging, CUDA helpers
apps/            CLI executables (optimize, backtest)
tests/           Google Test unit tests (137 tests)
benchmarks/      GPU vs CPU performance comparison (Google Benchmark)
scripts/         Python helpers: cvxpy validation, data generation, plotting
```

## What's Implemented

**Monte Carlo Scenario Generation (CUDA)** -- Covariance estimation from historical returns, Cholesky decomposition (CPU), correlated sample generation on GPU via cuRAND. Column-major scenario matrix layout for coalesced memory access.

**Risk Computation (CUDA)** -- Portfolio loss computation: one CUDA thread per scenario. VaR and CVaR via GPU-accelerated sort (CUB) + reduction.

**ADMM Optimizer (C++/CUDA)** -- Rockafellar-Uryasev formulation of Mean-CVaR. ADMM with proximal gradient x-update (GPU-accelerated), Dykstra's alternating projection for constraint handling. Supports position limits, turnover, and sector exposure bounds.

**Efficient Frontier** -- Target-return sweep with warm-starting between solves.

**Backtesting Engine** -- Rolling-window rebalancing on historical data. Proportional transaction costs with minimum trade threshold. Strategy comparison: Mean-Variance, Mean-CVaR, Equal-Weight (1/N), Risk Parity.

## Performance (Measured on RTX 3060)

### Monte Carlo Scenario Generation

| Configuration | GPU | CPU | Speedup |
|---|---|---|---|
| 10K scenarios x 50 assets | 31 ms | 32 ms | 1.0x |
| 50K scenarios x 100 assets | 32 ms | 400 ms | **12.5x** |
| 100K scenarios x 100 assets | 99 ms | 722 ms | **7.3x** |
| 100K scenarios x 500 assets | 1098 ms | -- | -- |

GPU advantage grows with problem size. At 50K x 100, GPU is 12.5x faster. The 10K x 50 case is too small to offset kernel launch overhead.

### ADMM Optimization (CPU path, converged)

| Assets | Scenarios | Time | Iterations |
|---|---|---|---|
| 2 | 10K | 57 ms | 65 |
| 5 | 10K | 55 ms | 46 |
| 10 | 10K | 64 ms | 40 |
| 5 | 50K | 286 ms | 42 |
| 10 | 50K | 406 ms | 44 |

### Efficient Frontier (5 points, 20K scenarios)

| Assets | Time | Total Iterations |
|---|---|---|
| 3 | 414 ms | 200 |
| 5 | 771 ms | 310 |
| 10 | 1,233 ms | 375 |

### Full Pipeline (scenario generation + ADMM solve)

| Assets | Scenarios | Time |
|---|---|---|
| 5 | 50K | 324 ms |
| 10 | 50K | 493 ms |

### VRAM Usage (RTX 3060, 6 GB)

| Component | 100K x 500 |
|---|---|
| Scenario matrix (float32) | 191 MB |
| cuRAND states | 5 MB |
| Cholesky factor | 1 MB |
| ADMM working buffers | ~200 MB |
| **Total** | **~397 MB** (6.5% of 6 GB) |

## Building

### Prerequisites
- NVIDIA GPU (compute capability 7.0+)
- CUDA Toolkit 12.0+
- C++17 compiler (MSVC 2022, GCC 9+, Clang 10+)
- CMake 3.20+

### Build

```bash
git clone https://github.com/<username>/cuda-portfolio-optimizer.git
cd cuda-portfolio-optimizer

# Visual Studio generator (Windows)
cmake -B build -G "Visual Studio 17 2022"
cmake --build build --config Release

# Or Ninja (faster incremental builds)
cmake -B build -DCMAKE_BUILD_TYPE=Release -G Ninja
cmake --build build
```

All dependencies (Eigen, nlohmann/json, spdlog, Google Test, Google Benchmark) are fetched automatically via CMake FetchContent. Only CUDA must be pre-installed.

### Test

```bash
ctest --test-dir build -C Release --output-on-failure
# 137 tests, all passing
```

### Run

```bash
# Portfolio optimization (efficient frontier)
./build/Release/optimize --config config/optimize_5asset.json --output results/optimize/

# Backtest with strategy comparison
./build/Release/backtest --config config/backtest_5asset.json --output results/backtest/

# Benchmarks
./build/Release/bench_monte_carlo
./build/Release/bench_cvar
./build/Release/bench_admm
```

## Example: Efficient Frontier

```bash
./build/Release/optimize --config config/optimize_5asset.json --output results/
```

Output:
```
Efficient Frontier (10 points)
  Target Ret   Achieved Ret         CVaR  Iters  Conv
    0.020000     0.030548     0.244036     42   yes
    0.024444     0.030548     0.244036     17   yes
    0.028889     0.030548     0.244036     15   yes
    0.033333     0.033317     0.247964     80   yes
    0.037778     0.034867     0.257314    142   yes
    ...
```

Results are written to `results/frontier.csv` and `results/frontier_result.json`.

## Example: Backtest

```bash
# Generate synthetic price data first
python scripts/generate_sample_data.py

./build/Release/backtest --config config/backtest_5asset.json --output results/backtest/
```

Outputs equity curves, weights, and strategy comparison (CSV + JSON) for all four strategies.

## Validation

Optimization results are validated at multiple levels:

1. **Analytical** -- 2-asset closed-form efficient frontier (Markowitz)
2. **Cross-reference** -- ADMM results compared against Python cvxpy + ECOS solver. Weight L-inf difference < 0.05 for small problems, CVaR relative difference < 10%
3. **Statistical** -- Monte Carlo convergence tests (sample moments approach true parameters as N increases)
4. **Structural** -- Efficient frontier monotonicity, CVaR >= VaR, constraint satisfaction

Cross-validation workflow:
```bash
pip install -r scripts/requirements.txt
python scripts/validate_cvxpy.py    # Generates reference JSON
ctest --test-dir build -C Release   # C++ test loads and compares
```

## The Math

**CVaR** measures the average loss in the worst alpha% of outcomes. For a portfolio with weights w and scenario returns r_i:

- VaR_alpha = inf{ zeta : P(-r'w <= zeta) >= alpha }
- CVaR_alpha = E[-r'w | -r'w >= VaR_alpha]

The **Rockafellar-Uryasev** reformulation makes CVaR minimization tractable:

```
min_{w,zeta}  zeta + 1/(N*alpha) * sum_i max(0, -r_i'w - zeta)
s.t.  mu'w >= target_return,  1'w = 1,  w >= 0,  constraints
```

**ADMM** decomposes the constrained problem:
1. **x-update**: proximal gradient on augmented Lagrangian (scenarios -- GPU-acceleratable)
2. **z-update**: project onto constraint set via Dykstra's alternating projections
3. **u-update**: dual variable gradient ascent

**Correlated simulation**: given Sigma = LL' (Cholesky), generate r = mu + Lz where z ~ N(0,I).

## Dependencies

| Library | Purpose | Acquisition |
|---|---|---|
| CUDA 12+ | Compute kernels, cuRAND, CUB | Pre-installed |
| Eigen3 | CPU linear algebra, Cholesky | FetchContent |
| nlohmann/json | Config and output serialization | FetchContent |
| spdlog | Logging | FetchContent |
| Google Test | Unit tests | FetchContent |
| Google Benchmark | CPU vs GPU benchmarks | FetchContent |

## References

1. Rockafellar & Uryasev, [Optimization of Conditional Value-at-Risk](https://doi.org/10.21314/JOR.2000.038), Journal of Risk 2000
2. Boyd et al., [Distributed Optimization and Statistical Learning via ADMM](https://stanford.edu/~boyd/admm.html), Found. & Trends in ML 2011
3. Ledoit & Wolf, [Honey, I Shrunk the Sample Covariance Matrix](http://www.ledoit.net/honey.pdf), 2004
4. Duchi et al., [Efficient Projections onto the l1-Ball](https://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf), ICML 2008
5. Markowitz, Portfolio Selection, Journal of Finance 1952

## License

MIT
