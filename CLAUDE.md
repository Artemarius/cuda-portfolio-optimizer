# CLAUDE.md

## Architecture

```
src/
  data/          — Market data loader (CSV/Parquet), return computation, universe definition
  models/        — Return distribution models: historical, factor model, parametric
  simulation/    — GPU Monte Carlo scenario generator (correlated returns via Cholesky + cuRAND)
  risk/          — CVaR computation (CUDA), VaR, volatility, drawdown metrics
  optimizer/     — Convex optimizer: ADMM solver (C++/CUDA)
  constraints/   — Portfolio constraints: position limits, leverage, turnover, sector
  backtest/      — Rolling-window backtesting engine with transaction costs
  reporting/     — Efficient frontier, risk decomposition, strategy comparison (CSV/JSON output)
  utils/         — Timer, logging, math helpers, matrix utilities
tests/           — Google Test unit tests
benchmarks/      — GPU vs CPU performance comparison
```

## Key Technical Decisions

- **C++17** with CUDA 12+
- **Custom CUDA kernels** for Monte Carlo simulation and CVaR computation — no cuOpt or external solver libraries
- **Eigen3** for CPU-side linear algebra (covariance matrices, Cholesky decomposition)
- **cuRAND** for GPU random number generation
- **ADMM solver** implemented in C++/CUDA for the constrained Mean-CVaR optimization

## Code Style & Conventions

- Google C++ Style Guide baseline:
  - `snake_case` for functions and variables, `PascalCase` for types/classes
  - RAII everywhere, no raw `new`/`delete`
  - `constexpr` where possible
- CUDA kernels: prefix with `k_` (e.g., `k_monte_carlo_simulate`, `k_compute_cvar`)
- All public APIs must have doc comments
- Every module must have unit tests
- Mathematical formulas must be referenced in comments (paper + equation number when applicable)

## Build & Run

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -G Ninja
cmake --build build -- -j$(nproc)

# Run tests
ctest --test-dir build --output-on-failure

# Portfolio optimization
./build/optimize --config config/sp500_cvar.json --output results/

# Backtest
./build/backtest --config config/backtest_rolling.json --output results/backtest/

# GPU vs CPU benchmarks
./build/bench_montecarlo
```

## Dependencies

| Library | Purpose |
|---|---|
| CUDA 12+ | Compute kernels, cuRAND |
| CUB | GPU sort and reduction primitives |
| Eigen3 | CPU linear algebra, Cholesky, covariance |
| nlohmann/json | Config and output serialization |
| Google Test | Unit tests |
| Google Benchmark | CPU vs GPU performance comparison |
| spdlog | Logging |
| Arrow/Parquet C++ | Optional — efficient market data I/O |

## Development Priorities

1. **Mathematical correctness** — optimization must produce provably correct efficient frontiers. Validate against known analytical solutions (2-asset closed-form) and against cvxpy/scipy
2. **Formula traceability** — every formula in code references the source paper and equation number
3. **GPU/CPU parity** — implement both CPU and GPU paths for all compute-heavy operations. Benchmark and document speedups (target: 50-100x for Monte Carlo, 10-50x for optimization)
4. **Component independence** — optimizer, simulator, and backtester are independently usable
5. **Realistic constraints** — position limits, turnover, transaction costs. Not a toy optimizer

## Key Mathematical References

- **CVaR (Conditional Value-at-Risk):** CVaR_α = E[L | L ≥ VaR_α] — average loss in the worst α% of scenarios
- **Mean-CVaR optimization:** min CVaR_α(w) s.t. E[r'w] ≥ μ_target, constraints on w
- **Rockafellar-Uryasev formulation:** reformulates CVaR minimization as a linear program over scenarios. See: Rockafellar & Uryasev, *Optimization of Conditional Value-at-Risk*, J. Risk 2000
- **Cholesky decomposition:** Σ = LLᵀ → correlated samples = L × z where z ~ N(0,I)
- **ADMM:** splits constrained optimization into simpler subproblems. See: Boyd et al., *Distributed Optimization and Statistical Learning via ADMM*, 2011
- **Factor model:** R = Bf + ε, Σ = BΣ_f Bᵀ + D — reduces dimensionality of covariance estimation
- **Shrinkage estimation:** Ledoit & Wolf, *Honey, I Shrunk the Sample Covariance Matrix*, 2004
