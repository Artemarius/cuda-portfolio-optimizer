# ROADMAP.md — cuda-portfolio-optimizer

## Development Phases

This roadmap is structured for incremental development with testable deliverables at each phase. Each phase builds on the previous one. Phases are designed for Claude Code–assisted implementation with clear "Definition of Done" criteria.

---

## Phase 1 — Project Skeleton & Build System ✅

**Status:** Complete

**Goal:** Compilable CMake project with CUDA detected, dependencies resolved, empty library + test targets.

### Tasks

1. Create top-level `CMakeLists.txt`:
   - `project(cuda_portfolio_optimizer LANGUAGES CXX CUDA)`
   - C++17, CUDA 17, `CMAKE_CUDA_ARCHITECTURES 86` (RTX 3060)
   - Generator: Visual Studio 17 2022 or Ninja
2. Create `src/CMakeLists.txt` — static library target `cuda_portfolio_lib`
3. Create `apps/CMakeLists.txt` — executable targets `optimize`, `backtest`
4. Create `tests/CMakeLists.txt` — Google Test via FetchContent
5. Create `benchmarks/CMakeLists.txt` — Google Benchmark via FetchContent
6. Fetch/find dependencies:
   - Eigen3 (FetchContent or find_package)
   - nlohmann/json (FetchContent)
   - spdlog (FetchContent)
   - CUB (ships with CUDA Toolkit 12+)
   - cuRAND (ships with CUDA Toolkit)
7. Create `src/core/types.h`:
   - `using Scalar = float;` (GPU path), `using ScalarCPU = double;` (optimizer precision)
   - `using Index = int;`
   - Eigen matrix typedefs: `using MatrixXs = Eigen::MatrixXf;`, etc.
8. Create `src/utils/cuda_utils.h`:
   - `CUDA_CHECK(err)` macro
   - `device_query()` — print GPU name, SM count, memory
9. Create `src/utils/timer.h`:
   - RAII CPU timer (`std::chrono`)
   - CUDA event timer wrapper
10. Minimal `apps/optimize_main.cpp` — prints "cuda-portfolio-optimizer" + GPU info
11. Minimal test: `tests/test_smoke.cpp` — Google Test links and runs

### Definition of Done

- `cmake -B build -G "Visual Studio 17 2022"` succeeds
- `cmake --build build --config Release` compiles with zero errors
- `ctest --test-dir build -C Release` runs smoke test
- `./build/Release/optimize` prints GPU info
- All dependencies resolve without manual installation

### Notes

- Use `FetchContent` for all dependencies except CUDA toolkit components. This makes the repo self-contained — `git clone` + `cmake` + `build` with no manual steps.
- Do NOT add `find_package(CUDA)` — use native CMake CUDA language support (`enable_language(CUDA)` via `project(...LANGUAGES CUDA)`).
- Set `CMAKE_CUDA_ARCHITECTURES 86` explicitly. Do not use `CMAKE_CUDA_ARCHITECTURES all` — it triples compile time for no benefit on a single-GPU dev machine.

---

## Phase 2 — Market Data Loading & Return Computation ✅

**Status:** Complete (data layer: CSV loading, returns, universe)

**Goal:** Load historical price CSV → compute return matrix. CPU-only, no CUDA yet.

### Implemented

1. `src/data/market_data.h`:
   - `PriceData` struct: dates, tickers, prices (MatrixXd, T x N)
   - `ReturnData` struct: dates, tickers, returns (MatrixXd, T-1 x N), return_type
   - Enums: `MissingDataPolicy` (kDropRows, kForwardFill), `ReturnType` (kSimple, kLog)
2. `src/data/csv_loader.h/cpp`:
   - `load_csv_prices(path, policy)` — load all tickers from wide-format CSV
   - `load_csv_prices(path, tickers, policy)` — filter to specific tickers
   - `load_csv_prices(path, universe, policy)` — filter by universe (tickers + date range)
   - Handles: Windows \r\n, UTF-8 BOM, non-positive price warnings, locale-safe parsing
   - Missing data: kDropRows removes rows with any gap; kForwardFill uses previous day's price
3. `src/data/returns.h/cpp`:
   - `compute_returns(PriceData, ReturnType)` → ReturnData (with metadata)
   - `compute_returns(MatrixXd, ReturnType)` → MatrixXd (raw matrix overload)
   - `compute_excess_returns(ReturnData, risk_free_rate, periods_per_year)` → ReturnData
   - End-of-period date convention for return labeling
4. `src/data/universe.h/cpp`:
   - `Universe` struct: tickers, start_date, end_date
   - `load_universe(json_path)` — JSON config via nlohmann/json

### Tests (16 passing)

- `tests/test_data.cpp`:
  - CSV loading: dimensions, values, ticker filtering, universe filtering, missing data (both policies), error cases
  - Returns: simple vs log (verified against hand-computed values to 1e-10), matrix overload, excess returns, edge cases
  - Universe: JSON loading, nonexistent file
- Test fixtures: `tests/data/prices_2asset.csv`, `prices_5asset.csv`, `prices_missing.csv`, `universe_test.json`

### Deferred to Later Phases

- Sample/shrinkage covariance estimators → Phase 2b or Phase 3
- Config struct (config.h) → when CLI apps need it
- `scripts/download_data.py` → when real SP500 data is needed

---

## Phase 3 — Monte Carlo Scenario Generation (CUDA) ✅

**Status:** Complete (GPU kernel, Cholesky, cuRAND, CPU reference, benchmarks)

**Goal:** Generate correlated return scenarios on GPU. This is the first CUDA kernel and the foundation for everything downstream.

### Implemented

1. `src/simulation/cholesky_utils.h/cpp`:
   - `CholeskyResult` struct: `L_cpu` (MatrixXd), `L_flat` (vector<float>, n x n row-major), `n`
   - `compute_cholesky(cov)` — Eigen LLT in double, pack to flat float row-major for GPU
   - `validate_cholesky(result, cov, tol)` — checks ||LLT - cov||_inf < tol
   - Full n x n storage (not triangular-packed) for simpler GPU indexing (500x500 = 1MB)
2. `src/simulation/scenario_matrix.h/cu`:
   - `ScenarioMatrix` RAII class: GPU-resident float matrix (column-major)
   - Column-major layout: element (i,j) at `j * n_scenarios + i` — coalesced reads for downstream kernels
   - Move-only (no copy), cudaFree in destructor without CUDA_CHECK
   - `to_host()` → MatrixXs, `from_host(MatrixXs)`, `from_host(vector<Scalar>)`
   - Logs allocation size in MB via spdlog
3. `src/simulation/monte_carlo.h/cu`:
   - `MonteCarloConfig` struct: n_scenarios, n_assets, seed, threads_per_block
   - Opaque `CurandStates` type — curand_kernel.h confined to .cu file only
   - `CurandStatesGuard` — RAII unique_ptr with custom deleter (handles incomplete type)
   - `create_curand_states(n, seed)` / `destroy_curand_states(states)` — init kernel per thread
   - `k_monte_carlo_simulate` kernel: one thread per scenario, 256 threads/block
     - Two-phase reverse-order write: generate z ~ N(0,I), then r = mu + L*z in-place (i = n-1..0)
     - Reverse order safe because L is lower-triangular: r[i] only reads z[0..i]
     - L accessed row-major from global memory (fits in L2 cache on RTX 3060)
   - `generate_scenarios_gpu(mu, cholesky, config, states)` — host orchestration with optional cuRAND reuse
   - `generate_scenarios_cpu(mu, cholesky, config)` — std::mt19937 reference in double precision

### Tests (15 passing)

- `tests/test_monte_carlo.cpp`:
  - Cholesky: identity, 2-asset (LLT reconstruction + L_flat packing), non-PD throws, non-square throws
  - ScenarioMatrix: GPU alloc + host roundtrip, move semantics (constructor + assignment)
  - GPU: mean convergence (CLT: 3*sigma/sqrt(N)), covariance convergence (5% relative), correlation structure (rho=0.8, within 0.02), reproducibility (same seed), different seeds, cuRAND state reuse (states advance between calls), 5-asset dense correlation
  - CPU: mean convergence, covariance convergence (3% relative for double precision)

### Benchmarks

- `benchmarks/bench_monte_carlo.cpp`:
  - GPU: 9 configs (10K/50K/100K scenarios x 50/100/500 assets)
  - CPU: 6 configs (10K/50K/100K x 50/100, skip 500-asset)
  - Pre-allocates cuRAND states outside timing loop, reports time/throughput/VRAM

### Measured Performance (RTX 3060)

| Config | GPU | CPU | Speedup |
|---|---|---|---|
| 100K x 500 | 187 ms | — | — |
| 100K x 100 | — | 756 ms | ~21x/item |
| VRAM (100K x 500) | 191 MB | — | 3.1% of 6GB |

### Notes

- **Memory layout matters.** Column-major scenario matrix means when computing portfolio loss (dot product of weights × one scenario), all threads access the same column simultaneously → coalesced reads.
- **cuRAND state initialization is expensive.** Initialize once via `create_curand_states`, reuse across calls. States advance automatically between invocations.
- **RTX 3060 6GB budget:** 100K × 500 scenario matrix = 191MB. cuRAND states = 4.6MB. L + mu = ~1MB. Total ~197MB. Peak observed: 1.2GB (including CUDA context). 80% free.

---

## Phase 4 — Risk Computation (CUDA) ✅

**Status:** Complete (CVaR/VaR, portfolio loss kernel, statistics, GPU/CPU parity)

**Goal:** Compute VaR, CVaR, and other risk metrics on GPU from scenario matrix.

### Implemented

1. `src/risk/device_vector.h/cu`:
   - `DeviceVector<T>` RAII template: move-only, cudaMalloc/Free, to_host/from_host
   - Explicit instantiation for `Scalar` (float)
   - Mirrors `ScenarioMatrix` patterns (no-throw destructor, CUDA_CHECK in constructor)
2. `src/risk/risk_result.h`:
   - `RiskConfig` struct: confidence_level (alpha), threads_per_block
   - `RiskResult` struct: VaR, CVaR, expected_return, volatility, Sharpe, Sortino — all ScalarCPU (double)
3. `src/risk/portfolio_loss.h/cu`:
   - `k_compute_portfolio_loss` kernel: one thread per scenario, weights in shared memory (2KB for 500 assets)
   - Column-major coalesced reads: `r(sid, j) = d_scenarios[j * n_scenarios + sid]`
   - `compute_portfolio_loss_gpu(scenarios, weights)` → `DeviceVector<Scalar>` (stays on GPU)
   - `compute_portfolio_loss_cpu(scenarios_host, weights)` → `VectorXd` via Eigen matrix-vector multiply
4. `src/risk/cvar.h/cu`:
   - GPU `compute_risk_gpu` flow:
     1. CUB `DeviceRadixSort::SortKeys` — sort losses ascending into separate buffer (input preserved)
     2. VaR = sorted[floor(alpha * N)], n_tail = N - var_index
     3. CUB `DeviceReduce::Sum` on tail (pointer offset) → CVaR = tail_sum / n_tail
     4. `k_compute_loss_statistics` kernel: single-pass reduction for sum, sum_sq, sum_downside_sq
     5. Derive: expected_return, volatility, Sharpe, Sortino (all in double)
   - `k_compute_loss_statistics`: grid-stride loop, block-level shared memory reduction, atomicAdd to global accumulators (double precision)
   - CPU `compute_risk_cpu`: std::sort + loop in double precision
   - Convenience: `compute_portfolio_risk_gpu/cpu(scenarios, weights)` — full pipeline in one call

### Tests (19 passing)

- `tests/test_risk.cpp`:
  - DeviceVector: allocate + roundtrip, move semantics, size mismatch error
  - Portfolio loss: deterministic 2-asset 4-scenario exact values (GPU + CPU), weight size mismatch
  - CVaR deterministic: known 10-element vector at alpha = 0.50/0.80/0.90, unsorted input, input preservation
  - CVaR properties: CVaR >= VaR at 5 alpha levels, monotonicity in alpha
  - Statistical: 200K N(mu,sigma^2) scenarios vs analytical normal CVaR formula (within 5%)
  - GPU/CPU parity: 3-asset 50K scenarios, VaR/CVaR/E[r]/vol within float tolerance (~1e-3)
  - Statistics: deterministic expected return, volatility, Sharpe, Sortino
  - Edge cases: single element, invalid alpha throws

### Benchmarks

- `benchmarks/bench_cvar.cpp`:
  - Portfolio loss: GPU vs CPU at 10K-100K scenarios x 50-500 assets
  - CVaR only: GPU vs CPU from pre-computed losses
  - Full pipeline: scenarios → loss → CVaR, GPU vs CPU

### Deferred to Later Phases

- Risk decomposition (component CVaR) → Phase 5 or 6 when optimizer needs it
- Maximum drawdown → Phase 7 (backtest engine)

---

## Phase 5 — ADMM Optimizer Core ✅

**Status:** Complete (ADMM solver, projections, R-U objective, GPU kernel, efficient frontier)

**Goal:** Working ADMM solver for unconstrained and simply-constrained Mean-CVaR optimization.

### Implemented

1. `src/optimizer/projections.h/cpp`:
   - `project_simplex(v)` — O(n log n) sorting-based simplex projection (Duchi et al. 2008)
   - `project_box(v, lb, ub)` — element-wise clamping
   - `project_simplex_box(v, lb, ub)` — Dykstra's alternating projection (Boyle & Dykstra 1986) for the intersection of simplex and box constraints
2. `src/optimizer/objective.h/cpp`:
   - `evaluate_objective_cpu(scenarios, w, zeta, alpha)` — Rockafellar-Uryasev CVaR objective F(w,ζ) = ζ + (1/(Nα)) Σᵢ max(0, -rᵢᵀw - ζ), with subgradients ∂F/∂w and ∂F/∂ζ
   - `find_optimal_zeta(scenarios, w, alpha)` — finds VaR (optimal ζ) by sorting losses
3. `src/optimizer/admm_solver.h/cpp`:
   - `AdmmConfig` struct: rho, adaptive rho bounds, convergence tolerances, box constraints, target return
   - `AdmmResult` struct: weights, CVaR, expected return, iteration history
   - `admm_solve(scenarios, mu, config, w_init)` — full ADMM loop:
     - x-update: proximal gradient descent on augmented R-U objective (joint w and ζ optimization)
     - z-update: projection onto constraint set (simplex, box, target return)
     - u-update: dual variable
     - Adaptive ρ: Boyd et al. 2011 §3.4.1, Eq. (3.13) — balances primal/dual residuals
     - Convergence: Boyd 2011 Section 3.3, Eq. (3.11)-(3.12)
4. `src/optimizer/admm_kernels.h/cu`:
   - `k_evaluate_ru_objective` kernel: per-scenario loss computation + tail accumulation
   - Weights in shared memory, column-major coalesced reads (same pattern as portfolio_loss kernel)
   - Double-precision atomicAdd for gradient accumulation (accuracy)
   - `evaluate_objective_gpu(scenarios, w, zeta)` — host wrapper
5. `src/optimizer/efficient_frontier.h/cpp`:
   - `FrontierPoint` struct: target return, achieved return, CVaR, weights, convergence info
   - `compute_efficient_frontier(scenarios, mu, config)` — sweeps target returns, warm-starts from previous solution

### Tests (26 passing: 16 projection + 10 ADMM)

- `tests/test_projections.cpp`:
  - Simplex: already-on-simplex, uniform, negative entries, single element, all-negative, idempotent, large (100-dim)
  - Box: within bounds, clamp lower/upper, dimension mismatch, idempotent
  - Combined: wide bounds (matches pure simplex), position limits (max 40%), feasible unchanged, tight bounds
- `tests/test_admm_solver.cpp`:
  - Objective: basic R-U evaluation (hand-computed), optimal zeta, invalid alpha
  - Single asset: w = [1.0] (converges in 6 iterations)
  - 2-asset: constraint satisfaction (sum-to-1, non-negative, CVaR > 0)
  - Box constraints: 3-asset with max 50% per position — bounds respected
  - Equal expected returns: minimum-risk portfolio allocates most to lowest-variance asset
  - GPU/CPU parity: objective value and gradient match within float tolerance
  - Efficient frontier: basic (5 points, valid weights), monotonic risk (CVaR increases with return)

### Convergence Performance

| Problem | Iterations | Time |
|---|---|---|
| 1 asset, 1K scenarios | 6 | 1 ms |
| 2 assets, 10K scenarios | 59 | 148 ms |
| 3 assets, 10K scenarios (box) | 86 | 190 ms |
| 3 assets, 20K scenarios (frontier, 5 pts) | 30-195 per point | 650 ms total |

### Deferred to Later Phases

- GPU-accelerated x-update integration into the ADMM loop (kernel exists, not yet wired into solver) → Phase 6 or 8
- Cross-validation against cvxpy → Phase 9 (validation suite)
- 5-asset and larger problems → Phase 8 (scalability)

---

## Phase 6 — Realistic Constraints

**Goal:** Add position limits, turnover, leverage, and sector constraints to the optimizer.

### Tasks

1. `src/constraints/constraint_set.h/cpp`:
   - Unified constraint representation
   - Parse from JSON config
   - Feasibility check: does a given w satisfy all constraints?
2. `src/constraints/position_limits.h/cpp`:
   - Long-only: w ≥ 0
   - Position bounds: w_min ≤ w ≤ w_max (e.g., max 5% per asset)
   - These are handled by the box projection in ADMM z-update
3. `src/constraints/turnover.h/cpp`:
   - ||w - w_prev||₁ ≤ τ (L1 turnover constraint)
   - Requires w_prev (previous portfolio) as input
   - ADMM: add as indicator function, project onto L1 ball centered at w_prev
4. `src/constraints/sector_exposure.h/cpp`:
   - Σ_{j ∈ sector_k} w_j ≤ s_max (sector caps)
   - Σ_{j ∈ sector_k} w_j ≥ s_min (sector floors)
   - ADMM: additional consensus variable and projection
5. Update ADMM solver to handle multiple constraint sets via consensus ADMM or splitting

### Tests

- Position limits: optimizer output respects bounds
- Turnover: when τ is very large → unconstrained solution; when τ = 0 → w = w_prev
- Sector: known 2-sector problem, verify sector weights within bounds
- Combined: all constraints active simultaneously

### Definition of Done

- All constraints parsed from JSON config
- Optimizer respects all constraints (verified by feasibility checker)
- Adding constraints doesn't break convergence (may increase iterations)

---

## Phase 7 — Backtesting Engine

**Goal:** Rolling-window backtest with multiple strategies and transaction costs.

### Tasks

1. `src/backtest/strategy.h/cpp`:
   - Strategy interface: given historical returns window → produce weights
   - Implementations:
     - `MeanVarianceStrategy` — Markowitz mean-variance (baseline)
     - `MeanCVaRStrategy` — our Mean-CVaR optimizer
     - `EqualWeightStrategy` — 1/N (surprisingly hard to beat)
     - `RiskParityStrategy` — inverse-volatility weighting
2. `src/backtest/transaction_costs.h/cpp`:
   - Proportional cost: cost = c × Σ|w_new,j - w_old,j| × portfolio_value
   - Configurable cost rate (e.g., 10 bps)
   - Minimum trade threshold: don't rebalance tiny positions
3. `src/backtest/backtester.h/cpp`:
   - Rolling window:
     - At each rebalance date (monthly):
       1. Extract trailing window (e.g., 252 days) of returns
       2. Estimate μ, Σ (with shrinkage)
       3. Generate Monte Carlo scenarios (for CVaR strategy)
       4. Optimize portfolio
       5. Apply transaction costs
       6. Record portfolio value
   - Output: daily portfolio values, weights over time, turnover, costs
4. `src/reporting/report_writer.h/cpp`:
   - CSV output: equity curves, weights, risk metrics per period
   - JSON output: summary statistics, strategy comparison
   - Metrics: annualized return, volatility, Sharpe, Sortino, max drawdown, Calmar ratio, turnover

### Tests

- `test_backtester.cpp`:
  - Equal-weight on 2 assets with known returns → verify portfolio value by hand
  - Transaction costs: known trades at known cost rate → verify total cost
  - Rolling window: correct number of rebalancing events for date range

### Definition of Done

- Full backtest on 5 years of SP500 data with monthly rebalancing completes in < 5 minutes
- Strategy comparison table: MVO vs CVaR vs 1/N vs Risk Parity
- Equity curves and risk metrics output as CSV
- Transaction costs correctly reduce portfolio value

---

## Phase 8 — Factor Model & Scalability

**Goal:** Add factor model for large universes and optimize for 500-asset case.

### Tasks

1. `src/models/factor_model.h/cpp`:
   - PCA-based factor extraction: R = Bf + ε
   - Covariance decomposition: Σ = BΣ_fBᵀ + D
   - Configurable number of factors (e.g., 5-10 for SP500)
   - Factor model Monte Carlo: generate factor scenarios, then asset returns
2. Memory optimization for large universes:
   - Tiled scenario generation if N × n exceeds VRAM budget
   - In-place CVaR computation (don't materialize full loss vector if possible)
3. Profile and optimize CUDA kernels:
   - Nsight Compute analysis
   - Occupancy optimization
   - Shared memory usage for weight vector in loss computation

### Definition of Done

- Factor model covariance matches sample covariance for well-factored data
- 500-asset optimization completes end-to-end in < 1 second
- Monte Carlo kernel achieves > 50% occupancy on RTX 3060
- Peak VRAM usage documented for all configurations

---

## Phase 9 — Benchmarks, Validation & Documentation

**Goal:** Production-quality benchmarks, comprehensive validation, and polished documentation.

### Tasks

1. **Benchmarks:**
   - GPU vs CPU comparison table with actual measured numbers
   - Scaling study: how does time grow with N_scenarios and N_assets?
   - RTX 3060 specific: note SM count (28), memory bandwidth, theoretical throughput
2. **Cross-validation suite:**
   - `scripts/validate_cvxpy.py`:
     - Solve same problems with cvxpy + ECOS/SCS
     - Dump solutions as JSON
     - C++ test loads JSON and compares
   - Test cases: 2, 5, 10, 50, 100 assets
   - Compare: weights (L∞ norm), objective value, constraint satisfaction
3. **README polish:**
   - Actual benchmark numbers (not just targets)
   - Example output: efficient frontier plot, backtest equity curve
   - Build instructions verified from clean clone
4. **Code documentation:**
   - All public APIs have doxygen-style doc comments
   - Mathematical formulas referenced in comments with paper + equation number
   - Architecture diagram in README

### Definition of Done

- Benchmark table with real numbers on RTX 3060
- cvxpy validation passes for all test sizes
- `git clone` → `cmake` → `build` → `run` works on Windows with CUDA 12.8
- Every formula in code has a paper reference

---

## Phase 10 — Polish & Portfolio Integration

**Goal:** Final polish, connect to portfolio site, ensure the repo tells a compelling story.

### Tasks

1. Repository presentation:
   - Clean commit history (squash WIP commits)
   - Comprehensive `.gitignore`
   - LICENSE (MIT)
   - GitHub Actions CI (optional — Windows + CUDA is hard in CI, but if feasible)
2. Results visualization:
   - `scripts/plot_frontier.py` — matplotlib efficient frontier
   - `scripts/plot_backtest.py` — equity curves, drawdown chart
   - Include sample output images in README
3. Link from portfolio site (Artemarius.github.io):
   - Project card with description, key metrics, tech stack
   - Link to repo with highlight of benchmark results
4. Final documentation review:
   - PROJECT.md reflects actual implementation (not just plans)
   - CLAUDE.md accurate for Claude Code development
   - README suitable for a quant firm hiring manager reading in 2 minutes

### Definition of Done

- Repo is presentable to a hiring manager
- README tells the story: motivation → what's implemented → results → how to build
- No dead code, no TODOs in public-facing code
- Portfolio site links to repo with project summary

---

## Appendix: VRAM Budget (RTX 3060 6GB)

| Component | Size (100K × 500) | Size (100K × 100) |
|---|---|---|
| Scenario matrix (float32) | 200 MB | 40 MB |
| cuRAND states | 5 MB | 5 MB |
| Cholesky factor L | 1 MB | 0.04 MB |
| Loss vector | 0.4 MB | 0.4 MB |
| ADMM working buffers | ~200 MB | ~40 MB |
| **Total** | **~406 MB** | **~85 MB** |
| **Available** | **6144 MB** | **6144 MB** |
| **Headroom** | **93%** | **99%** |

## Appendix: Dependency Versions

| Dependency | Version | Acquisition |
|---|---|---|
| CUDA Toolkit | 12.8 | Pre-installed |
| CUB | Ships with CUDA 12.8 | — |
| cuRAND | Ships with CUDA 12.8 | — |
| Eigen3 | 3.4.x | FetchContent |
| nlohmann/json | 3.11.x | FetchContent |
| spdlog | 1.x | FetchContent |
| Google Test | 1.14.x | FetchContent |
| Google Benchmark | 1.8.x | FetchContent |

## Appendix: File Naming Conventions

- Headers: `.h` (not `.hpp` — matches Google C++ Style)
- C++ source: `.cpp`
- CUDA source: `.cu`
- CUDA headers with device code: `.cuh` (if needed, prefer `.h` with `__host__ __device__` annotations)
- Tests: `test_<module>.cpp`
- Benchmarks: `bench_<module>.cpp`
