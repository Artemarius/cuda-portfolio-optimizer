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

## Phase 3 — Monte Carlo Scenario Generation (CUDA)

**Goal:** Generate correlated return scenarios on GPU. This is the first CUDA kernel and the foundation for everything downstream.

### Tasks

1. `src/simulation/cholesky_utils.h/cpp`:
   - Eigen Cholesky decomposition: Σ = LLᵀ
   - Pack lower triangular L into flat row-major array for GPU transfer
   - Validate: L × Lᵀ ≈ Σ within tolerance
2. `src/simulation/scenario_matrix.h`:
   - `ScenarioMatrix` class: GPU-resident (N_scenarios × N_assets)
   - Column-major layout for coalesced memory access during portfolio loss computation
   - Host ↔ device transfer methods
   - VRAM tracking: log allocated bytes, warn if approaching 6GB limit
3. `src/simulation/monte_carlo.cu`:
   - `k_monte_carlo_simulate` kernel:
     - Each thread generates one scenario (row of the scenario matrix)
     - cuRAND device API: `curand_normal()` for z ~ N(0,I)
     - Correlated return: r = μ + L × z
     - Thread-local cuRAND state initialized from global seed + thread index
   - Grid/block sizing: 256 threads/block, ceil(N_scenarios / 256) blocks
4. `src/simulation/monte_carlo.h/cpp`:
   - Host-side orchestration:
     - Allocate device memory for L, μ, scenario matrix
     - Upload L, μ to device
     - Launch kernel
     - Optionally download scenarios to host for validation
   - CPU reference implementation for validation (same math, `std::mt19937`)
5. VRAM budget utility in `src/utils/cuda_utils.h`:
   - `get_free_vram()`, `log_vram_usage()`
   - Pre-check: scenario matrix + working buffers fit in available VRAM

### Tests

- `test_monte_carlo.cpp`:
  - **Mean convergence:** sample mean of scenarios → μ as N → ∞ (within 3σ/√N)
  - **Covariance convergence:** sample covariance of scenarios → Σ as N → ∞
  - **Correlation structure:** 2-asset case with known ρ, verify empirical correlation
  - **GPU vs CPU parity:** same seed → same output (or statistically equivalent)
  - **Independence:** different seeds → different output (KS test on marginals)

### Benchmarks

- `bench_monte_carlo.cpp`:
  - GPU vs CPU: 10K, 50K, 100K scenarios × 50, 100, 500 assets
  - Report: time (ms), throughput (scenarios/sec), speedup ratio
  - Memory: peak VRAM usage per configuration

### Definition of Done

- 100K scenarios × 500 assets generated in < 100ms on RTX 3060
- Sample covariance of generated scenarios matches input Σ within statistical tolerance
- CPU reference produces identical mathematical results
- VRAM usage logged and stays under 4GB (leaving headroom for optimizer)

### Notes

- **Memory layout matters.** Column-major scenario matrix means when computing portfolio loss (dot product of weights × one scenario), all threads access the same column simultaneously → coalesced reads. Row-major would cause strided access → 10x slower.
- **cuRAND state initialization is expensive.** Initialize once, store states in device memory, reuse across multiple optimizer iterations if regenerating scenarios.
- **RTX 3060 6GB budget:** 100K × 500 × 4 bytes = 200MB for scenarios. L matrix = 500 × 500 × 4 = 1MB. cuRAND states = 100K × 48 bytes ≈ 5MB. Total ≈ 206MB. Comfortable.

---

## Phase 4 — Risk Computation (CUDA)

**Goal:** Compute VaR, CVaR, and other risk metrics on GPU from scenario matrix.

### Tasks

1. `src/risk/risk_metrics.cu`:
   - `k_compute_portfolio_loss` kernel:
     - Input: scenario matrix (N × n), weight vector w (n)
     - Output: loss vector (N): loss_i = -rᵢᵀw
     - One thread per scenario, dot product with shared memory for weights
   - `k_compute_var_cvar`:
     - Sort losses using CUB `DeviceRadixSort`
     - VaR_α = losses[floor(α × N)]
     - CVaR_α = mean(losses[floor(α × N)..N-1])
     - CUB `DeviceReduce::Sum` for the tail mean
2. `src/risk/risk_metrics.h/cpp`:
   - Host orchestration: allocate, launch, retrieve
   - CPU reference implementation for all metrics
   - Additional metrics (can be CPU-only initially):
     - Volatility: σ_p = std(rᵀw) across scenarios
     - Sharpe: (E[rᵀw] - r_f) / σ_p
     - Sortino: (E[rᵀw] - r_f) / σ_downside
     - Maximum drawdown (from scenario paths — requires multi-period scenarios or historical)
3. `src/risk/risk_decomposition.h/cpp`:
   - Component CVaR: contribution of asset j to portfolio CVaR
   - CVaR_j = w_j × E[r_j | portfolio loss ≥ VaR_α]
   - Sum of component CVaR = total CVaR (verify this identity)

### Tests

- `test_risk_metrics.cpp`:
  - **Known distribution:** N(0,1) scenarios → VaR_0.05 ≈ -1.645, CVaR_0.05 ≈ -2.063
  - **GPU vs CPU parity:** same inputs → same VaR/CVaR within 1e-5
  - **Monotonicity:** increasing α → decreasing |CVaR|
  - **CVaR ≥ VaR:** always, for any α and any portfolio
  - **Component CVaR sums to total:** Σ CVaR_j = CVaR_portfolio

### Benchmarks

- `bench_cvar.cpp`:
  - GPU vs CPU: 10K, 50K, 100K scenarios
  - Break down: loss computation time vs sort time vs reduction time

### Definition of Done

- CVaR computation on 100K scenarios in < 10ms on GPU
- Matches CPU reference within floating-point tolerance
- Risk decomposition components sum to total CVaR

---

## Phase 5 — ADMM Optimizer Core

**Goal:** Working ADMM solver for unconstrained and simply-constrained Mean-CVaR optimization.

### Tasks

1. `src/optimizer/objective.h/cpp`:
   - Rockafellar-Uryasev objective evaluation:
     - F(w, ζ) = ζ + (1/(Nα)) Σᵢ max(0, -rᵢᵀw - ζ)
   - Gradient computation (subgradient for the max terms)
   - Both CPU and GPU evaluation paths
2. `src/optimizer/projections.h/cpp`:
   - Simplex projection: project w onto {w : 1ᵀw = 1, w ≥ 0} — Duchi et al. 2008
   - Box projection: w_min ≤ w ≤ w_max (element-wise clamp)
   - Combined simplex + box: iterative projection (Dykstra's algorithm or similar)
3. `src/optimizer/admm_solver.h/cpp`:
   - ADMM iteration:
     ```
     x^{k+1} = argmin_x { f(x) + (ρ/2)||x - z^k + u^k||² }   (x-update)
     z^{k+1} = Π_C(x^{k+1} + u^k)                              (z-update: projection)
     u^{k+1} = u^k + x^{k+1} - z^{k+1}                         (dual update)
     ```
   - Convergence criteria: primal residual ||x - z|| < ε_pri, dual residual ||ρ(z^k - z^{k-1})|| < ε_dual
   - ρ (penalty parameter) adaptive update: Boyd et al. 2011 §3.4.1
   - Maximum iterations, logging per-iteration residuals
4. `src/optimizer/admm_kernels.cu`:
   - GPU-accelerated x-update: involves scenario matrix multiplication
   - This is the expensive step — matrix-vector products across all scenarios
5. `src/optimizer/efficient_frontier.h/cpp`:
   - Sweep target returns μ_target from μ_min to μ_max
   - Solve Mean-CVaR for each target
   - Output: vector of (μ_target, CVaR, weights)

### Tests

- `test_admm_solver.cpp`:
  - **2-asset analytical case:** closed-form efficient frontier (Markowitz), verify ADMM frontier matches
  - **Trivial case:** single asset → w = [1], CVaR = scenario CVaR
  - **Equal expected returns:** all assets same μ → minimum CVaR = minimum variance portfolio (approximately)
  - **Constraint satisfaction:** all output weights satisfy 1ᵀw = 1, w_min ≤ w ≤ w_max
  - **Cross-validation:** 5-asset problem solved with ADMM vs Python cvxpy → same frontier ± tolerance
- `test_projections.cpp`:
  - Simplex: random vector → projected vector sums to 1, all ≥ 0
  - Box: projected vector within [w_min, w_max]
  - Idempotency: projecting an already-feasible point returns itself

### Definition of Done

- ADMM converges on 2-asset and 5-asset problems
- Efficient frontier is monotonically increasing (more return → more risk)
- Matches cvxpy solution within 1e-4 on the 5-asset case
- Convergence in < 500 iterations for typical problems

### Notes

- **The x-update is the bottleneck.** For the Rockafellar-Uryasev formulation, it involves computing max(0, -rᵢᵀw - ζ) across all N scenarios for each candidate w. This is a perfect GPU workload.
- **Start without GPU acceleration.** Get the ADMM logic correct on CPU first, then move the x-update to CUDA. Correctness before performance.
- **ρ tuning is critical.** Too small → slow convergence. Too large → oscillation. Use adaptive ρ from Boyd et al. 2011 from the start.

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
