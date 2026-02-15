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

## Phase 6 — Realistic Constraints ✅

**Status:** Complete (position limits, turnover, sector constraints, unified ConstraintSet, Dykstra's projection)

**Goal:** Add position limits, turnover, and sector constraints to the optimizer.

### Implemented

1. `src/constraints/constraint_set.h/cpp`:
   - `PositionLimits`: per-asset w_min/w_max bounds
   - `TurnoverConstraint`: L1 turnover ||w - w_prev||_1 <= tau with previous portfolio
   - `SectorBound` / `SectorConstraints`: per-sector min/max exposure with named sectors
   - `ConstraintSet`: unified container with `validate()`, `is_feasible()`, `num_constraint_sets()`
   - `parse_constraints(json, n_assets)` — JSON parsing with full validation
2. `src/optimizer/projections.h/cpp` — 3 new projection operators:
   - `project_l1_ball(v, center, radius)` — L1 ball projection via Duchi 2008 soft-thresholding, O(n log n)
   - `project_sector(v, indices, s_min, s_max)` — sector sum projection via uniform adjustment
   - `project_constraints(v, constraints)` — generalized N-set Dykstra's alternating projection (Boyle & Dykstra 1986), cycles through simplex → box → L1 ball → each sector, with convergence check on both x and increment stability
3. `src/optimizer/admm_solver.h/cpp` — migrated from separate box constraint fields to unified `ConstraintSet`:
   - `AdmmConfig.constraints` replaces `has_box_constraints`/`w_min`/`w_max`
   - z-update uses `project_constraints()` for all constraint types
   - Target return correction re-projects through `project_constraints()`

### Tests (26 constraint tests + 2 new ADMM tests = 108 total passing)

- `tests/test_constraints.cpp`:
  - L1 ball: already-in-ball, zero radius, basic, symmetric, negative entries, idempotent, dimension mismatch (7)
  - Sector: within bounds, exceeds max, below min, empty sector (4)
  - ConstraintSet: empty feasible, simplex/box/turnover/sector violations, all-feasible combined (6)
  - Validation: dimension mismatch, infeasible bounds, num_constraint_sets counting (3)
  - Dykstra's: simplex-only matches project_simplex, simplex+box feasible, tau=0 forces w_prev, large tau inactive, sector bounds respected, all constraints combined (6)
- `tests/test_admm_solver.cpp`:
  - Box constraints: migrated to ConstraintSet API (existing, updated)
  - Turnover: 3 assets, tau=0.3, optimizer output within turnover limit
  - Sector: 4 assets, sector {0,1} capped at 40%, sector sum within limit

### Design Decisions

- **Single header for all constraint types** — `ConstraintSet` contains all optional constraints with boolean flags. Simpler than separate files per constraint type.
- **Generalized Dykstra's** — extends the existing 2-set simplex+box pattern to N sets. Convergence checked on both x stability and increment stability to prevent premature termination.
- **Constraint migration** — replaced `AdmmConfig.has_box_constraints`/`w_min`/`w_max` with `AdmmConfig.constraints` (unified `ConstraintSet`). All existing tests updated and passing.

---

## Phase 7 — Backtesting Engine ✅

**Status:** Complete (rolling-window engine, 4 strategies, transaction costs, CSV/JSON reporting, CLI)

**Goal:** Rolling-window backtest with multiple strategies and transaction costs.

### Implemented

1. `src/backtest/transaction_costs.h/cpp`:
   - `TransactionCostConfig`: proportional cost rate (default 10 bps), minimum trade threshold
   - `TransactionCostResult`: total cost, cost fraction, turnover, effective weights
   - `compute_transaction_costs(w_new, w_old, value, config)` — threshold filtering, renormalization, L1 turnover
2. `src/backtest/strategy.h/cpp`:
   - `Strategy` abstract interface: `allocate(returns_window, w_prev)` → `AllocationResult`
   - `EqualWeightStrategy` — 1/N (always succeeds)
   - `RiskParityStrategy` — inverse-volatility weighting: w_i = (1/sigma_i) / sum(1/sigma_j)
   - `MeanVarianceStrategy` — sample covariance + Eigen LDLT, global min-variance or target-return (Merton 1972 closed-form), optional shrinkage toward identity, long-only clamping
   - `MeanCVaRStrategy` — full pipeline: estimate mu/Sigma → Cholesky → Monte Carlo scenarios (GPU or CPU) → ADMM solve, warm-start from w_prev, turnover constraint integration
   - `create_strategy(name)` factory function
3. `src/backtest/backtest_config.h/cpp`:
   - `BacktestConfig` struct: data settings, rolling window (lookback, rebalance freq), strategy, ADMM/MC config, transaction costs, output dir
   - `load_backtest_config(json_path)` — JSON parsing following existing `constraint_set.cpp` patterns
4. `src/backtest/backtest_engine.h/cpp`:
   - `PortfolioSnapshot`: date, value, cost, turnover, return, weights, rebalance flag
   - `BacktestSummary`: total/annualized return, vol, Sharpe, Sortino, max drawdown, Calmar, transaction cost aggregates
   - `run_backtest(returns, strategy, config)` — rolling-window loop:
     1. Start at t = lookback_window, equal-weight initial portfolio
     2. Daily: portfolio return w'r_t, value update, weight drift w_i *= (1+r_i)/(1+r_p)
     3. Every rebalance_frequency days: slice window, call strategy, apply transaction costs
   - `compute_backtest_summary(snapshots, name)` — aggregate metrics
5. `src/reporting/report_writer.h/cpp`:
   - `write_equity_curve_csv` — date, value, return, cost, rebalance flag
   - `write_weights_csv` — rebalance-day rows only, one column per asset
   - `write_summary_json` — single strategy summary
   - `write_comparison_json` / `write_comparison_csv` — multi-strategy comparison
6. `apps/backtest_main.cpp` — full CLI:
   - `--config <path>` and `--output <dir>` arguments
   - Loads config, prices, computes returns
   - Runs single strategy or all 4 (strategy_name = "all")
   - Shared cuRAND states across MeanCVaR rebalances
   - Writes per-strategy reports + comparison table
   - Prints summary to stdout via spdlog

### Tests (28 passing, 136 total)

- `tests/test_backtest.cpp`:
  - Transaction costs: zero turnover, full rebalance, proportional correctness, threshold suppression, dimension mismatch (5)
  - Strategies: equal weight sum-to-1, risk parity inverse-vol weighting (2:1 ratio), equal-vol → 1/N, min-variance weights sum-to-1, 2-asset analytical verification, MeanCVaR valid weights, factory known/unknown names (8)
  - Config: JSON round-trip parse, missing file error (2)
  - Engine: known 2-asset returns, transaction costs reduce value, rebalance count, weight drift correctness, insufficient data throws (5)
  - Summary: max drawdown on known curve [100,110,105,120,90,100]→25%, Sharpe positive (2)
  - Reporting: equity curve CSV write/read, summary JSON structure, weights CSV rebalance-only rows, comparison CSV/JSON (5)
- Test fixtures: `tests/data/backtest_config.json`

### Design Decisions

- **Strategy receives raw return sub-window**, not the full dataset. The engine handles slicing. Strategies are stateless between calls.
- **CurandStates* passed as raw pointer** to MeanCVaRStrategy. The engine/CLI owns the CurandStatesGuard and passes `.get()`. States are reused across all rebalance windows.
- **Sample covariance with optional shrinkage** `(1-d)*S + d*trace(S)/n*I` for robustness when T < N. Full Ledoit-Wolf deferred to Phase 8.
- **No GPU in the backtest loop itself** — the loop is sequential. GPU acceleration is within MeanCVaRStrategy (scenario generation via existing CUDA kernels).
- **Long-only enforced** by all strategies (simplex constraint or clamp+renormalize).

---

## Phase 8 — Factor Model & Scalability

**Status:** Complete

**Goal:** Add factor model for large universes and optimize for 500-asset case.

### Implemented

1. `src/models/factor_model.h/cpp`:
   - `FactorModelConfig` struct: `n_factors` (int), `min_variance_explained` (double, 0 = disabled)
   - `FactorModelResult` struct: loadings (N x k), factor_returns (T x k), factor_covariance (k x k), idiosyncratic_var (N), eigenvalues (N sorted desc), mu (N), variance_explained
   - `fit_factor_model(returns, config)` — PCA via Eigen `SelfAdjointEigenSolver` on sample covariance, top-k eigenvectors as loadings, factor returns F = X*B, residual variance with 1e-10 floor for PD guarantee, auto-k selection by variance explained threshold
   - `reconstruct_covariance(model)` — Σ = B * Σ_f * B' + diag(D)
   - `compute_cholesky_from_factor_model(model)` — reconstructs cov then delegates to existing `compute_cholesky()`
2. `src/models/factor_monte_carlo.h/cu`:
   - `k_factor_monte_carlo` GPU kernel: one thread per scenario, shared memory for B (N x k), mu (N), sqrt_D (N)
   - Generates k normals z_f, computes f = L_f * z_f (lower-tri multiply in registers), then r_i = mu_i + B_i'*f + sqrt(D_i)*z_e
   - Complexity: O(Nk + k²) per scenario vs O(N²) for full Cholesky — 25x fewer FLOPs for N=500, k=10
   - `generate_scenarios_factor_gpu(mu, model, config, states)` — host orchestration: Cholesky of k×k factor covariance, convert to float, upload, launch kernel
   - `generate_scenarios_factor_cpu(mu, model, config)` — std::mt19937 reference in double precision
   - `src/simulation/curand_states.cuh` — shared CurandStates struct definition for cross-TU access
3. `src/models/tiled_scenario_generator.h/cu`:
   - `TiledConfig` struct: `vram_fraction` (default 0.7), `min_tile_size`
   - `generate_scenarios_tiled(mu, cholesky, mc_config, tiled_config)` — queries free VRAM, computes tile size, generates in chunks, downloads each tile to CPU
   - `generate_scenarios_factor_tiled(mu, model, mc_config, tiled_config)` — same tiling logic with factor MC kernel
   - cuRAND states allocated for tile_size (not n_scenarios), reused across tiles
4. CLI and strategy integration:
   - `OptimizeConfig`: `covariance_method` ("sample" | "factor"), `FactorModelConfig`, `use_factor_mc`
   - `optimize_main.cpp`: branches on covariance method — factor model fit + factor MC or sample cov + Cholesky MC
   - `MeanCVaRConfig`: `use_factor_model`, `FactorModelConfig`, `use_factor_mc`
   - `MeanCVaRStrategy::allocate()`: factor model branch with graceful fallback to sample covariance on failure
   - `BacktestConfig`: `use_factor_model`, `FactorModelConfig`, `use_factor_mc` fields with JSON parsing
   - `backtest_main.cpp`: wires factor config into MeanCVaRConfig
   - Sample configs: `config/optimize_factor.json`, `config/backtest_factor.json`

### Tests (27 factor model tests, 164 total passing)

- `tests/test_factor_model.cpp`:
  - PCA: identity covariance k=N, known 2-factor synthetic data recovery, eigenvalue descending order, variance explained computation, full-rank reconstruction matches sample cov (within 1e-9), auto factor selection (min_variance_explained=0.90) (6)
  - Edge cases: single factor, k=N, single asset, n_factors > N clamped (4)
  - Error cases: T < 2 throws, N=0 throws (2)
  - Cholesky compatibility: CholeskyResult from factor model validates against reconstructed cov (1)
  - Dimensions, PD guarantee, mu extraction (3)
  - Factor MC CPU: mean convergence (CLT bound), covariance convergence (5% tolerance) (2)
  - Factor MC GPU: mean convergence, covariance convergence, reproducibility (same seed), cuRAND state reuse, GPU-CPU parity, equivalence to full Cholesky MC (6)
  - Tiled generation: full Cholesky tiled, factor tiled, output dimensions (3)

### Benchmarks (RTX 3060)

**Factor model fitting:**

| Assets | Factors | Time |
|---|---|---|
| 50 | 10 | 0.66 ms |
| 100 | 10 | 2.96 ms |
| 500 | 10 | 169 ms |

**Factor MC GPU vs Full Cholesky MC GPU (100K scenarios):**

| Assets | Factor MC | Full Cholesky | Speedup |
|---|---|---|---|
| 50 | 2.3 ms | 2.9 ms | 1.3x |
| 100 | 2.7 ms | 9.2 ms | **3.4x** |
| 500 | 11.0 ms | 172 ms | **15.6x** |

**Factor MC CPU (100K scenarios):**

| Assets | Time | GPU Speedup |
|---|---|---|
| 50 | 274 ms | 120x |
| 100 | 463 ms | 171x |
| 500 | 2,202 ms | 200x |

**Scenario generation throughput (100K x 500 assets):**
- Factor MC GPU: 5.0 G items/s
- Full Cholesky GPU: 291 M items/s — factor MC is **17x faster**

### Design Decisions

- **Factor structure exploited in Monte Carlo kernel** — O(Nk) per scenario vs O(N²) for full Cholesky. For N=500, k=10 this yields 15.6x speedup on GPU. The speedup scales with N/k.
- **Shared memory for B, mu, sqrt_D** — cooperative block load into shared memory (~24 KB for N=500, k=10). Factor Cholesky L_f stays in global memory (only k×k = 400 bytes for k=10).
- **curand_states.cuh shared header** — extracted CurandStates struct from monte_carlo.cu to enable cross-TU access. Forward declaration in public headers preserves the opaque-type pattern.
- **Graceful fallback** — MeanCVaRStrategy falls back to sample covariance if factor model fitting fails.
- **Tiled generation** — splits large scenarios into GPU-sized tiles, downloads each to host. cuRAND states allocated for tile_size, not total n_scenarios, reducing VRAM.

### Definition of Done

- Factor model covariance matches sample covariance for well-factored data ✅
- Factor MC GPU is 15.6x faster than full Cholesky for 500 assets ✅
- 164 tests pass, 0 regressions ✅
- Peak VRAM usage documented for all configurations ✅
- ADMM solver remains the bottleneck for 500-asset end-to-end (GPU x-update integration deferred)

---

## Phase 9 — Benchmarks, Validation & Documentation ✅

**Status:** Complete (ADMM benchmarks, cvxpy cross-validation, optimize CLI, sample configs, README rewrite)

**Goal:** Production-quality benchmarks, comprehensive validation, and polished documentation.

### Implemented

1. **ADMM Benchmark** (`benchmarks/bench_admm.cpp`):
   - Single ADMM solve: 2/5/10 assets × 10K/50K scenarios (6 configurations)
   - Efficient frontier: 3/5/10 assets × 5-point sweep (3 configurations)
   - Full pipeline: scenario generation + ADMM solve, 5/10 assets × 50K scenarios
   - Follows existing bench_monte_carlo.cpp patterns (Google Benchmark, DoNotOptimize, counters)
2. **Cross-validation suite** (`scripts/validate_cvxpy.py`):
   - Generates 9 test cases: 2/5/10 assets with unconstrained, box (40%), target return, and combined constraints
   - Solves R-U formulation with cvxpy + ECOS (SCS fallback)
   - Outputs reference JSON to `tests/data/validation/`
   - C++ test (`tests/test_validation.cpp`) loads reference JSON, generates scenarios from same mu/Sigma, runs ADMM, compares: weight L-inf < 0.05 (small) / 0.10 (larger), CVaR relative < 10%
   - Graceful skip when reference JSONs don't exist (builds without Python)
3. **Optimize CLI** (`apps/optimize_main.cpp`):
   - Full rewrite from stub to working CLI with `--config`, `--output`, `--help`
   - `OptimizeConfig` struct + JSON parsing (`src/optimizer/optimize_config.h/cpp`)
   - Supports direct mu/covariance or CSV price source
   - Two modes: single-point ADMM solve or efficient frontier
   - Report writers: `write_frontier_csv()` and `write_optimize_result_json()`
4. **Sample configs and data**:
   - `config/optimize_5asset.json` — 5-asset frontier optimization with box constraints
   - `config/backtest_5asset.json` — 5-asset backtest with all 4 strategies
   - `scripts/generate_sample_data.py` — synthetic GBM price data (252 days × 5 assets)
5. **Plotting scripts**:
   - `scripts/plot_frontier.py` — efficient frontier chart (matplotlib)
   - `scripts/plot_backtest.py` — equity curve comparison chart
6. **README rewrite**:
   - Architecture ASCII diagram, measured RTX 3060 benchmark numbers
   - Working build/test/run instructions, example CLI output
   - Validation workflow, math section, dependencies, VRAM budget
7. **Updated `.gitignore`**: results/, data/, __pycache__/, tests/data/validation/

### Measured Performance (RTX 3060)

| Benchmark | Configuration | Time | Iterations |
|---|---|---|---|
| ADMM solve | 2 assets, 10K scenarios | 57 ms | 65 |
| ADMM solve | 5 assets, 10K scenarios | 55 ms | 46 |
| ADMM solve | 10 assets, 10K scenarios | 64 ms | 40 |
| ADMM solve | 5 assets, 50K scenarios | 286 ms | 42 |
| ADMM solve | 10 assets, 50K scenarios | 406 ms | 44 |
| Efficient frontier | 3 assets, 5 pts, 20K scenarios | 414 ms | 200 total |
| Efficient frontier | 5 assets, 5 pts, 20K scenarios | 771 ms | 310 total |
| Efficient frontier | 10 assets, 5 pts, 20K scenarios | 1,233 ms | 375 total |
| Full pipeline | 5 assets, 50K scenarios | 324 ms | — |
| Full pipeline | 10 assets, 50K scenarios | 493 ms | — |

### Tests (137 passing, 1 skipped)

- `tests/test_validation.cpp`: parameterized test with graceful skip when no reference files exist
- `GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST` suppresses GTest warning for empty parameter set

### Files Added/Modified

| File | Status | Purpose |
|---|---|---|
| `benchmarks/bench_admm.cpp` | New | ADMM optimizer benchmarks |
| `benchmarks/CMakeLists.txt` | Modified | bench_admm target |
| `scripts/validate_cvxpy.py` | New | cvxpy cross-validation |
| `scripts/requirements.txt` | New | Python dependencies |
| `scripts/generate_sample_data.py` | New | Synthetic price data |
| `scripts/plot_frontier.py` | New | Efficient frontier plot |
| `scripts/plot_backtest.py` | New | Equity curve plot |
| `tests/test_validation.cpp` | New | C++ cross-validation tests |
| `tests/CMakeLists.txt` | Modified | test_validation target |
| `src/optimizer/optimize_config.h` | New | OptimizeConfig struct |
| `src/optimizer/optimize_config.cpp` | New | Config JSON parsing |
| `src/CMakeLists.txt` | Modified | New source files |
| `apps/optimize_main.cpp` | Rewritten | Stub → working CLI |
| `src/reporting/report_writer.h/cpp` | Modified | Frontier CSV + optimize JSON |
| `config/optimize_5asset.json` | New | Sample optimize config |
| `config/backtest_5asset.json` | New | Sample backtest config |
| `README.md` | Rewritten | Real benchmarks, architecture, examples |
| `.gitignore` | Modified | Generated output dirs |

---

## Phase 10 — Polish & Portfolio Integration

**Status:** Complete

**Goal:** Final polish, connect to portfolio site, ensure the repo tells a compelling story.

### Completed

- `scripts/plot_frontier.py` — matplotlib efficient frontier
- `scripts/plot_backtest.py` — equity curves, drawdown chart
- Comprehensive `.gitignore` — test output files, AI tooling configs, generated output dirs
- README suitable for a quant firm hiring manager (rewritten in Phase 9)
- LICENSE file (MIT)
- No TODOs, FIXMEs, or dead code in public-facing code
- CLAUDE.md updated: accurate config paths, models/ description, tiled generation reference
- README.md updated: 164 test count, bench_factor_model in run instructions
- Test output files (test_*.csv, test_*.json, nul) added to .gitignore

### Remaining (Optional)

1. GitHub Actions CI — see Phase 15 below
2. ~~Include sample output images in README~~ — done (docs/images/ committed in Phase 11)
3. ~~Link from portfolio site (Artemarius.github.io)~~ — done

### Definition of Done

- Repo is presentable to a hiring manager ✅
- README tells the story: motivation → what's implemented → results → how to build ✅
- No dead code, no TODOs in public-facing code ✅
- LICENSE file present ✅

---

## Phase 11 — 50-Stock Demo & GPU Crossover Analysis ✅

**Status:** Complete

**Goal:** Scale demo to 50 stocks where GPU advantage is demonstrable, benchmark at multiple asset counts to find the crossover point, fix a constraint parsing bug.

### Implemented

1. **Bug fix — `w_max` constraint for CSV-based configs**:
   - `optimize_config.cpp` only constructed position limit vectors when `mu_values` was non-empty (direct specification). CSV-based configs silently dropped `w_max`.
   - Fix: added `w_max_scalar` field to `OptimizeConfig`, store scalar unconditionally during parsing, construct vectors in `optimize_main.cpp` after `n_assets` is known from CSV.

2. **Extended ADMM benchmarks** (`benchmarks/bench_admm.cpp`):
   - Added 25/50/75/100-asset configs to all 6 benchmark functions (CPU + GPU single solve, frontier, full pipeline).
   - Parameterized `n_scenarios` in frontier benchmarks (50K for 25+ assets).

3. **50-stock data pipeline** (`scripts/download_data.py`):
   - Added `TICKERS_50`: 48 S&P 500 stocks across all 11 GICS sectors.
   - Added `argparse` with `--universe {10,50}` flag (default: 10 for backward compat).
   - Graceful handling of missing tickers with warnings.

4. **50-stock config files**:
   - `config/optimize_sp500_50.json`: factor model (k=10), factor MC, 100K scenarios, 10% position limits, GPU, 15-point frontier.
   - `config/backtest_sp500_50.json`: 50K scenarios, factor model, 0.5 shrinkage, all 4 strategies.

5. **Documentation updates**:
   - README: crossover benchmark tables, 50-stock demo instructions, updated examples.
   - ROADMAP: Phase 11 with benchmark results.

### GPU Crossover Analysis (RTX 3060, 50K scenarios)

**ADMM Solve:**

| Assets | CPU | GPU | Speedup |
|---|---|---|---|
| 2 | 214 ms | 297 ms | 0.72x |
| 5 | 208 ms | 320 ms | 0.65x |
| 10 | 312 ms | 422 ms | 0.74x |
| 25 | 1,141 ms | 734 ms | **1.55x** |
| 50 | 4,109 ms | 844 ms | **4.87x** |
| 75 | 13,219 ms | 1,266 ms | **10.4x** |
| 100 | 24,406 ms | 2,750 ms | **8.9x** |

**Crossover point: ~20 assets.** Below that, per-iteration kernel launch + host-device sync overhead exceeds compute savings. Above 25, GPU parallel scenario evaluation dominates. CPU time scales roughly O(N^2) per iteration (scenario matrix dot product), while GPU scales sub-linearly thanks to thread-level parallelism.

**Full Pipeline (scenario generation + ADMM):**

| Assets | CPU | GPU | Speedup |
|---|---|---|---|
| 5 | 234 ms | 271 ms | 0.86x |
| 10 | 320 ms | 289 ms | **1.11x** |
| 25 | 1,156 ms | 531 ms | **2.18x** |
| 50 | 4,188 ms | 1,016 ms | **4.12x** |
| 100 | 27,531 ms | 1,375 ms | **20.0x** |

### 50-Stock Demo Results

- Efficient frontier: 15 points computed in ~43 seconds (GPU), 12/15 converged within 500 iterations.
- Position limits (10% max) properly enforced — verified in output weights.
- Min-CVaR portfolio diversifies across defensive sectors: consumer staples, utilities, health care.
- Backtest: MeanCVaR achieves Sharpe 2.24, outperforming all baselines on 48 stocks.

### Definition of Done

- `w_max` constraint works for CSV-based configs (verified in 50-stock output) ✅
- Benchmarks run with 25/50/75/100-asset configs ✅
- GPU crossover point documented (~20 assets) ✅
- 50-stock optimization and backtest produce valid results ✅
- 173 tests still pass (now 223 after Phase 12) ✅

---

## Phase 12 — ADMM Convergence Improvements ✅

**Status:** Complete

**Goal:** Achieve reliable convergence for 50-100 asset problems. Previously 12/15 frontier points converged at 50 stocks within 500 iterations — this phase targeted 15/15.

### Background

The vanilla ADMM solver used fixed-learning-rate proximal gradient descent for the x-update with Boyd et al. 2011 adaptive rho. This worked well up to ~25 assets but struggled at 50+ assets because:

- The R-U objective landscape becomes increasingly ill-conditioned with more assets
- Fixed learning rate `x_update_lr = 0.01` was conservative for large N (too slow) but unstable if increased naively
- The proximal gradient x-update took 20 inner steps per ADMM iteration — insufficient for high-dimensional subproblems
- Adaptive rho oscillated between primal/dual balance without converging when both residuals were large

### Implemented

1. **Over-relaxation** (Boyd 2011 S3.4.3, Eq. 3.19-3.20):
   - `AdmmConfig::alpha_relax` parameter (default 1.0 = vanilla ADMM, recommended 1.5)
   - Blends x and z_prev in z/u-update: `x_hat = alpha * x + (1 - alpha) * z_prev`
   - Implemented in both CPU and GPU ADMM paths

2. **Backtracking line search** (Armijo condition, Nocedal & Wright 2006):
   - Replaces fixed learning rate in x-update proximal gradient step
   - Armijo condition: `f(x - lr*grad) <= f(x) - c * lr * ||grad||^2` with `c = 1e-4`
   - Shrink factor 0.5, max 10 backtracking steps per inner iteration
   - Standalone module: `src/optimizer/line_search.h/cpp` (reusable)
   - Integrated into both CPU and GPU x-update paths

3. **Anderson acceleration** (type-I, Zhang et al. 2020):
   - `AdmmConfig::anderson_depth` parameter (default 0 = disabled, recommended 3-5)
   - Accelerates the (z, zeta) fixed-point iteration by extrapolating from last m iterates
   - Standalone module: `src/optimizer/anderson_acceleration.h/cpp`
   - Safeguarded: rejects acceleration when residual increases, resets on NaN/non-finite
   - Resets on rho change to avoid stale iterate history
   - ColPivHouseholderQR least-squares solve for mixing coefficients

4. **Residual balancing** (Wohlberg 2017):
   - `AdmmConfig::residual_balancing` flag (default false)
   - Continuously adjusts rho to equalize normalized residuals: `rho *= sqrt((r_pri/eps_pri)/(r_dual/eps_dual))`
   - Per-iteration rho change clamped to `[1/rho_balance_tau, rho_balance_tau]`
   - **Most impactful improvement for larger problems** — reduced 25-asset iterations from 1700+ (non-converging) to 264

5. **Adaptive x-update step count**:
   - `effective_x_steps = max(x_update_steps, n_assets)` — scales inner iterations with problem size
   - Eliminates under-solving the x-subproblem for high-dimensional problems

6. **NaN safeguards**:
   - Anderson acceleration checks `allFinite()` before accepting extrapolated state
   - Both CPU and GPU paths detect non-finite x/z/zeta and abort early with warning
   - Prevents divergence from propagating through the iterate sequence

### New Files

| File | Purpose |
|---|---|
| `src/optimizer/anderson_acceleration.h` | Anderson accelerator class (standalone) |
| `src/optimizer/anderson_acceleration.cpp` | Type-I Anderson with ring buffer, QR solve |
| `src/optimizer/line_search.h` | Backtracking line search config and interface |
| `src/optimizer/line_search.cpp` | Armijo backtracking implementation |
| `tests/test_anderson_acceleration.cpp` | 12 unit tests for Anderson accelerator |
| `tests/test_line_search.cpp` | 13 unit tests for line search |
| `tests/test_convergence.cpp` | 10 convergence regression tests |

### Modified Files

| File | Changes |
|---|---|
| `src/optimizer/admm_solver.h` | New config fields: `alpha_relax`, `residual_balancing`, `rho_balance_tau`, `anderson_depth` |
| `src/optimizer/admm_solver.cpp` | Line search in x-update, over-relaxation, Anderson acceleration, residual balancing, NaN guards (both CPU and GPU paths) |
| `src/optimizer/optimize_config.cpp` | JSON parsing for new ADMM config fields |
| `src/CMakeLists.txt` | Added anderson_acceleration.cpp, line_search.cpp |
| `tests/CMakeLists.txt` | Added test_anderson_acceleration, test_line_search, test_convergence targets |

### Convergence Results

**Iteration counts (with alpha_relax=1.5, anderson_depth=3, residual_balancing=true):**

| Problem | Before Phase 12 | After Phase 12 | Improvement |
|---|---|---|---|
| 5-asset unconstrained | 55 iters | 35 iters | 36% fewer |
| 10-asset unconstrained | 37 iters | 26 iters | 30% fewer |
| 25-asset unconstrained | 1700+ (failed) | 264 iters | Converges |
| 5-asset frontier (5 pts) | 20-99 iters/pt | 20-38 iters/pt | ~50% fewer |

**Backward compatibility:** All new features default to disabled (alpha_relax=1.0, anderson_depth=0, residual_balancing=false). Existing 16 ADMM solver tests pass unchanged.

### Tests (223 total, 0 failures)

- `test_anderson_acceleration`: 12/12 passed — ring buffer, QR solve, safeguard, reset, dimensions
- `test_line_search`: 13/13 passed — Armijo condition, multi-dimensional, edge cases
- `test_convergence`: 10/10 passed — 2/10/25-asset problems, frontier monotonicity, warm start, adaptive rho, constrained, deterministic, parameterized scaling
- All existing tests: 188/188 passed (0 regressions)

### Design Decisions

- **Conservative defaults** — new features disabled by default to prevent breaking existing users. Enable via config for specific problems.
- **Standalone modules** — Anderson accelerator and line search are reusable components, not coupled to the ADMM solver.
- **Residual balancing is the key enabler** for larger problems. Over-relaxation and Anderson provide incremental improvement; residual balancing fixes the fundamental rho imbalance that causes divergence at 25+ assets.
- **NaN safeguards are essential** — Anderson extrapolation can produce non-finite values that pass the residual safeguard check. Explicit finite-check prevents cascading failures.

### Definition of Done

- 25-asset problem converges in 264 iterations (was failing at 1700+) ✅
- No regression on small problems (2-10 assets) ✅
- 223 tests pass, 0 failures ✅
- Backward-compatible defaults ✅

---

## Phase 13 — Black-Litterman Model

**Status:** Not started

**Goal:** Implement the Black-Litterman model for combining market equilibrium returns with subjective views. Produces a posterior mu/Sigma that feeds directly into the existing Monte Carlo + ADMM pipeline.

### Background

The current pipeline estimates expected returns (mu) from historical sample means, which are notoriously noisy. Black-Litterman (1992) solves this by:

1. Starting from **market equilibrium returns** (implied by market-cap weights and a risk aversion parameter)
2. Blending in **investor views** (absolute or relative, with confidence levels)
3. Producing a **posterior distribution** N(mu_BL, Sigma_BL) that tilts away from equilibrium only where views are expressed

This is the industry standard for return estimation at most institutional asset managers.

### Implementation Plan

1. `src/models/black_litterman.h/cpp`:
   - `BlackLittermanConfig` struct:
     - `risk_aversion` (delta): scalar, default 2.5. Can be estimated from market Sharpe ratio: delta = (E[r_m] - r_f) / sigma_m^2
     - `tau`: scalar uncertainty on equilibrium covariance (default 0.05). Controls how much views shift the posterior
     - `views`: vector of `View` structs
   - `View` struct:
     - `P`: pick vector (1 × N) — which assets the view is about. Absolute view: P = [0,0,1,0,0] (asset 3). Relative view: P = [0,1,0,-1,0] (asset 2 outperforms asset 4)
     - `q`: scalar — expected return (absolute) or return difference (relative)
     - `confidence`: scalar in (0, 1] — maps to view uncertainty omega = (1/confidence - 1) * P * tau * Sigma * P'
   - `BlackLittermanResult` struct: `mu_bl` (VectorXd), `sigma_bl` (MatrixXd), `implied_returns` (VectorXd)
   - `compute_implied_returns(sigma, w_market, delta)` → pi = delta * Sigma * w_mkt (reverse optimization)
   - `compute_black_litterman(sigma, w_market, views, config)` → `BlackLittermanResult`
     - Posterior mean: mu_BL = [(tau * Sigma)^{-1} + P' * Omega^{-1} * P]^{-1} * [(tau * Sigma)^{-1} * pi + P' * Omega^{-1} * q]
     - Posterior cov: Sigma_BL = Sigma + [(tau * Sigma)^{-1} + P' * Omega^{-1} * P]^{-1}
     - Reference: He & Litterman, *The Intuition Behind Black-Litterman Model Portfolios*, Goldman Sachs, 1999

2. Config integration:
   - `OptimizeConfig`: add `use_black_litterman`, `BlackLittermanConfig`, `market_cap_weights` (or estimate equal-weight as fallback)
   - `optimize_main.cpp`: if BL enabled, compute mu_BL/Sigma_BL before scenario generation
   - Views specified in JSON config:
     ```json
     "views": [
       {"assets": [2], "return": 0.10, "confidence": 0.8},
       {"assets": [1, -3], "return": 0.02, "confidence": 0.6}
     ]
     ```

3. Backtest integration:
   - New `BlackLittermanStrategy` (or a flag on `MeanCVaRStrategy`): uses BL posterior for mu/Sigma estimation at each rebalance
   - Market-cap weights can be loaded from CSV or estimated from price × shares data

### Tests

- Implied returns: known Sigma + w_mkt → verify pi = delta * Sigma * w_mkt
- No views: posterior mu_BL = implied returns, Sigma_BL = (1 + tau) * Sigma
- Single absolute view: posterior tilts toward viewed asset
- Relative view: long/short pair reflects expected outperformance
- High confidence → posterior near view; low confidence → posterior near equilibrium
- Full pipeline: BL posterior → Monte Carlo scenarios → ADMM → valid efficient frontier
- Cross-validation: 2-asset BL against analytical formula

### Definition of Done

- `compute_black_litterman()` produces correct posterior for known test cases
- Views specified via JSON config, parsed and validated
- Full pipeline works: BL → scenarios → ADMM → frontier
- Backtest strategy available
- All existing tests still pass

---

## Phase 14 — Python Bindings (pybind11)

**Status:** Not started

**Goal:** Wrap `cuda_portfolio_lib` with pybind11 so the GPU optimizer is callable from Python/Jupyter. Dramatically increases the audience — quant researchers can use the CUDA kernels without writing C++.

### Scope

Expose the core library functions, not the CLI apps. Users import the module and call functions directly:

```python
import cuda_portfolio as cpo

# Generate correlated scenarios on GPU
scenarios = cpo.generate_scenarios(mu, cov, n_scenarios=100_000, seed=42)

# Compute risk
risk = cpo.compute_risk(scenarios, weights, confidence=0.95)
print(f"CVaR: {risk.cvar:.4f}, VaR: {risk.var:.4f}")

# Optimize
result = cpo.optimize(scenarios, mu, target_return=0.08, w_max=0.10)
print(f"Weights: {result.weights}, CVaR: {result.cvar:.4f}")

# Efficient frontier
frontier = cpo.efficient_frontier(scenarios, mu, n_points=20)

# Black-Litterman (Phase 13)
bl = cpo.black_litterman(cov, market_weights, views, tau=0.05)
```

### Implementation Plan

1. **Build system**:
   - Add pybind11 via FetchContent
   - New CMake target `cuda_portfolio_python` (`pybind11_add_module`)
   - Conditional build: `option(BUILD_PYTHON_BINDINGS "Build pybind11 module" OFF)`
   - Output: `cuda_portfolio.pyd` (Windows) / `cuda_portfolio.so` (Linux)

2. **Binding layer** (`python/bindings.cpp`):
   - Eigen ↔ NumPy zero-copy via pybind11's Eigen support (`#include <pybind11/eigen.h>`)
   - Scenario generation: `generate_scenarios(mu, cov, n_scenarios, seed)` → numpy array
   - Factor model: `fit_factor_model(returns, n_factors)` → FactorModelResult with numpy arrays
   - Risk: `compute_risk(scenarios, weights, confidence)` → RiskResult (as Python dataclass or named dict)
   - Optimizer: `optimize(scenarios, mu, config_dict)` → dict with weights, cvar, iterations
   - Efficient frontier: `efficient_frontier(scenarios, mu, n_points, config_dict)` → list of dicts
   - Backtest: potentially too complex for v1 — defer to v2

3. **GPU resource management**:
   - cuRAND states managed internally (create on first call, reuse, destroy on module unload)
   - ScenarioMatrix stays on GPU between `generate_scenarios` and `optimize` when used together
   - Context manager for explicit GPU memory control:
     ```python
     with cpo.GpuContext(seed=42) as ctx:
         scenarios = ctx.generate_scenarios(mu, cov, 100_000)
         result = ctx.optimize(scenarios, mu)
     # GPU memory freed on exit
     ```

4. **Installation**:
   - `pip install .` via `setup.py` or `pyproject.toml` with CMake extension
   - Requires: CUDA toolkit + C++ compiler (same as building the library)
   - Publish pre-built wheels for Windows + CUDA 12.x (stretch goal)

5. **Example notebooks** (`notebooks/`):
   - `01_quickstart.ipynb`: generate scenarios, optimize, plot frontier
   - `02_backtest_comparison.ipynb`: compare strategies using Python plotting
   - `03_black_litterman.ipynb`: BL views → GPU optimization

### Tests

- NumPy array round-trip: pass numpy array → get numpy array back, values match
- GPU scenario generation from Python: mean/covariance convergence
- Optimize from Python: matches C++ result for same inputs
- Error handling: Python exceptions for invalid inputs (wrong dimensions, non-PD matrix)
- Memory: no GPU memory leaks after repeated calls

### Dependencies

| Library | Purpose | Acquisition |
|---|---|---|
| pybind11 | C++/Python binding | FetchContent |
| numpy | Array interop | pip (user's env) |
| pytest | Python-side tests | pip (dev) |

### Definition of Done

- `import cuda_portfolio` works after `pip install .`
- Core functions callable from Python with numpy arrays
- GPU memory managed correctly (no leaks)
- Example notebook runs end-to-end
- README section with Python usage examples

---

## Phase 15 — GitHub Actions CI

**Status:** Not started

**Goal:** Automated build and test pipeline so every push and PR is verified. Demonstrates engineering rigor to reviewers.

### Why This Is Hard

CUDA CI on GitHub Actions is non-trivial because:

- **GitHub-hosted runners have no GPU.** Standard `ubuntu-latest` / `windows-latest` runners lack NVIDIA hardware entirely. CUDA kernels cannot execute.
- **CUDA Toolkit is large.** The full toolkit is ~4-6 GB. Installing it per-run adds 3-5 minutes and consumes cache budget.
- **Windows + CUDA + Visual Studio** is the primary build target, but Windows runners are slower and more expensive (2x minute multiplier).

### Strategy: Two-Tier CI

Split into a **build-and-CPU-test tier** (runs on every push, no GPU needed) and an optional **GPU tier** (self-hosted runner or manual trigger).

#### Tier 1 — Build + CPU Tests (GitHub-hosted, every push)

Runs on `ubuntu-latest`. Installs CUDA toolkit (compiler only — no runtime GPU needed for compilation). Builds the full project and runs the subset of tests that don't require GPU execution.

1. **CUDA Toolkit installation**: Use the [`Jimver/cuda-toolkit`](https://github.com/Jimver/cuda-toolkit) action. Install compiler components only (`cuda-compiler`, `cuda-cudart-dev`, `libcurand-dev`) — skip drivers and samples. Cache the installation via `actions/cache` keyed on toolkit version.
2. **Build matrix**:
   - Linux (ubuntu-latest) + GCC 12 + CUDA 12.8 + Ninja — primary CI target
   - Windows (windows-latest) + MSVC 2022 + CUDA 12.8 — optional, expensive (2x minutes)
3. **CMake configure + build**: `cmake -B build -DCMAKE_BUILD_TYPE=Release -G Ninja && cmake --build build --config Release`
4. **CPU-only tests**: Run tests that don't launch CUDA kernels. This requires a test filtering mechanism:
   - Option A: Google Test `--gtest_filter=-*GPU*` naming convention — rename GPU-dependent tests to include `GPU` in the test name
   - Option B: CMake `ctest -L cpu` label filtering — add `set_tests_properties(test_X PROPERTIES LABELS "cpu")` for CPU-safe tests
   - Option C: Compile-time `#ifdef CUDA_AVAILABLE` guard — detect GPU at runtime, skip gracefully
   - **Recommended: Option C** — `cudaGetDeviceCount()` at test startup, `GTEST_SKIP()` for GPU tests when count == 0. Zero changes to test naming, works locally and in CI identically.
5. **What gets validated**:
   - Compilation succeeds (catches syntax errors, missing includes, CUDA kernel compilation errors)
   - All CPU-path code is correct: data loading, returns, projections, constraints, Cholesky, factor model fitting, backtest engine logic, report writers, config parsing
   - ~60-70% of the test suite runs without a GPU

#### Tier 2 — GPU Tests (self-hosted runner, manual/nightly)

Full test suite including CUDA kernel execution. Only feasible with a self-hosted runner that has an NVIDIA GPU.

1. **Self-hosted runner setup**: Personal machine (RTX 3060) registered as a GitHub Actions runner. Tagged with `self-hosted, gpu, cuda`.
2. **Trigger**: `workflow_dispatch` (manual) or `schedule` (nightly cron). Not on every push — the machine isn't always on.
3. **Runs**: Full `ctest --test-dir build -C Release --output-on-failure` + benchmarks.
4. **Benchmark regression detection** (stretch goal): Save benchmark JSON output, compare against baseline, flag regressions > 10%.

### Tasks

1. Add runtime GPU detection utility:
   - `src/utils/cuda_utils.h`: add `bool has_cuda_device()` — wraps `cudaGetDeviceCount`, returns false on error or count == 0
   - Use in test fixtures: `if (!has_cuda_device()) GTEST_SKIP() << "No CUDA device";`
2. Tag existing GPU-dependent tests:
   - Audit each test file, add skip guard to tests that call CUDA kernels (monte_carlo GPU, risk GPU, ADMM GPU, component CVaR GPU, factor MC GPU, admm_kernels GPU)
   - CPU tests (data, projections, constraints, backtest, reporting, factor model fit) run unchanged
3. Create `.github/workflows/ci.yml`:
   - Tier 1 job: `build-and-test-cpu` on `ubuntu-latest`
   - Install CUDA toolkit via `Jimver/cuda-toolkit@v0.2.16` with caching
   - Build with Ninja, run `ctest` (GPU tests auto-skip)
4. Create `.github/workflows/gpu-tests.yml`:
   - Tier 2 job: `test-gpu` on `[self-hosted, gpu]`
   - `workflow_dispatch` + optional nightly cron
   - Full test suite + benchmarks
5. Add CI status badge to README

### Estimated CI Run Times

| Tier | Runner | CUDA Install | Build | Tests | Total |
|---|---|---|---|---|---|
| Tier 1 (cached) | ubuntu-latest | ~30s (cached) | ~2-3 min | ~10s | **~3-4 min** |
| Tier 1 (cold) | ubuntu-latest | ~3-4 min | ~2-3 min | ~10s | **~6-7 min** |
| Tier 2 | self-hosted GPU | 0 (pre-installed) | ~1-2 min | ~30s | **~2-3 min** |

### Definition of Done

- Every push to `main` triggers Tier 1 (build + CPU tests pass on Linux)
- GPU tests gracefully skip with `GTEST_SKIP()` when no device is present
- Self-hosted GPU workflow runs full suite on manual trigger
- CI badge in README shows passing status
- No test regressions — all 223+ tests still pass locally

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
