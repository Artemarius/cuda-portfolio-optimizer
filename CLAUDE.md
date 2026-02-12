# CLAUDE.md

## Architecture

```
src/
  core/          — Fundamental types, config, portfolio result structs
  data/          — Market data loader (CSV), return computation, universe definition
  models/        — Return distribution models: PCA factor model, factor Monte Carlo, tiled generation
  simulation/    — GPU Monte Carlo scenario generator (correlated returns via Cholesky + cuRAND)
  risk/          — CVaR computation (CUDA), VaR, volatility, drawdown metrics
  optimizer/     — Convex optimizer: ADMM solver (C++/CUDA), projections, efficient frontier
  constraints/   — Portfolio constraints: position limits, leverage, turnover, sector
  backtest/      — Rolling-window backtesting engine with transaction costs
  reporting/     — Efficient frontier, risk decomposition, strategy comparison (CSV/JSON output)
  utils/         — Timer, logging, CUDA helpers, math utilities
apps/            — CLI executables (optimize, backtest) — link against cuda_portfolio_lib
tests/           — Google Test unit tests
benchmarks/      — GPU vs CPU performance comparison (Google Benchmark)
scripts/         — Python helpers: data download, cvxpy validation, plotting
```

### Build Targets

| Target | Type | Description |
|---|---|---|
| `cuda_portfolio_lib` | Static library | All src/ modules — the core library |
| `optimize` | Executable | CLI: config → optimize → report |
| `backtest` | Executable | CLI: config → backtest → report |
| `bench_*` | Executables | Google Benchmark GPU vs CPU comparisons |
| `test_*` | Executables | Google Test unit tests |

Library and executables are separate — `apps/` links against `cuda_portfolio_lib`. This enables component reuse: someone can use the simulator or optimizer independently.

## Key Technical Decisions

- **C++17** with CUDA 12.8, `CMAKE_CUDA_ARCHITECTURES 86` (RTX 3060)
- **Windows primary** (Visual Studio 2022 / Ninja), but no platform-specific APIs — code is de facto portable
- **CMake 3.20+** with native CUDA language support (`project(... LANGUAGES CXX CUDA)`) — do NOT use `find_package(CUDA)`
- **FetchContent** for all dependencies except CUDA toolkit components — repo must be self-contained
- **Custom CUDA kernels** for Monte Carlo simulation and CVaR computation — no cuOpt or external solver libraries
- **Eigen3** for CPU-side linear algebra (covariance matrices, Cholesky decomposition)
- **cuRAND** for GPU random number generation
- **CUB** for GPU sort and reduction primitives (ships with CUDA 12.8)
- **ADMM solver** implemented in C++/CUDA for the constrained Mean-CVaR optimization. GPU x-update via `k_evaluate_ru_objective` kernel with pre-allocated device buffers (`GpuAdmmBuffers`), CPU z-update/u-update (cheap projections), double-precision final refinement
- **Opaque GPU buffer pattern**: `GpuAdmmBuffers` (like `CurandStates`) — struct defined in `.cuh`, forward-declared in `.h`, factory/deleter/guard in public API. Avoids per-call cudaMalloc/cudaFree (~6000 calls per ADMM solve)
- **Dual precision strategy:** `float` for GPU scenario matrix and risk computation (throughput + VRAM), `double` for optimizer convergence checks and CPU-side estimation

## Code Style & Conventions

- Google C++ Style Guide baseline:
  - `snake_case` for functions and variables, `PascalCase` for types/classes
  - RAII everywhere, no raw `new`/`delete`
  - `constexpr` where possible
- CUDA kernels: prefix with `k_` (e.g., `k_monte_carlo_simulate`, `k_compute_cvar`)
- All public APIs must have doc comments (doxygen-style `///`)
- Every module must have unit tests
- Mathematical formulas must be referenced in comments (paper + equation number when applicable)
- File naming:
  - Headers: `.h` (not `.hpp`)
  - C++ source: `.cpp`
  - CUDA source: `.cu`
  - CUDA headers with device code: `.cuh` (if needed, prefer `.h` with `__host__ __device__`)
  - Tests: `test_<module>.cpp`
  - Benchmarks: `bench_<module>.cpp`

## Core Types (src/core/types.h)

```cpp
// GPU path: float for throughput and VRAM efficiency
using Scalar = float;
// CPU path: double for optimizer convergence and estimation
using ScalarCPU = double;
using Index = int;

// Eigen typedefs
using VectorXs = Eigen::VectorXf;
using MatrixXs = Eigen::MatrixXf;
using VectorXd = Eigen::VectorXd;
using MatrixXd = Eigen::MatrixXd;
```

## Build & Run

```bash
# Configure (Visual Studio generator)
cmake -B build -G "Visual Studio 17 2022"

# Configure (Ninja — faster incremental builds)
cmake -B build -DCMAKE_BUILD_TYPE=Release -G Ninja

# Build
cmake --build build --config Release

# Run tests
ctest --test-dir build -C Release --output-on-failure

# Portfolio optimization
./build/Release/optimize --config config/optimize_5asset.json --output results/

# Backtest
./build/Release/backtest --config config/backtest_5asset.json --output results/backtest/

# GPU vs CPU benchmarks
./build/Release/bench_monte_carlo
./build/Release/bench_factor_model
```

## Dependencies

All resolved via FetchContent except CUDA toolkit components. `git clone` + `cmake` + `build` with no manual steps.

| Library | Purpose | Acquisition |
|---|---|---|
| CUDA 12.8 | Compute kernels, cuRAND, CUB | Pre-installed |
| CUB | GPU sort and reduction primitives | Ships with CUDA 12.8 |
| Eigen3 | CPU linear algebra, Cholesky, covariance | FetchContent |
| nlohmann/json | Config and output serialization | FetchContent |
| Google Test | Unit tests | FetchContent |
| Google Benchmark | CPU vs GPU performance comparison | FetchContent |
| spdlog | Logging | FetchContent |

## Development Priorities

1. **Mathematical correctness** — optimization must produce provably correct efficient frontiers. Validate against known analytical solutions (2-asset closed-form) and against cvxpy/scipy
2. **Formula traceability** — every formula in code references the source paper and equation number
3. **Correctness before performance** — ADMM logic proven correct on CPU, then x-update bottleneck moved to CUDA (done: `admm_solve(ScenarioMatrix&, ...)` GPU overload)
4. **GPU/CPU parity** — both CPU and GPU paths exist for all compute-heavy operations (Monte Carlo, ADMM x-update, CVaR). Benchmark and document speedups
5. **Component independence** — optimizer, simulator, and backtester are independently usable via the static library
6. **Realistic constraints** — position limits, turnover, transaction costs. Not a toy optimizer

## VRAM Budget (RTX 3060 6GB)

Peak usage for the largest target configuration (100K scenarios × 500 assets):

| Component | Size |
|---|---|
| Scenario matrix (float32, 100K × 500) | 200 MB |
| cuRAND states (100K) | 5 MB |
| Cholesky factor L (500 × 500) | 1 MB |
| Loss vector (100K) | 0.4 MB |
| ADMM working buffers | ~200 MB |
| **Total** | **~406 MB** |
| **Available** | **6144 MB** |

No VRAM pressure. For scaling beyond 500 assets or 500K scenarios, tiled generation is implemented in `src/models/tiled_scenario_generator.h`.

## GPU Memory Patterns

- **Scenario matrix:** column-major layout (N_scenarios × N_assets). When computing portfolio loss (dot product of weights × one scenario row), all threads in a warp access the same column → coalesced reads.
- **cuRAND state:** initialize once, store in device memory, reuse across optimizer iterations. Initialization is expensive (~2ms for 100K states).
- **Weight vector:** small enough (500 × 4 = 2KB) to fit in shared memory for the loss computation kernel.
- **ADMM buffers (`GpuAdmmBuffers`):** pre-allocated once per `admm_solve` call — `d_weights` (float), `d_sum_excess` (double), `d_grad_w` (double × N_assets), `d_count` (int). Zeroed via `cudaMemset` + weights uploaded via `cudaMemcpy` each inner step. Eliminates ~6000 cudaMalloc/cudaFree pairs per solve.

## Key Mathematical References

- **CVaR (Conditional Value-at-Risk):** CVaR_α = E[L | L ≥ VaR_α] — average loss in the worst α% of scenarios
- **Mean-CVaR optimization:** min CVaR_α(w) s.t. E[r'w] ≥ μ_target, constraints on w
- **Rockafellar-Uryasev formulation:** reformulates CVaR minimization as a linear program over scenarios. See: Rockafellar & Uryasev, *Optimization of Conditional Value-at-Risk*, J. Risk 2000
- **Cholesky decomposition:** Σ = LLᵀ → correlated samples = L × z where z ~ N(0,I)
- **ADMM:** splits constrained optimization into simpler subproblems. See: Boyd et al., *Distributed Optimization and Statistical Learning via ADMM*, 2011. Adaptive ρ: §3.4.1
- **Factor model:** R = Bf + ε, Σ = BΣ_f Bᵀ + D — reduces dimensionality of covariance estimation
- **Shrinkage estimation:** Ledoit & Wolf, *Honey, I Shrunk the Sample Covariance Matrix*, 2004
- **Simplex projection:** Duchi et al., *Efficient Projections onto the l1-Ball*, 2008

## Validation Strategy

Optimization results are validated at multiple levels:

1. **Analytical:** 2-asset closed-form efficient frontier (Markowitz)
2. **Cross-reference:** 5–10 asset problems solved with both ADMM and Python cvxpy/scipy → compare weights, objective, constraints
3. **Statistical:** Monte Carlo convergence tests (sample mean/covariance → true μ/Σ as N → ∞)
4. **Structural:** efficient frontier monotonicity, CVaR ≥ VaR, constraint satisfaction, component CVaR sums to total

Cross-validation workflow: `scripts/validate_cvxpy.py` solves reference problems and dumps JSON. C++ test loads JSON and compares.
