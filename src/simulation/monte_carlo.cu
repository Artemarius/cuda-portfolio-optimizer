#include "simulation/monte_carlo.h"

#include <random>
#include <vector>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <spdlog/spdlog.h>

#include "simulation/curand_states.cuh"
#include "utils/cuda_utils.h"

namespace cpo {

// ── cuRAND initialization kernel ───────────────────────────────────

/// Initialize cuRAND states. Each thread gets a unique subsequence.
/// @param states Output device array of cuRAND states.
/// @param seed Base seed for all states.
/// @param n Total number of states.
__global__ void k_init_curand(curandState_t* states, uint64_t seed, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        // Each thread: same seed, different subsequence, offset 0.
        curand_init(seed, static_cast<unsigned long long>(tid), 0,
                    &states[tid]);
    }
}

CurandStatesGuard create_curand_states(Index n_states, uint64_t seed) {
    // Allocate via new (not make_unique) since CurandStatesGuard uses custom deleter.
    auto* raw = new CurandStates();
    raw->n_states = n_states;

    size_t alloc_bytes = static_cast<size_t>(n_states) * sizeof(curandState_t);
    CUDA_CHECK(cudaMalloc(&raw->d_states, alloc_bytes));
    spdlog::info("cuRAND states: allocated {} states ({:.1f} MB)", n_states,
                 alloc_bytes / (1024.0 * 1024.0));

    int threads = 256;
    int blocks = (n_states + threads - 1) / threads;
    k_init_curand<<<blocks, threads>>>(raw->d_states, seed, n_states);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    spdlog::debug("cuRAND states initialized (seed={})", seed);
    return CurandStatesGuard(raw);
}

void destroy_curand_states(CurandStates* states) {
    if (states && states->d_states) {
        cudaFree(states->d_states);
        states->d_states = nullptr;
    }
    delete states;
}

// ── Monte Carlo GPU kernel ─────────────────────────────────────────

/// Generate correlated return scenarios on the GPU.
///
/// One thread per scenario. Two-phase approach:
///   Phase 1: Generate z[j] ~ N(0,1) for j = 0..n_assets-1, store in scenario columns.
///   Phase 2: Compute r[i] = mu[i] + sum_{j=0}^{i} L[i,j]*z[j] in reverse order (i = n-1..0).
///            Writing r[i] from high to low is safe because L is lower-triangular:
///            r[i] only reads z[0..i], so overwriting z[i] doesn't affect z[k<i].
///
/// Layout: scenario matrix is column-major, element (scenario, asset) at
///         asset * n_scenarios + scenario.
/// L is row-major: L[i,j] at i * n_assets + j.
///
/// @param d_scenarios Output scenario matrix (column-major, n_scenarios x n_assets).
/// @param d_L Cholesky factor L (row-major float, n_assets x n_assets).
/// @param d_mu Mean return vector (float, n_assets).
/// @param states cuRAND states (one per scenario).
/// @param n_scenarios Number of scenarios.
/// @param n_assets Number of assets.
__global__ void k_monte_carlo_simulate(float* __restrict__ d_scenarios,
                                       const float* __restrict__ d_L,
                                       const float* __restrict__ d_mu,
                                       curandState_t* __restrict__ states,
                                       int n_scenarios, int n_assets) {
    int sid = blockIdx.x * blockDim.x + threadIdx.x;
    if (sid >= n_scenarios) return;

    // Load this thread's RNG state into registers.
    curandState_t local_state = states[sid];

    // Phase 1: Generate z ~ N(0,1) and store in scenario matrix columns.
    for (int j = 0; j < n_assets; ++j) {
        float z = curand_normal(&local_state);
        d_scenarios[j * n_scenarios + sid] = z;
    }

    // Phase 2: Compute r = mu + L*z in reverse order (overwrite z in-place).
    // Reverse order is safe because L is lower-triangular:
    //   r[i] = mu[i] + sum_{j=0}^{i} L[i,j] * z[j]
    // When writing r[i], we only read z[0..i]. Since we write from i=n-1 down,
    // z[k] for k < i is never overwritten before it's needed.
    for (int i = n_assets - 1; i >= 0; --i) {
        float sum = d_mu[i];
        for (int j = 0; j <= i; ++j) {
            // L is row-major: L[i,j] at i * n_assets + j
            sum += d_L[i * n_assets + j] *
                   d_scenarios[j * n_scenarios + sid];
        }
        d_scenarios[i * n_scenarios + sid] = sum;
    }

    // Write back updated RNG state for potential reuse.
    states[sid] = local_state;
}

// ── Host orchestration ─────────────────────────────────────────────

ScenarioMatrix generate_scenarios_gpu(const VectorXd& mu,
                                      const CholeskyResult& cholesky,
                                      const MonteCarloConfig& config,
                                      CurandStates* states) {
    const Index n_assets = cholesky.n;
    const Index n_scenarios = config.n_scenarios;

    if (mu.size() != n_assets) {
        throw std::runtime_error(
            "generate_scenarios_gpu: mu size (" +
            std::to_string(mu.size()) + ") != n_assets (" +
            std::to_string(n_assets) + ")");
    }

    spdlog::info("Monte Carlo GPU: {} scenarios x {} assets (seed={})",
                 n_scenarios, n_assets, config.seed);

    // Convert mu to float.
    std::vector<Scalar> h_mu(n_assets);
    for (Index i = 0; i < n_assets; ++i) {
        h_mu[i] = static_cast<Scalar>(mu(i));
    }

    // Upload L and mu to device.
    Scalar* d_L = nullptr;
    Scalar* d_mu = nullptr;
    size_t L_bytes = cholesky.L_flat.size() * sizeof(Scalar);
    size_t mu_bytes = static_cast<size_t>(n_assets) * sizeof(Scalar);

    CUDA_CHECK(cudaMalloc(&d_L, L_bytes));
    CUDA_CHECK(cudaMalloc(&d_mu, mu_bytes));
    CUDA_CHECK(cudaMemcpy(d_L, cholesky.L_flat.data(), L_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mu, h_mu.data(), mu_bytes,
                          cudaMemcpyHostToDevice));

    // Allocate output scenario matrix.
    ScenarioMatrix scenarios(n_scenarios, n_assets);

    // Create temporary cuRAND states if caller didn't provide them.
    CurandStatesGuard temp_states;
    CurandStates* active_states = states;
    if (!active_states) {
        temp_states = create_curand_states(n_scenarios, config.seed);
        active_states = temp_states.get();
    }

    // Launch kernel.
    int threads = config.threads_per_block;
    int blocks = (n_scenarios + threads - 1) / threads;
    k_monte_carlo_simulate<<<blocks, threads>>>(
        scenarios.device_ptr(), d_L, d_mu, active_states->d_states,
        n_scenarios, n_assets);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Cleanup device temporaries (L, mu).
    CUDA_CHECK(cudaFree(d_L));
    CUDA_CHECK(cudaFree(d_mu));

    log_vram_usage();
    return scenarios;
}

// ── CPU reference implementation ───────────────────────────────────

MatrixXd generate_scenarios_cpu(const VectorXd& mu,
                                const CholeskyResult& cholesky,
                                const MonteCarloConfig& config) {
    const Index n_assets = cholesky.n;
    const Index n_scenarios = config.n_scenarios;

    if (mu.size() != n_assets) {
        throw std::runtime_error(
            "generate_scenarios_cpu: mu size (" +
            std::to_string(mu.size()) + ") != n_assets (" +
            std::to_string(n_assets) + ")");
    }

    spdlog::info("Monte Carlo CPU: {} scenarios x {} assets (seed={})",
                 n_scenarios, n_assets, config.seed);

    std::mt19937 rng(static_cast<unsigned>(config.seed));
    std::normal_distribution<double> normal(0.0, 1.0);

    // r_i = mu + L_cpu * z, all in double.
    MatrixXd result(n_scenarios, n_assets);
    VectorXd z(n_assets);

    for (Index i = 0; i < n_scenarios; ++i) {
        for (Index j = 0; j < n_assets; ++j) {
            z(j) = normal(rng);
        }
        result.row(i) = (mu + cholesky.L_cpu * z).transpose();
    }

    return result;
}

}  // namespace cpo
