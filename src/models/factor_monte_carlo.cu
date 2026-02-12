#include "models/factor_monte_carlo.h"

#include <random>
#include <vector>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <Eigen/Cholesky>
#include <spdlog/spdlog.h>

#include "simulation/curand_states.cuh"
#include "utils/cuda_utils.h"

namespace cpo {

// ── GPU kernel ───────────────────────────────────────────────────────

/// Factor-based Monte Carlo scenario generation kernel.
///
/// One thread per scenario. Uses shared memory for B, mu, sqrt_D.
/// Factor Cholesky multiply (k x k) is done in registers since k is small.
///
/// Generates: r_i = mu_i + sum_j B[i,j]*f[j] + sqrt(D_i) * z_e_i
/// where f = L_f * z_f, z_f ~ N(0,I), z_e ~ N(0,1).
///
/// @param d_scenarios Output: column-major (n_scenarios x n_assets).
/// @param d_B         Factor loadings: row-major (n_assets x n_factors).
/// @param d_L_f       Factor Cholesky: row-major (n_factors x n_factors).
/// @param d_sqrt_D    sqrt(idiosyncratic variance): (n_assets).
/// @param d_mu        Mean returns: (n_assets).
/// @param states      cuRAND states (one per scenario).
/// @param n_scenarios Number of scenarios.
/// @param n_assets    Number of assets (N).
/// @param n_factors   Number of factors (k).
__global__ void k_factor_monte_carlo(
    float* __restrict__ d_scenarios,
    const float* __restrict__ d_B,
    const float* __restrict__ d_L_f,
    const float* __restrict__ d_sqrt_D,
    const float* __restrict__ d_mu,
    curandState_t* __restrict__ states,
    int n_scenarios, int n_assets, int n_factors) {

    // Shared memory layout:
    //   [0 .. N*k-1]:         B (row-major, N x k)
    //   [N*k .. N*k+N-1]:     mu (N)
    //   [N*k+N .. N*k+2*N-1]: sqrt_D (N)
    extern __shared__ float smem[];
    float* s_B      = smem;
    float* s_mu     = smem + n_assets * n_factors;
    float* s_sqrt_D = s_mu + n_assets;

    // Cooperative load of B, mu, sqrt_D into shared memory.
    int tid_in_block = threadIdx.x;
    int block_size = blockDim.x;

    // Load B (N*k elements).
    int n_B = n_assets * n_factors;
    for (int i = tid_in_block; i < n_B; i += block_size) {
        s_B[i] = d_B[i];
    }
    // Load mu and sqrt_D (N elements each).
    for (int i = tid_in_block; i < n_assets; i += block_size) {
        s_mu[i] = d_mu[i];
        s_sqrt_D[i] = d_sqrt_D[i];
    }
    __syncthreads();

    int sid = blockIdx.x * blockDim.x + threadIdx.x;
    if (sid >= n_scenarios) return;

    // Load this thread's RNG state into registers.
    curandState_t local_state = states[sid];

    // Step 1: Generate k standard normals and compute correlated factors.
    // Since k is small (typically 5-20), use stack-allocated arrays.
    // CUDA places small arrays in registers or local memory.
    float z_f[64];  // Max supported factors.
    float f[64];

    // Generate z_f ~ N(0, I_k).
    for (int j = 0; j < n_factors; ++j) {
        z_f[j] = curand_normal(&local_state);
    }

    // f = L_f * z_f (lower-triangular matrix-vector multiply).
    // L_f is row-major: L_f[j,m] at j * n_factors + m.
    for (int j = 0; j < n_factors; ++j) {
        float sum = 0.0f;
        for (int m = 0; m <= j; ++m) {
            sum += d_L_f[j * n_factors + m] * z_f[m];
        }
        f[j] = sum;
    }

    // Step 2: For each asset, compute r_i = mu_i + B_i' * f + sqrt(D_i) * z_e.
    for (int i = 0; i < n_assets; ++i) {
        // B_i' * f = sum_j B[i,j] * f[j].
        // B is row-major in shared memory: B[i,j] at i * n_factors + j.
        float Bf_i = 0.0f;
        for (int j = 0; j < n_factors; ++j) {
            Bf_i += s_B[i * n_factors + j] * f[j];
        }

        // Idiosyncratic noise.
        float z_e = curand_normal(&local_state);

        // Write column-major: element (sid, i) at i * n_scenarios + sid.
        d_scenarios[i * n_scenarios + sid] =
            s_mu[i] + Bf_i + s_sqrt_D[i] * z_e;
    }

    // Write back updated RNG state.
    states[sid] = local_state;
}

// ── Host orchestration ───────────────────────────────────────────────

ScenarioMatrix generate_scenarios_factor_gpu(
    const VectorXd& mu,
    const FactorModelResult& model,
    const MonteCarloConfig& config,
    CurandStates* states) {

    const Index n_assets = model.n_assets;
    const Index n_factors = model.n_factors;
    const Index n_scenarios = config.n_scenarios;

    if (static_cast<Index>(mu.size()) != n_assets) {
        throw std::runtime_error(
            "generate_scenarios_factor_gpu: mu size (" +
            std::to_string(mu.size()) + ") != n_assets (" +
            std::to_string(n_assets) + ")");
    }
    if (n_factors > 64) {
        throw std::runtime_error(
            "generate_scenarios_factor_gpu: n_factors (" +
            std::to_string(n_factors) +
            ") exceeds kernel max (64)");
    }

    spdlog::info("Factor Monte Carlo GPU: {} scenarios x {} assets, "
                 "k={} factors (seed={})",
                 n_scenarios, n_assets, n_factors, config.seed);

    // 1. Compute k x k Cholesky of factor covariance.
    Eigen::LLT<MatrixXd> llt(model.factor_covariance);
    if (llt.info() != Eigen::Success) {
        throw std::runtime_error(
            "generate_scenarios_factor_gpu: factor covariance "
            "Cholesky failed");
    }
    MatrixXd L_f_cpu = llt.matrixL();

    // 2. Convert all inputs to float vectors for GPU upload.
    //    B: row-major (N x k).
    std::vector<Scalar> h_B(n_assets * n_factors);
    for (Index i = 0; i < n_assets; ++i) {
        for (Index j = 0; j < n_factors; ++j) {
            h_B[i * n_factors + j] =
                static_cast<Scalar>(model.loadings(i, j));
        }
    }

    //    L_f: row-major (k x k).
    std::vector<Scalar> h_L_f(n_factors * n_factors, 0.0f);
    for (Index i = 0; i < n_factors; ++i) {
        for (Index j = 0; j <= i; ++j) {
            h_L_f[i * n_factors + j] =
                static_cast<Scalar>(L_f_cpu(i, j));
        }
    }

    //    sqrt_D: (N).
    std::vector<Scalar> h_sqrt_D(n_assets);
    for (Index i = 0; i < n_assets; ++i) {
        h_sqrt_D[i] = static_cast<Scalar>(
            std::sqrt(model.idiosyncratic_var(i)));
    }

    //    mu: (N).
    std::vector<Scalar> h_mu(n_assets);
    for (Index i = 0; i < n_assets; ++i) {
        h_mu[i] = static_cast<Scalar>(mu(i));
    }

    // 3. Upload to device.
    Scalar* d_B = nullptr;
    Scalar* d_L_f = nullptr;
    Scalar* d_sqrt_D = nullptr;
    Scalar* d_mu = nullptr;

    size_t B_bytes = h_B.size() * sizeof(Scalar);
    size_t Lf_bytes = h_L_f.size() * sizeof(Scalar);
    size_t sqrtD_bytes = h_sqrt_D.size() * sizeof(Scalar);
    size_t mu_bytes = h_mu.size() * sizeof(Scalar);

    CUDA_CHECK(cudaMalloc(&d_B, B_bytes));
    CUDA_CHECK(cudaMalloc(&d_L_f, Lf_bytes));
    CUDA_CHECK(cudaMalloc(&d_sqrt_D, sqrtD_bytes));
    CUDA_CHECK(cudaMalloc(&d_mu, mu_bytes));

    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), B_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_L_f, h_L_f.data(), Lf_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sqrt_D, h_sqrt_D.data(), sqrtD_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mu, h_mu.data(), mu_bytes,
                          cudaMemcpyHostToDevice));

    spdlog::debug("Factor MC device uploads: B={:.1f} KB, L_f={:.1f} KB, "
                  "sqrt_D={:.1f} KB, mu={:.1f} KB",
                  B_bytes / 1024.0, Lf_bytes / 1024.0,
                  sqrtD_bytes / 1024.0, mu_bytes / 1024.0);

    // 4. Allocate output scenario matrix.
    ScenarioMatrix scenarios(n_scenarios, n_assets);

    // 5. Create temporary cuRAND states if caller didn't provide them.
    CurandStatesGuard temp_states;
    CurandStates* active_states = states;
    if (!active_states) {
        temp_states = create_curand_states(n_scenarios, config.seed);
        active_states = temp_states.get();
    }

    // 6. Launch kernel with dynamic shared memory.
    int threads = config.threads_per_block;
    int blocks = (n_scenarios + threads - 1) / threads;
    size_t smem_bytes = static_cast<size_t>(n_assets * n_factors + 2 * n_assets)
                        * sizeof(Scalar);

    spdlog::debug("Factor MC kernel: {} blocks x {} threads, "
                  "{:.1f} KB shared memory",
                  blocks, threads, smem_bytes / 1024.0);

    k_factor_monte_carlo<<<blocks, threads, smem_bytes>>>(
        scenarios.device_ptr(), d_B, d_L_f, d_sqrt_D, d_mu,
        active_states->d_states,
        n_scenarios, n_assets, n_factors);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 7. Cleanup device temporaries.
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_L_f));
    CUDA_CHECK(cudaFree(d_sqrt_D));
    CUDA_CHECK(cudaFree(d_mu));

    log_vram_usage();
    return scenarios;
}

// ── CPU reference implementation ─────────────────────────────────────

MatrixXd generate_scenarios_factor_cpu(
    const VectorXd& mu,
    const FactorModelResult& model,
    const MonteCarloConfig& config) {

    const Index n_assets = model.n_assets;
    const Index n_factors = model.n_factors;
    const Index n_scenarios = config.n_scenarios;

    if (static_cast<Index>(mu.size()) != n_assets) {
        throw std::runtime_error(
            "generate_scenarios_factor_cpu: mu size (" +
            std::to_string(mu.size()) + ") != n_assets (" +
            std::to_string(n_assets) + ")");
    }

    spdlog::info("Factor Monte Carlo CPU: {} scenarios x {} assets, "
                 "k={} factors (seed={})",
                 n_scenarios, n_assets, n_factors, config.seed);

    // Cholesky of factor covariance in double precision.
    Eigen::LLT<MatrixXd> llt(model.factor_covariance);
    if (llt.info() != Eigen::Success) {
        throw std::runtime_error(
            "generate_scenarios_factor_cpu: factor covariance "
            "Cholesky failed");
    }
    MatrixXd L_f = llt.matrixL();

    // Precompute sqrt(D) in double.
    VectorXd sqrt_D(n_assets);
    for (Index i = 0; i < n_assets; ++i) {
        sqrt_D(i) = std::sqrt(model.idiosyncratic_var(i));
    }

    std::mt19937 rng(static_cast<unsigned>(config.seed));
    std::normal_distribution<double> normal(0.0, 1.0);

    MatrixXd result(n_scenarios, n_assets);
    VectorXd z_f(n_factors);

    for (Index s = 0; s < n_scenarios; ++s) {
        // Generate factor normals and compute correlated factors.
        for (Index j = 0; j < n_factors; ++j) {
            z_f(j) = normal(rng);
        }
        VectorXd f = L_f * z_f;

        // For each asset: r_i = mu_i + B_i' * f + sqrt(D_i) * z_e.
        for (Index i = 0; i < n_assets; ++i) {
            double Bf_i = model.loadings.row(i).dot(f);
            double z_e = normal(rng);
            result(s, i) = mu(i) + Bf_i + sqrt_D(i) * z_e;
        }
    }

    return result;
}

}  // namespace cpo
