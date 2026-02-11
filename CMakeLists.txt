cmake_minimum_required(VERSION 3.20)
project(cuda_portfolio_optimizer LANGUAGES CXX CUDA)

# ── C++ and CUDA standards ──────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)

# ── CUDA Toolkit (cuRAND, CUB, etc.) ───────────────────────────────
find_package(CUDAToolkit REQUIRED)

# ── FetchContent dependencies ───────────────────────────────────────
include(FetchContent)

# Eigen3 — header-only CPU linear algebra (skip its CMake build system
# entirely to avoid 900+ test registrations polluting ctest)
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        3.4.0
    GIT_SHALLOW    TRUE
)
FetchContent_GetProperties(eigen)
if(NOT eigen_POPULATED)
    FetchContent_Populate(eigen)
endif()
add_library(eigen_headers INTERFACE)
add_library(Eigen3::Eigen ALIAS eigen_headers)
target_include_directories(eigen_headers INTERFACE ${eigen_SOURCE_DIR})

# nlohmann/json — config and output serialization
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
    GIT_SHALLOW    TRUE
)

# spdlog — logging
FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG        v1.14.1
    GIT_SHALLOW    TRUE
)

# Google Test — unit tests
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
    GIT_SHALLOW    TRUE
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Google Benchmark — performance comparison
FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        v1.8.4
    GIT_SHALLOW    TRUE
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(nlohmann_json spdlog googletest googlebenchmark)

# ── Subdirectories ──────────────────────────────────────────────────
add_subdirectory(src)
add_subdirectory(apps)

enable_testing()
add_subdirectory(tests)
add_subdirectory(benchmarks)
