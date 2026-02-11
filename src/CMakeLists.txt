add_library(cuda_portfolio_lib STATIC
    utils/cuda_utils.cu
)

target_include_directories(cuda_portfolio_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(cuda_portfolio_lib PUBLIC
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    CUDA::cudart
    CUDA::curand
)

# Note: CUDA_SEPARABLE_COMPILATION will be enabled in later phases
# when multiple .cu files need cross-TU device calls.
