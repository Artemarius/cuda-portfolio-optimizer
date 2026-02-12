add_library(cuda_portfolio_lib STATIC
    utils/cuda_utils.cu
    data/csv_loader.cpp
    data/returns.cpp
    data/universe.cpp
    simulation/cholesky_utils.cpp
    simulation/scenario_matrix.cu
    simulation/monte_carlo.cu
    risk/device_vector.cu
    risk/portfolio_loss.cu
    risk/cvar.cu
    optimizer/projections.cpp
    optimizer/objective.cpp
    optimizer/admm_solver.cpp
    optimizer/admm_kernels.cu
    optimizer/efficient_frontier.cpp
    optimizer/optimize_config.cpp
    constraints/constraint_set.cpp
    backtest/transaction_costs.cpp
    backtest/strategy.cpp
    backtest/backtest_config.cpp
    backtest/backtest_engine.cpp
    reporting/report_writer.cpp
    models/factor_model.cpp
    models/factor_monte_carlo.cu
    models/tiled_scenario_generator.cu
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
