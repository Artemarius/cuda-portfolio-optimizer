add_executable(bench_monte_carlo bench_monte_carlo.cpp)
target_link_libraries(bench_monte_carlo PRIVATE
    cuda_portfolio_lib
    benchmark::benchmark
)

add_executable(bench_cvar bench_cvar.cpp)
target_link_libraries(bench_cvar PRIVATE
    cuda_portfolio_lib
    benchmark::benchmark
)

add_executable(bench_admm bench_admm.cpp)
target_link_libraries(bench_admm PRIVATE
    cuda_portfolio_lib
    benchmark::benchmark
)

add_executable(bench_factor_model bench_factor_model.cpp)
target_link_libraries(bench_factor_model PRIVATE
    cuda_portfolio_lib
    benchmark::benchmark
)
