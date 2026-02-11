# Benchmark targets will be added in Phase 3+ (Monte Carlo, CVaR, etc.)
# Each benchmark follows the pattern:
#   add_executable(bench_<module> bench_<module>.cpp)
#   target_link_libraries(bench_<module> PRIVATE cuda_portfolio_lib benchmark::benchmark)
