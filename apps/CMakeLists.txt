add_executable(optimize optimize_main.cpp)
target_link_libraries(optimize PRIVATE cuda_portfolio_lib)

add_executable(backtest backtest_main.cpp)
target_link_libraries(backtest PRIVATE cuda_portfolio_lib)
