include(GoogleTest)

add_executable(test_smoke test_smoke.cpp)
target_link_libraries(test_smoke PRIVATE
    cuda_portfolio_lib
    GTest::gtest_main
)
gtest_discover_tests(test_smoke)

add_executable(test_data test_data.cpp)
target_link_libraries(test_data PRIVATE
    cuda_portfolio_lib
    GTest::gtest_main
)
target_compile_definitions(test_data PRIVATE
    TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
gtest_discover_tests(test_data)

add_executable(test_monte_carlo test_monte_carlo.cpp)
target_link_libraries(test_monte_carlo PRIVATE
    cuda_portfolio_lib
    GTest::gtest_main
)
gtest_discover_tests(test_monte_carlo)

add_executable(test_risk test_risk.cpp)
target_link_libraries(test_risk PRIVATE
    cuda_portfolio_lib
    GTest::gtest_main
)
gtest_discover_tests(test_risk)
