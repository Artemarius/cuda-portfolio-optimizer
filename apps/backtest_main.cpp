#include <spdlog/spdlog.h>

#include "utils/cuda_utils.h"

int main(int argc, char* argv[]) {
    spdlog::info("cuda-portfolio-optimizer: backtest");
    spdlog::info("==================================");

    cpo::device_query();

    return 0;
}
