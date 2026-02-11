#include <spdlog/spdlog.h>

#include "utils/cuda_utils.h"

int main(int argc, char* argv[]) {
    spdlog::info("cuda-portfolio-optimizer");
    spdlog::info("========================");

    cpo::device_query();
    cpo::log_vram_usage();

    return 0;
}
