#include "data/universe.h"

#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace cpo {

Universe load_universe(const std::string& json_path) {
    std::ifstream file(json_path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open universe file: " + json_path);
    }

    nlohmann::json j;
    try {
        j = nlohmann::json::parse(file);
    } catch (const nlohmann::json::parse_error& e) {
        throw std::runtime_error("Failed to parse universe JSON: " +
                                 std::string(e.what()));
    }

    Universe universe;
    universe.tickers = j.at("tickers").get<std::vector<std::string>>();

    if (j.contains("start_date")) {
        universe.start_date = j["start_date"].get<std::string>();
    }
    if (j.contains("end_date")) {
        universe.end_date = j["end_date"].get<std::string>();
    }

    return universe;
}

}  // namespace cpo
