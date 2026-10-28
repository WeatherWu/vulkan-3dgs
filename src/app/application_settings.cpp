#include "app/application_settings.hpp"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace vulkan3DGS {
namespace {
constexpr const char* validationKey = "vulkan_validation";
}

void ApplicationSettingsStore::load(const std::filesystem::path& path,
                                    ApplicationSettings& settings) {
    std::ifstream input(path);
    if (!input.is_open()) {
        if (std::filesystem::exists(path))
            throw std::runtime_error("Could not read application settings: " + path.string());
        return;
    }
    std::string line;
    while (std::getline(input, line)) {
        std::istringstream row(line);
        std::string key;
        row >> key;
        if (key == validationKey) {
            int value = 1;
            settings.validationLayersEnabled = !(row >> value) || value != 0;
        }
    }
    if (input.bad())
        throw std::runtime_error("Could not read application settings: " + path.string());
}

void ApplicationSettingsStore::save(const std::filesystem::path& path,
                                    const ApplicationSettings& settings) {
    std::vector<std::string> preserved;
    std::ifstream input(path);
    if (!input.is_open() && std::filesystem::exists(path))
        throw std::runtime_error("Could not read shared settings: " + path.string());
    std::string line;
    while (std::getline(input, line)) {
        std::istringstream row(line);
        std::string key;
        row >> key;
        if (key != validationKey) preserved.push_back(line);
    }
    if (input.bad()) throw std::runtime_error("Could not read shared settings: " + path.string());
    input.close();
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::trunc);
    if (!output.is_open())
        throw std::runtime_error("Could not save application settings: " + path.string());
    for (const auto& kept : preserved)
        output << kept << '\n';
    output << validationKey << ' ' << (settings.validationLayersEnabled ? 1 : 0) << '\n';
    output.flush();
    if (!output) throw std::runtime_error("Could not save application settings: " + path.string());
}

} // namespace vulkan3DGS
