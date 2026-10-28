#pragma once

#include <filesystem>
#include <optional>
#include <stdexcept>

namespace vulkan3DGS {

struct ApplicationSettings {
    bool validationLayersEnabled = true;
};

enum class ApplicationValidationSource { SavedSetting, CommandLine, Smoke };

struct ApplicationValidationSelection {
    bool requested = true;
    ApplicationValidationSource source = ApplicationValidationSource::SavedSetting;
};

inline ApplicationValidationSelection
resolveApplicationValidation(const ApplicationSettings& settings, std::optional<bool> override,
                             bool smoke) {
    if (smoke && override == false)
        throw std::invalid_argument("Training smoke requires validation on");
    if (override) return {*override, ApplicationValidationSource::CommandLine};
    if (smoke) return {true, ApplicationValidationSource::Smoke};
    return {settings.validationLayersEnabled, ApplicationValidationSource::SavedSetting};
}

class ApplicationSettingsStore {
public:
    static void load(const std::filesystem::path& path, ApplicationSettings& settings);
    static void save(const std::filesystem::path& path, const ApplicationSettings& settings);
};

} // namespace vulkan3DGS
