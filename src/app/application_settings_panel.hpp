#pragma once

#include "app/application_settings.hpp"
#include <string>

namespace vulkan3DGS {

class ApplicationSettingsPanel {
public:
    static bool draw(ApplicationSettings& settings, ApplicationValidationSelection selection,
                     bool actualValidation, const std::string& saveError);
};

} // namespace vulkan3DGS
