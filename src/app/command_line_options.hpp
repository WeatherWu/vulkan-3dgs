#pragma once

#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace vulkan3DGS {

struct ApplicationCommandLineOptions {
    std::optional<std::string> gpuSelector;
    bool validationLayersEnabled = true;
    bool showHelp = false;
};

// Arguments exclude argv[0]. Parsing does not initialize GLFW or Vulkan.
inline ApplicationCommandLineOptions
parseApplicationCommandLine(std::span<const std::string_view> arguments) {
    ApplicationCommandLineOptions options;
    bool validationSpecified = false;
    for (size_t index = 0; index < arguments.size(); ++index) {
        const auto argument = arguments[index];
        if (argument == "-h" || argument == "--help") {
            options.showHelp = true;
        } else if (argument == "--gpu" || argument.starts_with("--gpu=")) {
            const auto value =
                argument == "--gpu"
                    ? (++index < arguments.size() ? arguments[index] : std::string_view{})
                    : argument.substr(6);
            if (value.empty())
                throw std::invalid_argument("--gpu requires an index, device name, or UUID");
            options.gpuSelector = value;
        } else if (argument == "--validation" || argument.starts_with("--validation=")) {
            if (validationSpecified)
                throw std::invalid_argument("--validation may only be specified once");
            validationSpecified = true;
            const auto value =
                argument == "--validation"
                    ? (++index < arguments.size() ? arguments[index] : std::string_view{})
                    : argument.substr(13);
            if (value != "on" && value != "off")
                throw std::invalid_argument("--validation requires 'on' or 'off'");
            options.validationLayersEnabled = value == "on";
        } else {
            throw std::invalid_argument("Unknown command-line option: " + std::string(argument));
        }
    }
    return options;
}

} // namespace vulkan3DGS
