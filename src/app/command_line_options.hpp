#pragma once

#include "training/app/training_smoke_options.hpp"
#include <charconv>
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
    std::optional<TrainingSmokeOptions> trainingSmoke;
};

// Arguments exclude argv[0]. Parsing does not initialize GLFW or Vulkan.
inline ApplicationCommandLineOptions
parseApplicationCommandLine(std::span<const std::string_view> arguments) {
    ApplicationCommandLineOptions options;
    bool validationSpecified = false;
    bool smokeStepsSpecified = false, smokeReportSpecified = false;
    TrainingSmokeOptions smoke;
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
        } else if (argument == "--training-smoke-steps" ||
                   argument.starts_with("--training-smoke-steps=")) {
            if (smokeStepsSpecified)
                throw std::invalid_argument("Duplicate --training-smoke-steps");
            smokeStepsSpecified = true;
            const auto value =
                argument == "--training-smoke-steps"
                    ? (++index < arguments.size() ? arguments[index] : std::string_view{})
                    : argument.substr(23);
            if (value.empty())
                throw std::invalid_argument("--training-smoke-steps requires a value");
            const auto parsed =
                std::from_chars(value.data(), value.data() + value.size(), smoke.stopIteration);
            if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size())
                throw std::invalid_argument(
                    "--training-smoke-steps requires an integer in 1..29999");
        } else if (argument == "--training-smoke-report" ||
                   argument.starts_with("--training-smoke-report=")) {
            if (smokeReportSpecified)
                throw std::invalid_argument("Duplicate --training-smoke-report");
            smokeReportSpecified = true;
            const auto value =
                argument == "--training-smoke-report"
                    ? (++index < arguments.size() ? arguments[index] : std::string_view{})
                    : argument.substr(24);
            if (value.empty() || value.starts_with("--"))
                throw std::invalid_argument("--training-smoke-report requires a path");
            smoke.report = std::string(value);
        } else {
            throw std::invalid_argument("Unknown command-line option: " + std::string(argument));
        }
    }
    if (smokeStepsSpecified || smokeReportSpecified) {
        if (!smokeStepsSpecified || !smokeReportSpecified || !options.validationLayersEnabled)
            throw std::invalid_argument(
                "Training smoke requires both smoke options and --validation on");
        smoke.validate();
        options.trainingSmoke = std::move(smoke);
    }
    return options;
}

} // namespace vulkan3DGS
