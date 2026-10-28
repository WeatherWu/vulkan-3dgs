#include "app/command_line_options.hpp"

#include <initializer_list>
#include <iostream>
#include <stdexcept>

using namespace vulkan3DGS;
namespace {
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
ApplicationCommandLineOptions parse(std::initializer_list<std::string_view> arguments) {
    return parseApplicationCommandLine({arguments.begin(), arguments.size()});
}
void reject(std::initializer_list<std::string_view> arguments) {
    try {
        (void)parse(arguments);
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error("Invalid command line accepted");
}
} // namespace
int main() {
    try {
        const auto defaults = parse({});
        check(!defaults.validationLayersOverride.has_value() && !defaults.showHelp &&
                  !defaults.gpuSelector,
              "Unspecified validation must defer to application settings");
        for (const auto argument : {"on", "off"}) {
            const auto separated = parse({"--validation", argument});
            const auto joined = std::string("--validation=") + argument;
            check(separated.validationLayersOverride == (std::string_view(argument) == "on") &&
                      parse({joined}).validationLayersOverride ==
                          separated.validationLayersOverride,
                  "Validation argument forms disagree");
        }
        const auto combined = parse({"--gpu", "RTX 3090", "--validation=off"});
        check(combined.gpuSelector == "RTX 3090" && combined.validationLayersOverride == false,
              "GPU selector was lost");
        check(parse({"--validation", "on", "--gpu=uuid:abc"}).gpuSelector == "uuid:abc",
              "GPU equals form was lost");
        check(parse({"--gpu=0", "--gpu=1"}).gpuSelector == "1", "Existing GPU override semantics");
        check(parse({"-h"}).showHelp && parse({"--validation=off", "--help"}).showHelp,
              "Help parsing changed");
        reject({"--validation"});
        reject({"--validation="});
        reject({"--validation", ""});
        reject({"--validation", "true"});
        reject({"--validation=0"});
        reject({"--validation=OFF"});
        reject({"--validation", "--help"});
        reject({"--validation=off", "--validation=on"});
        reject({"--validation=on", "--validation", "on"});
        reject({"--disable-validation"});
        reject({"--gpu"});
        reject({"--gpu="});
        const auto smoke =
            parse({"--training-smoke-steps", "3200", "--training-smoke-report", "smoke A.json"});
        check(smoke.trainingSmoke && smoke.trainingSmoke->stopIteration == 3200 &&
                  smoke.trainingSmoke->report == "smoke A.json" &&
                  !smoke.validationLayersOverride.has_value(),
              "Smoke CLI boundary/report parsing");
        check(parse({"--training-smoke-steps=3200", "--training-smoke-report=smoke.json"})
                      .trainingSmoke->stopIteration == 3200,
              "Smoke equals-form parsing");
        for (auto invalid : {"0", "-1", "30000", "4294967296", "1x", ""})
            reject({"--training-smoke-steps", invalid, "--training-smoke-report", "smoke.json"});
        reject({"--training-smoke-steps", "3200"});
        reject({"--training-smoke-report", "smoke.json"});
        reject({"--training-smoke-steps=3200", "--training-smoke-report="});
        reject(
            {"--training-smoke-steps=3200", "--training-smoke-report=x.json", "--validation=off"});
        reject({"--training-smoke-steps=2", "--training-smoke-steps=3",
                "--training-smoke-report=x.json"});
        reject({"--training-smoke-steps=2", "--training-smoke-report=x.json",
                "--training-smoke-report=y.json"});
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "Application command-line contracts passed\n";
}
