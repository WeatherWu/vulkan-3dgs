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
        check(defaults.validationLayersEnabled && !defaults.showHelp && !defaults.gpuSelector,
              "Default validation must remain on in every build configuration");
        for (const auto argument : {"on", "off"}) {
            const auto separated = parse({"--validation", argument});
            const auto joined = std::string("--validation=") + argument;
            check(separated.validationLayersEnabled == (std::string_view(argument) == "on") &&
                      parse({joined}).validationLayersEnabled == separated.validationLayersEnabled,
                  "Validation argument forms disagree");
        }
        const auto combined = parse({"--gpu", "RTX 3090", "--validation=off"});
        check(combined.gpuSelector == "RTX 3090" && !combined.validationLayersEnabled,
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
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "Application command-line contracts passed\n";
}
