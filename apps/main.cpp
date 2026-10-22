#include "app/application.hpp"
#include "app/command_line_options.hpp"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifndef _WIN32
#include <unistd.h>
#endif

namespace {

void printUsage(const char* executable) {
    std::cout << "Usage: " << executable << " [--gpu <index|name|uuid>] [--validation <on|off>]\n"
              << "  --gpu VALUE  Select a training GPU by Vulkan index, name, or UUID.\n"
              << "  --validation on|off  Request Vulkan validation layers (default: on).\n"
              << "                       Use off for timing; training numeric checks are unchanged.\n"
              << "  -h, --help   Show this help.\n";
}

#ifndef _WIN32
bool hasVirtualGLPreload(const char* preload) {
    return preload &&
           (std::strstr(preload, "libdlfaker") ||
            std::strstr(preload, "libvglfaker"));
}

void restartWithoutVirtualGLPreload(char** argv) {
    const char* preload = std::getenv("LD_PRELOAD");
    const char* cleaned = std::getenv("VULKAN_3DGS_LD_PRELOAD_CLEANED");
    if (!hasVirtualGLPreload(preload) || (cleaned && std::strcmp(cleaned, "1") == 0)) {
        return;
    }

    std::cerr << "Detected VirtualGL LD_PRELOAD; restarting without it for Vulkan compatibility." << std::endl;
    unsetenv("LD_PRELOAD");
    setenv("VULKAN_3DGS_LD_PRELOAD_CLEANED", "1", 1);
    execvp(argv[0], argv);

    std::cerr << "Fatal error: failed to restart without VirtualGL LD_PRELOAD" << std::endl;
    std::exit(1);
}
#endif

} // namespace

int main(int argc, char** argv) {
#ifndef _WIN32
    restartWithoutVirtualGLPreload(argv);
#endif

    try {
        std::vector<std::string_view> arguments;
        for (int index = 1; index < argc; ++index) arguments.emplace_back(argv[index]);
        const auto options = vulkan3DGS::parseApplicationCommandLine(arguments);
        if (options.showHelp) {
            printUsage(argv[0]);
            return 0;
        }

        vulkan3DGS::Application app("vulkan-3dgs",
                                    1280,
                                    720,
                                    vulkan3DGS::RenderMode::GaussianGraphics,
                                    options.gpuSelector,
                                    options.validationLayersEnabled);
        app.run();
    } catch (const std::exception& error) {
        std::cerr << "Fatal error: " << error.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Fatal error: unknown exception" << std::endl;
        return 1;
    }

    return 0;
}
