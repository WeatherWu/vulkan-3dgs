#include "application.hpp"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

#ifndef _WIN32
#include <unistd.h>
#endif

namespace {

struct CommandLineOptions {
    std::optional<std::string> gpuSelector;
    bool showHelp = false;
};

void printUsage(const char* executable) {
    std::cout << "Usage: " << executable << " [--gpu <index|name|uuid>]\n"
              << "  --gpu VALUE  Select a training GPU by Vulkan index, name, or UUID.\n"
              << "  -h, --help   Show this help.\n";
}

CommandLineOptions parseCommandLine(int argc, char** argv) {
    CommandLineOptions options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "-h" || argument == "--help") {
            options.showHelp = true;
        } else if (argument == "--gpu") {
            if (++index >= argc || std::string(argv[index]).empty()) {
                throw std::invalid_argument("--gpu requires an index, device name, or UUID");
            }
            options.gpuSelector = argv[index];
        } else if (argument.rfind("--gpu=", 0) == 0) {
            const std::string value = argument.substr(std::strlen("--gpu="));
            if (value.empty()) {
                throw std::invalid_argument("--gpu requires an index, device name, or UUID");
            }
            options.gpuSelector = value;
        } else {
            throw std::invalid_argument("Unknown command-line option: " + argument);
        }
    }
    return options;
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
        const CommandLineOptions options = parseCommandLine(argc, argv);
        if (options.showHelp) {
            printUsage(argv[0]);
            return 0;
        }

        vulkan3DGS::Application app("vulkan-3dgs",
                                    1280,
                                    720,
                                    vulkan3DGS::RenderMode::GaussianGraphics,
                                    options.gpuSelector);
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
