#include "application.hpp"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>

#ifndef _WIN32
#include <unistd.h>

namespace {

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

} // namespace
#endif

int main(int argc, char** argv) {
    (void)argc;
#ifndef _WIN32
    restartWithoutVirtualGLPreload(argv);
#endif

    try {
        vulkan3DGS::Application app("vulkan-3dgs", 1280, 720, vulkan3DGS::RenderMode::GaussianGraphics);
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
