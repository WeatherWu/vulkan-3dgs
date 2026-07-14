#include "application.hpp"

#include <exception>
#include <iostream>

int main() {
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
