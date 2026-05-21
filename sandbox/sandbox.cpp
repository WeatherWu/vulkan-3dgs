#include "application.hpp"

int main(){
    vk_gs::Application app("Vulkan Gaussian Splatting", 1280, 720, vk_gs::RenderMode::GaussianGraphics);
    app.run();

    return 0;
}
