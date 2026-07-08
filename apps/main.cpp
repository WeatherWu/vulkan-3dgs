#include "application.hpp"

int main(){
    vulkan3DGS::Application app("vulkan-3dgs", 1280, 720, vulkan3DGS::RenderMode::GaussianGraphics);
    app.run();

    return 0;
}
