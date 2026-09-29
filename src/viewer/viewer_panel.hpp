#pragma once

#include "viewer/viewer_controller.hpp"

namespace vulkan3DGS {
class ViewerPanel {
public:
    static void draw(ViewerController& controller, float frameRate, int framebufferWidth,
                     int framebufferHeight);
};
} // namespace vulkan3DGS
