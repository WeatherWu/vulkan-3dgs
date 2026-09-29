#include "render/renderer.hpp"
#include "context/context.hpp"

namespace vulkan3DGS {

vk::Device Renderer::getDevice() const {
    return Context::Instance().Device();
}

vk::SurfaceKHR Renderer::getSurface() const {
    return Context::Instance().getSurface();
}

} // namespace vulkan3DGS
