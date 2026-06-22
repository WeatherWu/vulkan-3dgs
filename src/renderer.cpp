#include "renderer.hpp"
#include "context/context.hpp"

namespace vk_gs {

vk::Device Renderer::getDevice() const {
    return Context::Instance().Device();
}

vk::SurfaceKHR Renderer::getSurface() const {
    return Context::Instance().getSurface();
}

} // namespace vk_gs