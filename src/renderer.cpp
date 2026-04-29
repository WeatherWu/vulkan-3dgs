#include "renderer.hpp"
#include "context/context.hpp"
#include "utils/logger.hpp"

namespace vk_gs {

vk::Device Renderer::getDevice() const {
    return Context::Instance().Device();
}

vk::SurfaceKHR Renderer::getSurface() const {
    return Context::Instance().getSurface();
}

}