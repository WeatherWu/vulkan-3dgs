#pragma once

#include <vulkan/vulkan.hpp>

namespace vk_gs {

class RenderPass {
public:
    static constexpr vk::Format DepthFormat = vk::Format::eD32Sfloat;

    RenderPass();
    ~RenderPass();
    
    void initialize(vk::Format swapchain_format);
    void cleanup();
    
    vk::RenderPass getRenderPass() const { return renderPass_; }
    
private:
    vk::RenderPass renderPass_ = nullptr;
    
};

} // namespace vk_gs
