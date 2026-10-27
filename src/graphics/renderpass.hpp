#pragma once

#include <vulkan/vulkan.hpp>

namespace vulkan3DGS {

class RenderPass {
public:
    static constexpr vk::Format DepthFormat = vk::Format::eD32Sfloat;

    RenderPass();
    ~RenderPass();

    void initialize(vk::Format swapchain_format);
    void initialize(vk::Device device, vk::Format colorFormat,
                    vk::ImageLayout finalColorLayout = vk::ImageLayout::ePresentSrcKHR);
    static vk::SubpassDependency attachmentReuseDependency() {
        vk::SubpassDependency dependency;
        const auto stages = vk::PipelineStageFlagBits::eColorAttachmentOutput |
                            vk::PipelineStageFlagBits::eEarlyFragmentTests |
                            vk::PipelineStageFlagBits::eLateFragmentTests;
        dependency.setSrcSubpass(vk::SubpassExternal)
            .setDstSubpass(0)
            .setSrcStageMask(stages)
            .setDstStageMask(stages)
            .setSrcAccessMask(vk::AccessFlagBits::eColorAttachmentWrite |
                              vk::AccessFlagBits::eDepthStencilAttachmentWrite)
            .setDstAccessMask(vk::AccessFlagBits::eColorAttachmentRead |
                              vk::AccessFlagBits::eColorAttachmentWrite |
                              vk::AccessFlagBits::eDepthStencilAttachmentRead |
                              vk::AccessFlagBits::eDepthStencilAttachmentWrite);
        return dependency;
    }
    void cleanup();

    vk::RenderPass getRenderPass() const {
        return renderPass_;
    }

private:
    vk::RenderPass renderPass_ = nullptr;
    vk::Device device_ = nullptr;
};

} // namespace vulkan3DGS
