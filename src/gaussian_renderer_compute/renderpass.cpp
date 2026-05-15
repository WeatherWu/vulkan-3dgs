#include <stdexcept>

#include "renderpass.hpp"
#include "utils/logger.hpp"
#include "context/context.hpp"

#include <array>

namespace vk_gs {

RenderPass::RenderPass() = default;

RenderPass::~RenderPass() {
    cleanup();
}

void RenderPass::initialize(vk::Format swapchain_format) {
    auto device = Context::Instance().Device();
    
    LOG_INFO("Creating render pass");
    
    // 颜色附件描述
    vk::AttachmentDescription color_attachment{};
    color_attachment.setFormat(swapchain_format)
                    .setSamples(vk::SampleCountFlagBits::e1)
                    .setLoadOp(vk::AttachmentLoadOp::eClear)
                    .setStoreOp(vk::AttachmentStoreOp::eStore)
                    .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
                    .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
                    .setInitialLayout(vk::ImageLayout::eUndefined)
                    .setFinalLayout(vk::ImageLayout::ePresentSrcKHR);
    
    // 深度附件描述。vkgs keeps a depth attachment and enables depth test for
    // splats while leaving depth writes disabled.
    vk::AttachmentDescription depth_attachment{};
    depth_attachment.setFormat(RenderPass::DepthFormat)
                    .setSamples(vk::SampleCountFlagBits::e1)
                    .setLoadOp(vk::AttachmentLoadOp::eClear)
                    .setStoreOp(vk::AttachmentStoreOp::eDontCare)
                    .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
                    .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
                    .setInitialLayout(vk::ImageLayout::eUndefined)
                    .setFinalLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal);

    std::array<vk::AttachmentDescription, 2> attachments = {
        color_attachment,
        depth_attachment
    };

    // 颜色附件引用
    vk::AttachmentReference color_attachment_ref{};
    color_attachment_ref.setAttachment(0)
                        .setLayout(vk::ImageLayout::eColorAttachmentOptimal);

    vk::AttachmentReference depth_attachment_ref{};
    depth_attachment_ref.setAttachment(1)
                        .setLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal);
    
    // 子pass描述
    vk::SubpassDescription subpass{};
    subpass.setPipelineBindPoint(vk::PipelineBindPoint::eGraphics)
           .setColorAttachmentCount(1)
           .setPColorAttachments(&color_attachment_ref)
           .setPDepthStencilAttachment(&depth_attachment_ref);
    
    // 子pass依赖（确保渲染前图像处于正确的布局）
    vk::SubpassDependency dependency{};
    dependency.setSrcSubpass(vk::SubpassExternal)
              .setDstSubpass(0)
              .setSrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput |
                               vk::PipelineStageFlagBits::eEarlyFragmentTests)
              .setDstStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput |
                               vk::PipelineStageFlagBits::eEarlyFragmentTests)
              .setSrcAccessMask(vk::AccessFlagBits::eNone)
              .setDstAccessMask(vk::AccessFlagBits::eColorAttachmentWrite |
                                vk::AccessFlagBits::eDepthStencilAttachmentWrite);
    
    // 创建渲染通道
    vk::RenderPassCreateInfo render_pass_info{};
    render_pass_info.setAttachmentCount(static_cast<uint32_t>(attachments.size()))
                    .setPAttachments(attachments.data())
                    .setSubpassCount(1)
                    .setPSubpasses(&subpass)
                    .setDependencyCount(1)
                    .setPDependencies(&dependency);
    
    renderPass_ = device.createRenderPass(render_pass_info);
    
    if (!renderPass_) {
        LOG_ERROR("Failed to create render pass");
        throw std::runtime_error("Failed to create render pass");
    }
    
    LOG_INFO("Render pass created successfully");
}

void RenderPass::cleanup() {
    if (renderPass_) {
        Context::Instance().Device().destroyRenderPass(renderPass_);
        renderPass_ = nullptr;
        LOG_INFO("Render pass cleaned up");
    }
}

} // namespace vk_gs
