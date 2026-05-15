#include <algorithm>
#include <limits>
#include <array>

#include "swapchain.hpp"
#include "utils/logger.hpp"
#include "context/context.hpp"

namespace vk_gs {

Swapchain::Swapchain(vk::SurfaceKHR surface):
      surface_(surface)
    , imageCount_(0)
    , imageFormat_()
    , extent_() {
    LOG_INFO("Creating swapchain");
}

Swapchain::~Swapchain() {
    cleanup();
}

void Swapchain::createSwapchain(uint32_t width, uint32_t height) {
    LOG_INFO("Creating swapchain with size: {}x{}", width, height);
    
    auto phyDevice = Context::Instance().PhysicalDevice();
    auto device = Context::Instance().Device();
    auto surface = surface_;
    // 获取表面能力
    vk::SurfaceCapabilitiesKHR capabilities = phyDevice.getSurfaceCapabilitiesKHR(surface_);
    
    // 选择表面格式
    std::vector<vk::SurfaceFormatKHR> available_formats = phyDevice.getSurfaceFormatsKHR(surface_);
    if (available_formats.empty()) {
        LOG_ERROR("No available surface formats");
        throw std::runtime_error("No available surface formats");
    }
    vk::SurfaceFormatKHR surface_format = chooseSwapSurfaceFormat(available_formats);
    imageFormat_ = surface_format.format;
    LOG_INFO("Selected swapchain format: {}, color space: {}",
             vk::to_string(surface_format.format),
             vk::to_string(surface_format.colorSpace));
    
    // 选择呈现模式
    std::vector<vk::PresentModeKHR> available_present_modes = phyDevice.getSurfacePresentModesKHR(surface_);
    if (available_present_modes.empty()) {
        LOG_ERROR("No available present modes");
        throw std::runtime_error("No available present modes");
    }
    vk::PresentModeKHR present_mode = chooseSwapPresentMode(available_present_modes);
    
    // 选择交换链范围
    extent_ = chooseSwapExtent(capabilities, width, height);
    
    // 确定图像数量（强制使用2个图像以匹配MAX_FRAMES_IN_FLIGHT）
    imageCount_ = 2;
    
    // 验证是否符合硬件要求
    if (imageCount_ < capabilities.minImageCount) {
        LOG_WARN("Requested {} images but hardware requires minimum {}, using {}", 
                 imageCount_, capabilities.minImageCount, capabilities.minImageCount);
        imageCount_ = capabilities.minImageCount;
    }
    if (capabilities.maxImageCount > 0 && imageCount_ > capabilities.maxImageCount) {
        imageCount_ = capabilities.maxImageCount;
    }
    
    LOG_INFO("Swapchain configured with {} images", imageCount_);
    
    // 创建交换链
    vk::SwapchainCreateInfoKHR createInfo{};
    createInfo.setSurface(surface_)
              .setMinImageCount(imageCount_)
              .setImageFormat(surface_format.format)
              .setImageColorSpace(surface_format.colorSpace)
              .setImageExtent(extent_)
              .setImageArrayLayers(1)
              .setImageUsage(vk::ImageUsageFlagBits::eColorAttachment);
    
    // 设置队列家族索引（如果图形队列和呈现队列不同）
    uint32_t queueFamilyIndices[] = {
        0, // 需要从Device获取实际的队列家族索引
        0
    };
    
    // 简化处理：假设使用共享模式
    createInfo.setImageSharingMode(vk::SharingMode::eExclusive);
    
    createInfo.setPreTransform(capabilities.currentTransform)
              .setCompositeAlpha(vk::CompositeAlphaFlagBitsKHR::eOpaque)
              .setPresentMode(present_mode)
              .setClipped(VK_TRUE)
              .setOldSwapchain(nullptr);
    
    swapchain_ = device.createSwapchainKHR(createInfo);
    if (!swapchain_) {
        LOG_ERROR("Failed to create swapchain");
        throw std::runtime_error("Failed to create swapchain");
    }
    
    LOG_INFO("Swapchain created successfully");
    
    // 获取交换链图像
    std::vector<vk::Image> swapchain_images = device.getSwapchainImagesKHR(swapchain_);
    
    // 创建图像视图和帧缓冲
    images_.resize(swapchain_images.size());
    for (size_t i = 0; i < swapchain_images.size(); ++i) {
        images_[i].image = swapchain_images[i];
        
        // 创建图像视图
        vk::ImageViewCreateInfo viewInfo{};
        viewInfo.setImage(swapchain_images[i])
                .setViewType(vk::ImageViewType::e2D)
                .setFormat(imageFormat_)
                .setSubresourceRange(vk::ImageSubresourceRange(
                    vk::ImageAspectFlagBits::eColor,
                    0, 1, 0, 1
                ));
        
        images_[i].imageView = device.createImageView(viewInfo);
        if (!images_[i].imageView) {
            LOG_ERROR("Failed to create image view");
            throw std::runtime_error("Failed to create image view");
        }
        
        // 不在此处创建Framebuffer，等待RenderPass创建后再创建
        // Framebuffer将在外部通过createFramebuffers方法创建
    }
    
    LOG_INFO("Created {} swapchain images", images_.size());
}

void Swapchain::createFramebuffers(vk::Device device, vk::RenderPass renderPass, vk::Format depthFormat) {
    LOG_INFO("Creating framebuffers with render pass");
    createDepthResources(device, depthFormat);
    
    for (size_t i = 0; i < images_.size(); ++i) {
        std::array<vk::ImageView, 2> attachments = {
            images_[i].imageView,
            depthImageView_
        };

        vk::FramebufferCreateInfo framebufferInfo{};
        framebufferInfo.setRenderPass(renderPass)
                       .setAttachmentCount(static_cast<uint32_t>(attachments.size()))
                       .setPAttachments(attachments.data())
                       .setWidth(extent_.width)
                       .setHeight(extent_.height)
                       .setLayers(1);
        
        images_[i].framebuffer = device.createFramebuffer(framebufferInfo);
        if (!images_[i].framebuffer) {
            LOG_ERROR("Failed to create framebuffer {}", i);
            throw std::runtime_error("Failed to create framebuffer");
        }
    }
    
    LOG_INFO("Created {} framebuffers", images_.size());
}

void Swapchain::createDepthResources(vk::Device device, vk::Format depthFormat) {
    if (depthImageView_) {
        device.destroyImageView(depthImageView_);
        depthImageView_ = nullptr;
    }
    if (depthImage_) {
        device.destroyImage(depthImage_);
        depthImage_ = nullptr;
    }
    if (depthImageMemory_) {
        device.freeMemory(depthImageMemory_);
        depthImageMemory_ = nullptr;
    }

    vk::ImageCreateInfo imageInfo{};
    imageInfo.setImageType(vk::ImageType::e2D)
             .setExtent(vk::Extent3D(extent_.width, extent_.height, 1))
             .setMipLevels(1)
             .setArrayLayers(1)
             .setFormat(depthFormat)
             .setTiling(vk::ImageTiling::eOptimal)
             .setInitialLayout(vk::ImageLayout::eUndefined)
             .setUsage(vk::ImageUsageFlagBits::eDepthStencilAttachment)
             .setSamples(vk::SampleCountFlagBits::e1)
             .setSharingMode(vk::SharingMode::eExclusive);

    depthImage_ = device.createImage(imageInfo);
    vk::MemoryRequirements memRequirements = device.getImageMemoryRequirements(depthImage_);

    vk::MemoryAllocateInfo allocInfo{};
    allocInfo.setAllocationSize(memRequirements.size)
             .setMemoryTypeIndex(findMemoryType(memRequirements.memoryTypeBits,
                                                vk::MemoryPropertyFlagBits::eDeviceLocal));
    depthImageMemory_ = device.allocateMemory(allocInfo);
    device.bindImageMemory(depthImage_, depthImageMemory_, 0);

    vk::ImageViewCreateInfo viewInfo{};
    viewInfo.setImage(depthImage_)
            .setViewType(vk::ImageViewType::e2D)
            .setFormat(depthFormat)
            .setSubresourceRange(vk::ImageSubresourceRange(
                vk::ImageAspectFlagBits::eDepth,
                0, 1, 0, 1
            ));
    depthImageView_ = device.createImageView(viewInfo);
}

void Swapchain::cleanup() {
    auto device = Context::Instance().Device();
    if (swapchain_) {
        for (auto& image : images_) {
            // 销毁帧缓冲
            if (image.framebuffer) {
                device.destroyFramebuffer(image.framebuffer);
            }
            // 销毁图像视图
            if (image.imageView) {
                device.destroyImageView(image.imageView);
            }
        }
        images_.clear();

        if (depthImageView_) {
            device.destroyImageView(depthImageView_);
            depthImageView_ = nullptr;
        }
        if (depthImage_) {
            device.destroyImage(depthImage_);
            depthImage_ = nullptr;
        }
        if (depthImageMemory_) {
            device.freeMemory(depthImageMemory_);
            depthImageMemory_ = nullptr;
        }
        
        device.destroySwapchainKHR(swapchain_);
        swapchain_ = nullptr;
        
        LOG_INFO("Swapchain cleaned up");
    }
}

uint32_t Swapchain::findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) const {
    vk::PhysicalDeviceMemoryProperties memProperties =
        Context::Instance().PhysicalDevice().getMemoryProperties();

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) &&
            (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw std::runtime_error("Failed to find suitable memory type for depth image");
}

void Swapchain::recreateSwapchain(uint32_t width, uint32_t height) {
    LOG_INFO("Recreating swapchain");
    
    // 等待设备空闲
    auto device = Context::Instance().Device();
    device.waitIdle();
    
    // 清理旧的交换链
    cleanup();
    
    // 创建新的交换链
    createSwapchain(width, height);
}

vk::SurfaceFormatKHR Swapchain::chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& available_formats) {
    // Match vkgs: keep the swapchain image UNORM and use SRGB_NONLINEAR only
    // as the presentation color space. Using an SRGB image format applies
    // hardware encode/decode around blending and makes 3DGS colors look too bright.
    for (const auto& available_format : available_formats) {
        if (available_format.format == vk::Format::eB8G8R8A8Unorm &&
            available_format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
            return available_format;
        }
    }

    for (const auto& available_format : available_formats) {
        if (available_format.format == vk::Format::eR8G8B8A8Unorm &&
            available_format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
            return available_format;
        }
    }
    
    // 如果没有理想的格式，返回第一个可用的格式
    return available_formats[0];
}

vk::PresentModeKHR Swapchain::chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& available_present_modes) {
    // 优先选择 Mailbox 模式（三重缓冲，低延迟）
    for (const auto& available_present_mode : available_present_modes) {
        if (available_present_mode == vk::PresentModeKHR::eMailbox) {
            LOG_INFO("Selected present mode: Mailbox");
            return available_present_mode;
        }
    }
    
    // 其次选择 FIFO 模式（垂直同步，保证无撕裂）
    for (const auto& available_present_mode : available_present_modes) {
        if (available_present_mode == vk::PresentModeKHR::eFifo) {
            LOG_INFO("Selected present mode: FIFO (VSync)");
            return available_present_mode;
        }
    }
    
    // 如果都没有，返回第一个可用的
    LOG_WARN("Using fallback present mode");
    return available_present_modes[0];
}

vk::Extent2D Swapchain::chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities, uint32_t width, uint32_t height) {
    // 如果表面有指定的范围，直接使用
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        return capabilities.currentExtent;
    } else {
        // 否则根据窗口大小计算
        vk::Extent2D actualExtent = {width, height};
        
        // 限制在最小和最大范围内
        actualExtent.width = std::clamp(actualExtent.width, 
                                        capabilities.minImageExtent.width, 
                                        capabilities.maxImageExtent.width);
        actualExtent.height = std::clamp(actualExtent.height, 
                                         capabilities.minImageExtent.height, 
                                         capabilities.maxImageExtent.height);
        
        return actualExtent;
    }
}

} // namespace vk_gs
