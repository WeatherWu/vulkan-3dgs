#pragma once

#include <vulkan/vulkan.hpp>
#include <vector>


namespace vulkan3DGS {

enum class PresentModePreference {
    MaxFps = 0,
    LowLatency = 1,
    VSync = 2,
};

class Swapchain {
public:
    struct Image{
        vk::Image image;
        vk::ImageView imageView;
        vk::Framebuffer framebuffer;
    };
    
    Swapchain(vk::SurfaceKHR surface);
    ~Swapchain();

    void createSwapchain(uint32_t width, uint32_t height);
    void createFramebuffers(vk::Device device, vk::RenderPass renderPass, vk::Format depthFormat);
    void cleanup();

    void recreateSwapchain(uint32_t width, uint32_t height);
    void setPresentModePreference(PresentModePreference preference) { presentModePreference_ = preference; }
    PresentModePreference getPresentModePreference() const { return presentModePreference_; }

    vk::SwapchainKHR getSwapchain() const { return swapchain_; }
    const std::vector<Image>& getImages() const { return images_; }
    vk::Format getImageFormat() const { return imageFormat_; }
    vk::Extent2D getExtent() const { return extent_; }
    
    // 获取指定索引的 Framebuffer
    vk::Framebuffer getFramebuffer(uint32_t index) const { 
        return images_.at(index).framebuffer; 
    }
    
    // 获取图像数量
    uint32_t getImageCount() const { return static_cast<uint32_t>(images_.size()); }
    
private:
    vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& available_formats);
    vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& available_present_modes);
    vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities, uint32_t width, uint32_t height);

    vk::SurfaceKHR surface_;

    uint32_t imageCount_;
    vk::SwapchainKHR swapchain_ = nullptr;
    std::vector<Image> images_;
    
    vk::Format imageFormat_;
    vk::Extent2D extent_;

    vk::Image depthImage_ = nullptr;
    vk::DeviceMemory depthImageMemory_ = nullptr;
    vk::ImageView depthImageView_ = nullptr;
    PresentModePreference presentModePreference_ = PresentModePreference::MaxFps;

    void createDepthResources(vk::Device device, vk::Format depthFormat);
    uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) const;
};

} // namespace vulkan3DGS
