#pragma once

#include "vulkan/command_pool.hpp"
#include "vulkan/swapchain.hpp"

#include <GLFW/glfw3.h>
#include <vulkan/vulkan.hpp>

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

namespace vulkan3DGS {

class GraphicsFrameRuntime {
public:
    using FormatChangedHandler = std::function<vk::RenderPass(vk::Format, vk::Extent2D)>;

    GraphicsFrameRuntime() = default;
    ~GraphicsFrameRuntime() noexcept;

    GraphicsFrameRuntime(const GraphicsFrameRuntime&) = delete;
    GraphicsFrameRuntime& operator=(const GraphicsFrameRuntime&) = delete;

    void initialize(GLFWwindow* window, PresentModePreference preference, vk::Format depthFormat,
                    const FormatChangedHandler& createRenderPass);
    void cleanup();

    bool ensureReady(vk::RenderPass renderPass, vk::Format depthFormat,
                     const FormatChangedHandler& onFormatChanged);
    std::optional<uint32_t> acquireFrame(vk::RenderPass renderPass, vk::Format depthFormat,
                                         const FormatChangedHandler& onFormatChanged);
    void submit(uint32_t imageIndex);
    void present(vk::RenderPass renderPass, vk::Format depthFormat,
                 const FormatChangedHandler& onFormatChanged);

    void requestResize() {
        recreationPending_ = true;
    }
    void setPresentModePreference(PresentModePreference preference);

    uint32_t currentFrame() const {
        return currentFrame_;
    }
    uint32_t frameCount() const {
        return frameResourceCount_;
    }
    uint32_t imageCount() const {
        return swapchainImageCount_;
    }
    vk::Extent2D extent() const {
        return swapchain_->getExtent();
    }
    vk::Format imageFormat() const {
        return swapchain_->getImageFormat();
    }
    vk::Framebuffer framebuffer(uint32_t imageIndex) const {
        return swapchain_->getFramebuffer(imageIndex);
    }
    vk::CommandBuffer commandBuffer() const {
        return commandBuffers_[currentFrame_];
    }
    const std::vector<vk::Fence>& inFlightFences() const {
        return inFlightFences_;
    }

private:
    void createSyncObjects();
    void recreateRenderFinishedSemaphores();
    void recreate(uint32_t width, uint32_t height, vk::RenderPass renderPass,
                  vk::Format depthFormat, const FormatChangedHandler& onFormatChanged);

    GLFWwindow* window_ = nullptr;
    std::unique_ptr<Swapchain> swapchain_;
    CommandPool commandPool_;
    std::vector<vk::CommandBuffer> commandBuffers_;
    std::vector<vk::Semaphore> imageAvailableSemaphores_;
    std::vector<vk::Semaphore> renderFinishedSemaphores_;
    std::vector<vk::Fence> inFlightFences_;
    uint32_t currentFrame_ = 0;
    uint32_t swapchainImageCount_ = 0;
    uint32_t frameResourceCount_ = 0;
    uint32_t acquiredImageIndex_ = 0;
    bool imageReadyForPresent_ = false;
    bool presentWaitSemaphoreConsumed_ = false;
    bool recreationPending_ = false;
};

} // namespace vulkan3DGS
