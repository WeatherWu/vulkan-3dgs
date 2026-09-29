#include "graphics_frame_runtime.hpp"

#include "context/context.hpp"
#include "utils/logger.hpp"

#include <cstdio>
#include <stdexcept>

namespace vulkan3DGS {

GraphicsFrameRuntime::~GraphicsFrameRuntime() noexcept {
    try {
        cleanup();
    } catch (...) {
        std::fputs("GraphicsFrameRuntime cleanup failed during destruction\n", stderr);
    }
}

void GraphicsFrameRuntime::initialize(GLFWwindow* window, PresentModePreference preference,
                                      vk::Format depthFormat,
                                      const FormatChangedHandler& createRenderPass) {
    cleanup();
    window_ = window;

    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(window_, &width, &height);
    while (width <= 0 || height <= 0) {
        glfwWaitEvents();
        glfwGetFramebufferSize(window_, &width, &height);
    }

    auto& context = Context::Instance();
    swapchain_ = std::make_unique<Swapchain>(context.getSurface());
    swapchain_->setPresentModePreference(preference);
    swapchain_->createSwapchain(static_cast<uint32_t>(width), static_cast<uint32_t>(height));
    const vk::RenderPass renderPass =
        createRenderPass(swapchain_->getImageFormat(), swapchain_->getExtent());
    swapchain_->createFramebuffers(context.getDevice().getDevice(), renderPass, depthFormat);
    createSyncObjects();
}

void GraphicsFrameRuntime::cleanup() {
    if (!swapchain_ && !window_ && imageAvailableSemaphores_.empty() &&
        renderFinishedSemaphores_.empty() && inFlightFences_.empty()) {
        return;
    }
    const vk::Device device = Context::Instance().getDevice().getDevice();
    if (device) {
        device.waitIdle();
    }
    for (vk::Semaphore semaphore : imageAvailableSemaphores_) {
        if (semaphore) {
            device.destroySemaphore(semaphore);
        }
    }
    for (vk::Semaphore semaphore : renderFinishedSemaphores_) {
        if (semaphore) {
            device.destroySemaphore(semaphore);
        }
    }
    for (vk::Fence fence : inFlightFences_) {
        if (fence) {
            device.destroyFence(fence);
        }
    }
    imageAvailableSemaphores_.clear();
    renderFinishedSemaphores_.clear();
    inFlightFences_.clear();
    commandBuffers_.clear();
    commandPool_.cleanup();
    swapchain_.reset();
    window_ = nullptr;
    currentFrame_ = 0;
    swapchainImageCount_ = 0;
    frameResourceCount_ = 0;
    imageReadyForPresent_ = false;
    presentWaitSemaphoreConsumed_ = false;
    recreationPending_ = false;
}

bool GraphicsFrameRuntime::ensureReady(vk::RenderPass renderPass, vk::Format depthFormat,
                                       const FormatChangedHandler& onFormatChanged) {
    if (!window_ || !swapchain_) {
        return false;
    }

    int framebufferWidth = 0;
    int framebufferHeight = 0;
    glfwGetFramebufferSize(window_, &framebufferWidth, &framebufferHeight);
    if (framebufferWidth <= 0 || framebufferHeight <= 0) {
        recreationPending_ = true;
        imageReadyForPresent_ = false;
        return false;
    }

    const vk::Extent2D currentExtent = swapchain_->getExtent();
    const uint32_t width = static_cast<uint32_t>(framebufferWidth);
    const uint32_t height = static_cast<uint32_t>(framebufferHeight);
    if (recreationPending_ || currentExtent.width != width || currentExtent.height != height) {
        recreate(width, height, renderPass, depthFormat, onFormatChanged);
    }
    return true;
}

std::optional<uint32_t>
GraphicsFrameRuntime::acquireFrame(vk::RenderPass renderPass, vk::Format depthFormat,
                                   const FormatChangedHandler& onFormatChanged) {
    imageReadyForPresent_ = false;
    presentWaitSemaphoreConsumed_ = false;
    if (!ensureReady(renderPass, depthFormat, onFormatChanged)) {
        return std::nullopt;
    }

    const vk::Device device = Context::Instance().getDevice().getDevice();
    const vk::Result waitResult =
        device.waitForFences(1, &inFlightFences_[currentFrame_], VK_TRUE, UINT64_MAX);
    if (waitResult != vk::Result::eSuccess) {
        throw std::runtime_error("Failed to wait for graphics frame fence");
    }

    uint32_t imageIndex = 0;
    vk::Result result = vk::Result::eSuccess;
    try {
        result = device.acquireNextImageKHR(swapchain_->getSwapchain(), UINT64_MAX,
                                            imageAvailableSemaphores_[currentFrame_], nullptr,
                                            &imageIndex);
    } catch (const vk::OutOfDateKHRError&) {
        result = vk::Result::eErrorOutOfDateKHR;
    }
    if (result == vk::Result::eErrorOutOfDateKHR) {
        LOG_WARN("Swapchain out of date, recreating");
        recreationPending_ = true;
        (void)ensureReady(renderPass, depthFormat, onFormatChanged);
        return std::nullopt;
    }
    if (result == vk::Result::eSuboptimalKHR) {
        LOG_INFO("Suboptimal swapchain detected during acquire");
        recreationPending_ = true;
    } else if (result != vk::Result::eSuccess) {
        throw std::runtime_error("Failed to acquire swapchain image: " + vk::to_string(result));
    }
    acquiredImageIndex_ = imageIndex;
    return imageIndex;
}

void GraphicsFrameRuntime::submit(uint32_t imageIndex) {
    auto& context = Context::Instance();
    const vk::Device device = context.getDevice().getDevice();
    vk::SubmitInfo submitInfo{};
    const vk::Semaphore waitSemaphore = imageAvailableSemaphores_[currentFrame_];
    const vk::PipelineStageFlags waitStage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
    const vk::CommandBuffer buffer = commandBuffers_[currentFrame_];
    const vk::Semaphore signalSemaphore = renderFinishedSemaphores_[imageIndex];
    submitInfo.setWaitSemaphoreCount(1)
        .setPWaitSemaphores(&waitSemaphore)
        .setPWaitDstStageMask(&waitStage)
        .setCommandBufferCount(1)
        .setPCommandBuffers(&buffer)
        .setSignalSemaphoreCount(1)
        .setPSignalSemaphores(&signalSemaphore);
    device.resetFences(inFlightFences_[currentFrame_]);
    context.getDevice().getGraphicsQueue().submit(submitInfo, inFlightFences_[currentFrame_]);
    acquiredImageIndex_ = imageIndex;
    imageReadyForPresent_ = true;
}

void GraphicsFrameRuntime::present(vk::RenderPass renderPass, vk::Format depthFormat,
                                   const FormatChangedHandler& onFormatChanged) {
    if (!imageReadyForPresent_) {
        return;
    }

    auto& context = Context::Instance();
    const vk::Semaphore waitSemaphore = renderFinishedSemaphores_[acquiredImageIndex_];
    const vk::SwapchainKHR swapchain = swapchain_->getSwapchain();
    vk::PresentInfoKHR presentInfo{};
    presentInfo.setWaitSemaphoreCount(presentWaitSemaphoreConsumed_ ? 0u : 1u)
        .setPWaitSemaphores(presentWaitSemaphoreConsumed_ ? nullptr : &waitSemaphore)
        .setSwapchainCount(1)
        .setPSwapchains(&swapchain)
        .setPImageIndices(&acquiredImageIndex_);

    vk::Result result = vk::Result::eSuccess;
    bool recreateAfterPresent = recreationPending_;
    try {
        result = context.getDevice().getPresentQueue().presentKHR(presentInfo);
    } catch (const vk::OutOfDateKHRError&) {
        result = vk::Result::eErrorOutOfDateKHR;
    } catch (...) {
        imageReadyForPresent_ = false;
        presentWaitSemaphoreConsumed_ = false;
        throw;
    }
    if (result == vk::Result::eErrorOutOfDateKHR) {
        LOG_WARN("Swapchain out of date during present");
        recreateAfterPresent = true;
    } else if (result == vk::Result::eSuboptimalKHR) {
        LOG_INFO("Suboptimal swapchain detected during present");
        recreateAfterPresent = true;
    } else if (result != vk::Result::eSuccess) {
        imageReadyForPresent_ = false;
        throw std::runtime_error("Failed to present image: " + vk::to_string(result));
    }

    imageReadyForPresent_ = false;
    presentWaitSemaphoreConsumed_ = false;
    if (frameResourceCount_ > 0u) {
        currentFrame_ = (currentFrame_ + 1u) % frameResourceCount_;
    }
    if (recreateAfterPresent) {
        recreationPending_ = true;
        (void)ensureReady(renderPass, depthFormat, onFormatChanged);
    }
}

void GraphicsFrameRuntime::setPresentModePreference(PresentModePreference preference) {
    if (!swapchain_) {
        return;
    }
    swapchain_->setPresentModePreference(preference);
    recreationPending_ = true;
}

void GraphicsFrameRuntime::createSyncObjects() {
    auto& context = Context::Instance();
    const vk::Device device = context.getDevice().getDevice();
    swapchainImageCount_ = swapchain_->getImageCount();
    frameResourceCount_ = swapchainImageCount_;
    imageAvailableSemaphores_.resize(frameResourceCount_);
    renderFinishedSemaphores_.resize(swapchainImageCount_);
    inFlightFences_.resize(frameResourceCount_);
    commandBuffers_.resize(frameResourceCount_);

    vk::SemaphoreCreateInfo semaphoreInfo{};
    vk::FenceCreateInfo fenceInfo{};
    fenceInfo.setFlags(vk::FenceCreateFlagBits::eSignaled);
    for (uint32_t index = 0; index < frameResourceCount_; ++index) {
        imageAvailableSemaphores_[index] = device.createSemaphore(semaphoreInfo);
        inFlightFences_[index] = device.createFence(fenceInfo);
    }
    for (uint32_t index = 0; index < swapchainImageCount_; ++index) {
        renderFinishedSemaphores_[index] = device.createSemaphore(semaphoreInfo);
    }

    const auto graphicsIndex = context.getDevice().getQueueFamilyIndices().graphicsIndex;
    if (!graphicsIndex) throw std::runtime_error("Graphics queue family is unavailable");
    commandPool_.create(device, *graphicsIndex, vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
    for (vk::CommandBuffer& commandBuffer : commandBuffers_) {
        commandBuffer = commandPool_.allocateCommandBuffer();
    }
}

void GraphicsFrameRuntime::recreateRenderFinishedSemaphores() {
    const vk::Device device = Context::Instance().getDevice().getDevice();
    for (vk::Semaphore semaphore : renderFinishedSemaphores_) {
        if (semaphore) {
            device.destroySemaphore(semaphore);
        }
    }
    swapchainImageCount_ = swapchain_->getImageCount();
    renderFinishedSemaphores_.assign(swapchainImageCount_, vk::Semaphore{});
    vk::SemaphoreCreateInfo semaphoreInfo{};
    for (vk::Semaphore& semaphore : renderFinishedSemaphores_) {
        semaphore = device.createSemaphore(semaphoreInfo);
    }
}

void GraphicsFrameRuntime::recreate(uint32_t width, uint32_t height, vk::RenderPass renderPass,
                                    vk::Format depthFormat,
                                    const FormatChangedHandler& onFormatChanged) {
    if (width == 0u || height == 0u) {
        recreationPending_ = true;
        return;
    }
    LOG_INFO("Recreating swapchain: {}x{}", width, height);
    const vk::Device device = Context::Instance().getDevice().getDevice();
    device.waitIdle();
    const vk::Format oldFormat = swapchain_->getImageFormat();
    swapchain_->recreateSwapchain(width, height);
    recreateRenderFinishedSemaphores();
    if (oldFormat != swapchain_->getImageFormat()) {
        renderPass = onFormatChanged(swapchain_->getImageFormat(), swapchain_->getExtent());
    }
    swapchain_->createFramebuffers(device, renderPass, depthFormat);
    imageReadyForPresent_ = false;
    presentWaitSemaphoreConsumed_ = false;
    recreationPending_ = false;
}

} // namespace vulkan3DGS
