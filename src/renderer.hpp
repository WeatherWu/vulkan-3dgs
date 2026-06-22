#pragma once

#include <memory>
#include <functional>
#include <utility>
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.hpp>

namespace vk_gs {

struct TrainingExtent;

// ============================================================================
// Renderer — 渲染器基类
// ============================================================================
// 具体渲染器自行持有 swapchain / framebuffer / buffer 等资源。
// 基类只表达输出目标，不向外泄漏 command buffer、framebuffer 或 image index。
// ============================================================================
class Renderer {
public:
    using ImGuiDrawCallback = std::function<void()>;

    virtual ~Renderer() = default;

    virtual void initialize(GLFWwindow* window) = 0;
    virtual void cleanup() = 0;
    virtual void render() = 0;
    virtual void onResize(uint32_t width, uint32_t height) = 0;

    // 渲染到自身管理的 image。需要 acquire 的渲染器在内部完成，但不负责 present。
    virtual void renderToImage() = 0;

    // 呈现最近一次 renderToImage() 生成的可呈现 image。
    virtual void presentImage() = 0;

    // 渲染到自身管理的 buffer。没有离屏输出的渲染器可实现为空操作或抛出明确错误。
    virtual void renderToBuffer() = 0;

    virtual void setImGuiDrawCallback(ImGuiDrawCallback callback) {
        imguiDrawCallback_ = std::move(callback);
    }

protected:
    vk::Device getDevice() const;
    vk::SurfaceKHR getSurface() const;

    ImGuiDrawCallback imguiDrawCallback_;
};

// ============================================================================
// BackwardRenderer — 训练反向传播器基类
// ============================================================================
class BackwardRenderer {
public:
    virtual ~BackwardRenderer() = default;

    virtual void initialize(vk::Device device,
                            vk::PhysicalDevice physicalDevice,
                            vk::Queue computeQueue,
                            uint32_t computeQueueFamilyIndex,
                            uint32_t gaussianCount,
                            TrainingExtent extent) = 0;
    virtual void cleanup() = 0;

    virtual void backward() = 0;

    virtual bool isInitialized() const = 0;
};

} // namespace vk_gs
