#pragma once

#include <memory>
#include <functional>
#include <utility>
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.hpp>

namespace vk_gs {

// 渲染器基类（完全抽象化）
class Renderer {
public:
    using ImGuiDrawCallback = std::function<void()>;

    virtual ~Renderer() = default;

    // 纯虚接口
    virtual void initialize(GLFWwindow* window) = 0;
    virtual void cleanup() = 0;
    virtual void render() = 0;
    virtual void onResize(uint32_t width, uint32_t height) = 0;
    virtual void setImGuiDrawCallback(ImGuiDrawCallback callback) {
        imguiDrawCallback_ = std::move(callback);
    }
    
protected:
    // 提供底层环境访问的工具方法
    vk::Device getDevice() const;
    vk::SurfaceKHR getSurface() const;

    ImGuiDrawCallback imguiDrawCallback_;
};

} // namespace vk_gs
