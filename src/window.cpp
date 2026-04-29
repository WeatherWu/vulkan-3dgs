#include "window.hpp"
#include "utils/logger.hpp"

namespace vk_gs {

Window::Window(const std::string& title, int width, int height) 
    : width_(width), height_(height) {
    
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // 不使用OpenGL
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    
    window_ = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
    if (!window_) {
        LOG_ERROR("Failed to create GLFW window");
        throw std::runtime_error("GLFW window creation failed");
    }
    
    // 设置回调包装器
    glfwSetWindowUserPointer(window_, this);
    glfwSetKeyCallback(window_, key_callback_wrapper);
    glfwSetCursorPosCallback(window_, mouse_callback_wrapper);
    glfwSetFramebufferSizeCallback(window_, resize_callback_wrapper);
    
    LOG_INFO("Created window: {} ({}x{})", title, width, height);
}

Window::~Window() {
    if (window_) {
        glfwDestroyWindow(window_);
        window_ = nullptr;
    }
}

bool Window::should_close() const {
    return glfwWindowShouldClose(window_);
}

void Window::poll_events() {
    glfwPollEvents();
}

void Window::swap_buffers() {
    // Vulkan不使用glfwSwapBuffers
}

void Window::key_callback_wrapper(GLFWwindow* window, int key, int scancode, int action, int mods) {
    auto* win = static_cast<Window*>(glfwGetWindowUserPointer(window));
    if (win && win->key_callback_) {
        win->key_callback_(key, action);
    }
}

void Window::mouse_callback_wrapper(GLFWwindow* window, double xpos, double ypos) {
    auto* win = static_cast<Window*>(glfwGetWindowUserPointer(window));
    if (win && win->mouse_callback_) {
        win->mouse_callback_(xpos, ypos);
    }
}

void Window::resize_callback_wrapper(GLFWwindow* window, int width, int height) {
    auto* win = static_cast<Window*>(glfwGetWindowUserPointer(window));
    if (win) {
        win->width_ = width;
        win->height_ = height;
        if (win->resize_callback_) {
            win->resize_callback_(width, height);
        }
    }
}

} // namespace vk_gs