#pragma once

#include <GLFW/glfw3.h>
#include <string>
#include <functional>

namespace vk_gs {

class Window {
public:
    using KeyCallback = std::function<void(int key, int action)>;
    using MouseCallback = std::function<void(double x, double y)>;
    using ResizeCallback = std::function<void(int width, int height)>;
    
    Window(const std::string& title, int width, int height);
    ~Window();
    
    bool should_close() const;
    void poll_events();
    void swap_buffers();
    
    GLFWwindow* get_handle() const { return window_; }
    
    void set_key_callback(KeyCallback callback) { key_callback_ = callback; }
    void set_mouse_callback(MouseCallback callback) { mouse_callback_ = callback; }
    void set_resize_callback(ResizeCallback callback) { resize_callback_ = callback; }
    
    int get_width() const { return width_; }
    int get_height() const { return height_; }
    
private:
    static void key_callback_wrapper(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void mouse_callback_wrapper(GLFWwindow* window, double xpos, double ypos);
    static void resize_callback_wrapper(GLFWwindow* window, int width, int height);
    
    GLFWwindow* window_;
    int width_, height_;
    
    KeyCallback key_callback_;
    MouseCallback mouse_callback_;
    ResizeCallback resize_callback_;
};

} // namespace vk_gs