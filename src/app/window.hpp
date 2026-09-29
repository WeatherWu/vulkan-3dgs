#pragma once

#include <GLFW/glfw3.h>
#include <string>
#include <functional>
#include <vector>

namespace vulkan3DGS {

class Window {
public:
    using KeyCallback = std::function<void(int key, int action)>;
    using MouseCallback = std::function<void(double x, double y)>;
    using ScrollCallback = std::function<void(double xoffset, double yoffset)>;
    using ResizeCallback = std::function<void(int width, int height)>;
    using DropCallback = std::function<void(const std::vector<std::string>& paths)>;
    
    Window(const std::string& title, int width, int height);
    ~Window();
    
    bool should_close() const;
    void poll_events();
    void swap_buffers();
    
    GLFWwindow* get_handle() const { return window_; }
    
    void set_key_callback(KeyCallback callback) { key_callback_ = callback; }
    void set_mouse_callback(MouseCallback callback) { mouse_callback_ = callback; }
    void set_scroll_callback(ScrollCallback callback) { scroll_callback_ = callback; }
    void set_resize_callback(ResizeCallback callback) { resize_callback_ = callback; }
    void set_drop_callback(DropCallback callback) { drop_callback_ = callback; }
    
    int get_width() const { return width_; }
    int get_height() const { return height_; }
    
private:
    static void key_callback_wrapper(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void mouse_callback_wrapper(GLFWwindow* window, double xpos, double ypos);
    static void scroll_callback_wrapper(GLFWwindow* window, double xoffset, double yoffset);
    static void resize_callback_wrapper(GLFWwindow* window, int width, int height);
    static void drop_callback_wrapper(GLFWwindow* window, int count, const char** paths);
    
    GLFWwindow* window_;
    int width_, height_;
    
    KeyCallback key_callback_;
    MouseCallback mouse_callback_;
    ScrollCallback scroll_callback_;
    ResizeCallback resize_callback_;
    DropCallback drop_callback_;
};

} // namespace vulkan3DGS
