#pragma once

#include <vulkan/vulkan.hpp>
#include <memory>
#include <optional>
#include <vector>
#include <string>
#include <GLFW/glfw3.h>

#include "device.hpp"

namespace vulkan3DGS {

class Context {
public:
    Context(GLFWwindow* window);
    ~Context();

    static Context& Instance() {
        static std::unique_ptr<Context> instance(new Context(nullptr));
        return *instance;
    }

    static void initializeVulkanLoader();

    friend class vulkan3DGS::Device;
    
    void initialize(GLFWwindow* window, std::optional<std::string> gpuSelector = std::nullopt);
    void cleanup();
    
    // 查询是否已初始化
    bool isInitialized() const { return device_ != nullptr; }
    
    vk::SurfaceKHR& getSurface() { return surface_; }
    vulkan3DGS::Device& getDevice() { return *device_; }
    vk::Device Device() const { return device_->getDevice(); }
    vk::PhysicalDevice PhysicalDevice() const { return device_->getPhysicalDevice(); }
    vk::Instance getInstance() const { return instance_; }
    vk::Queue getTransferQueue() const { return device_->getTransferQueue(); }
    
    bool isValidationLayersEnabled() const { return enableValidationLayers_; }
    
private:
    void createInstance();
    void createSurface();
    void setupDebugMessenger();
    
    bool checkValidationLayerSupport();
    std::vector<const char*> getRequiredExtensions();
    


    GLFWwindow* window_= nullptr;
    
    vk::Instance instance_ = nullptr;
    vk::DebugUtilsMessengerEXT debug_messenger_ = nullptr;
    vk::SurfaceKHR surface_ = nullptr;

    std::unique_ptr<vulkan3DGS::Device> device_;
    
    bool enableValidationLayers_ = true;  // 启用验证层以调试Pipeline创建问题

    const std::vector<const char*> validationLayers_ = {
        "VK_LAYER_KHRONOS_validation"
    };
};

} // namespace vulkan3DGS
