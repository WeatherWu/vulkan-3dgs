#include "context.hpp"
#include "utils/logger.hpp"
#include "device.hpp"

#include <vulkan/vulkan.hpp>
#include <vulkan/vk_enum_string_helper.h>

// 定义动态加载器实例（必须在且仅在一个 .cpp 文件中定义）
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

#include <stdexcept>
#include <set>
#include <string>
#include <memory>

namespace vulkan3DGS {

// Vulkan调试消息回调函数
static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    vk::DebugUtilsMessageTypeFlagsEXT messageType,
    const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData,    
    void* pUserData) {
    
    switch (messageSeverity) {
        case vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose:
            LOG_DEBUG("Validation layer: {}", pCallbackData->pMessage);
            break;
        case vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo:
            LOG_DEBUG("Validation layer: {}", pCallbackData->pMessage);
            break;
        case vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning:
            LOG_WARN("Validation layer: {}", pCallbackData->pMessage);
            break;
        case vk::DebugUtilsMessageSeverityFlagBitsEXT::eError:
            LOG_ERROR("Validation layer: {}", pCallbackData->pMessage);
            break;
        default:
            LOG_DEBUG("Validation layer: {}", pCallbackData->pMessage);
            break;
    }
    
    return VK_FALSE;
}

Context::Context(GLFWwindow* window) : window_(window) {
}

Context::~Context() {
    cleanup();
}

void Context::initialize(GLFWwindow* window) {
    // 防止重复初始化导致设备丢失
    if (device_ != nullptr) {
        LOG_WARN("Context already initialized, skipping duplicate initialization");
        return;
    }
    
    window_ = window;
    
    createInstance();
    
    createSurface();
    
    if (enableValidationLayers_) {
        setupDebugMessenger();
    }

    device_ = std::make_unique<vulkan3DGS::Device>(surface_);
    device_->createDevice();
    
    LOG_INFO("Vulkan context initialized successfully");
}

void Context::cleanup() {
    if (device_ != nullptr) {
        device_.reset();
    }
    
    if (debug_messenger_) {
        instance_.destroyDebugUtilsMessengerEXT(debug_messenger_);
    }
    
    if (surface_) {
        instance_.destroySurfaceKHR(surface_);
    }
    
    if (instance_) {
        instance_.destroy();
    }
}

void Context::createInstance() {
    // 第一步：初始化动态加载器（必须在任何Vulkan API调用之前）
    static bool dispatcher_initialized = false;
    if (!dispatcher_initialized) {
        VULKAN_HPP_DEFAULT_DISPATCHER.init();
        dispatcher_initialized = true;
        LOG_DEBUG("Vulkan dynamic loader initialized");
    }
    
    if (enableValidationLayers_ && !checkValidationLayerSupport()) {
        LOG_WARN("Validation layers requested, but not available!");
        enableValidationLayers_ = false;
    }
    
    vk::ApplicationInfo appInfo{};
    appInfo.setPApplicationName("vulkan-3dgs")
           .setApplicationVersion(vk::makeVersion(1, 0, 0))
           .setPEngineName("No Engine")
           .setEngineVersion(vk::makeVersion(1, 0, 0))
           .setApiVersion(vk::ApiVersion12);  // 使用Vulkan 1.2以支持SPIR-V 1.5
    
    vk::InstanceCreateInfo createInfo{};
    createInfo.setPApplicationInfo(&appInfo);
    
    auto extensions = getRequiredExtensions();
    LOG_DEBUG("Got {} extensions", extensions.size());
    for (size_t i = 0; i < extensions.size(); ++i) {
        LOG_DEBUG("  Extension {}: {}", i, extensions[i]);
    }
    
    createInfo.setEnabledExtensionCount(static_cast<uint32_t>(extensions.size()))
              .setPEnabledExtensionNames(extensions);
    
    vk::DebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
    if (enableValidationLayers_) {
        LOG_DEBUG("Enabling validation layers");
        createInfo.setEnabledLayerCount(static_cast<uint32_t>(validationLayers_.size()))
                  .setPEnabledLayerNames(validationLayers_);

        debugCreateInfo.setMessageSeverity(vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
                                            vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
                                            vk::DebugUtilsMessageSeverityFlagBitsEXT::eError);
        debugCreateInfo.setMessageType(vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | 
                                        vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation | 
                                        vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance);
        debugCreateInfo.setPfnUserCallback(&debugCallback);
        
        createInfo.setPNext(&debugCreateInfo);
    } else {
        LOG_DEBUG("Validation layers disabled");
        createInfo.setEnabledLayerCount(0);
    }
    
    try {
        instance_ = vk::createInstance(createInfo, nullptr);
        LOG_DEBUG("Vulkan instance created successfully");
        
        // 第二步：使用Instance更新调度器以获取Instance级别的函数
        VULKAN_HPP_DEFAULT_DISPATCHER.init(instance_);
        LOG_DEBUG("Vulkan dispatcher updated with instance");
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to create Vulkan instance: {}", e.what());
        throw std::runtime_error("Failed to create Vulkan instance");
    }
}

void Context::createSurface() {
    VkSurfaceKHR surface;
    auto result = glfwCreateWindowSurface(instance_, window_, nullptr, &surface);
    if (result != VK_SUCCESS) {
        LOG_ERROR("Failed to create window surface");
        throw std::runtime_error("Failed to create window surface");
    }
    surface_ = surface;
    LOG_DEBUG("Window surface created successfully");
}

void Context::setupDebugMessenger() {
    vk::DebugUtilsMessengerCreateInfoEXT createInfo{};
    createInfo.setMessageSeverity(vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
                                  vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
                                  vk::DebugUtilsMessageSeverityFlagBitsEXT::eError);
    createInfo.setMessageType(vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | 
                              vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation | 
                              vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance);
    createInfo.setPfnUserCallback(&debugCallback);
    
    debug_messenger_ = instance_.createDebugUtilsMessengerEXT(createInfo);
    if(!debug_messenger_) {
        LOG_ERROR("Failed to create debug messenger");
        throw std::runtime_error("Failed to create debug messenger");
    }
    LOG_DEBUG("Debug messenger created successfully");
}

bool Context::checkValidationLayerSupport() {
    uint32_t layer_count;
    auto result = vk::enumerateInstanceLayerProperties(&layer_count, nullptr);
    if (result != vk::Result::eSuccess) {
        LOG_ERROR("Failed to enumerate instance layer properties data");
        return false;
    }

    std::vector<vk::LayerProperties> available_layers(layer_count);
    result = vk::enumerateInstanceLayerProperties(&layer_count, available_layers.data());
    if (result != vk::Result::eSuccess) {
        LOG_ERROR("Failed to enumerate instance layer properties");
        return false;
    }

    for (const char* layer_name : validationLayers_) {
        bool layer_found = false;
        
        for (const auto& layer_properties : available_layers) {
            if (strcmp(layer_name, layer_properties.layerName) == 0) {
                layer_found = true;
                break;
            }
        }
        
        if (!layer_found) {
            return false;
        }
    }
    
    return true;
}

std::vector<const char*> Context::getRequiredExtensions() {
    uint32_t glfw_extension_count = 0;
    const char** glfw_extensions;
    glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_extension_count);
    
    std::vector<const char*> extensions(glfw_extensions, glfw_extensions + glfw_extension_count);
    
    if (enableValidationLayers_) {
        extensions.push_back(vk::EXTDebugUtilsExtensionName);
    }
    
    return extensions;
}

} // namespace vulkan3DGS
