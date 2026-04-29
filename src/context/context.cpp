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

namespace vk_gs {

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
            LOG_INFO("Validation layer: {}", pCallbackData->pMessage);
            break;
        case vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning:
            LOG_WARN("Validation layer: {}", pCallbackData->pMessage);
            break;
        case vk::DebugUtilsMessageSeverityFlagBitsEXT::eError:
            LOG_ERROR("Validation layer: {}", pCallbackData->pMessage);
            break;
        default:
            LOG_INFO("Validation layer: {}", pCallbackData->pMessage);
            break;
    }
    
    return VK_FALSE;
}

Context::Context(GLFWwindow* window) : window_(window) {
    LOG_INFO("Creating Vulkan context");
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
    
    LOG_INFO("Starting Context initialization");
    
    window_ = window;
    
    LOG_INFO("Creating Vulkan instance");
    createInstance();
    
    LOG_INFO("Creating surface");
    createSurface();
    
    if (enableValidationLayers_) {
        LOG_INFO("Setting up debug messenger");
        setupDebugMessenger();
    }

    LOG_INFO("Creating logical device");
    device_ = std::make_unique<vk_gs::Device>(surface_);
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
    
    LOG_INFO("Vulkan context cleaned up");
}

void Context::createInstance() {
    LOG_INFO("Initializing Vulkan dynamic loader");
    // 第一步：初始化动态加载器（必须在任何Vulkan API调用之前）
    static bool dispatcher_initialized = false;
    if (!dispatcher_initialized) {
        VULKAN_HPP_DEFAULT_DISPATCHER.init();
        dispatcher_initialized = true;
        LOG_INFO("Vulkan dynamic loader initialized");
    }
    
    LOG_INFO("Checking validation layers");
    if (enableValidationLayers_ && !checkValidationLayerSupport()) {
        LOG_WARN("Validation layers requested, but not available!");
        enableValidationLayers_ = false;
    }
    
    LOG_INFO("Setting up application info");
    vk::ApplicationInfo appInfo{};
    appInfo.setPApplicationName("Vulkan 3DGS")
           .setApplicationVersion(vk::makeVersion(1, 0, 0))
           .setPEngineName("No Engine")
           .setEngineVersion(vk::makeVersion(1, 0, 0))
           .setApiVersion(vk::ApiVersion12);  // 使用Vulkan 1.2以支持SPIR-V 1.5
    
    LOG_INFO("Getting required extensions");
    vk::InstanceCreateInfo createInfo{};
    createInfo.setPApplicationInfo(&appInfo);
    
    auto extensions = getRequiredExtensions();
    LOG_INFO("Got {} extensions", extensions.size());
    for (size_t i = 0; i < extensions.size(); ++i) {
        LOG_INFO("  Extension {}: {}", i, extensions[i]);
    }
    
    createInfo.setEnabledExtensionCount(static_cast<uint32_t>(extensions.size()))
              .setPEnabledExtensionNames(extensions);
    
    vk::DebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
    if (enableValidationLayers_) {
        LOG_INFO("Enabling validation layers");
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
        LOG_INFO("Validation layers disabled");
        createInfo.setEnabledLayerCount(0);
    }
    
    LOG_INFO("Calling vk::createInstance");
    try {
        instance_ = vk::createInstance(createInfo, nullptr);
        LOG_INFO("Vulkan instance created successfully");
        
        // 第二步：使用Instance更新调度器以获取Instance级别的函数
        VULKAN_HPP_DEFAULT_DISPATCHER.init(instance_);
        LOG_INFO("Vulkan dispatcher updated with instance");
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
    LOG_INFO("Window surface created successfully");
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
    LOG_INFO("Debug messenger created successfully");
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

} // namespace vk_gs