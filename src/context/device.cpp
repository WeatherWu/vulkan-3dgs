#include "device.hpp"
#include "context.hpp"
#include "utils/logger.hpp"
#include "vulkan/command_pool.hpp"

#include <set>

namespace vk_gs {

Device::Device(vk::SurfaceKHR surface) : surface_(surface) {
    LOG_INFO("Device object created, waiting for explicit initialization");
}

Device::~Device() {
    cleanup();
}
    
void Device::createDevice() {
    LOG_INFO("Picking physical device");
    pickPhysicalDevice(surface_);
    
    LOG_INFO("Creating logical device");
    vk::DeviceCreateInfo createInfo;
    createInfo.setPEnabledExtensionNames(deviceExtensions_);
    std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
    float properties = 1.0;
    
    if (!queueFamilyIndices_.graphicsIndex || !queueFamilyIndices_.presentIndex) {
        LOG_ERROR("Queue family indices not set");
        throw std::runtime_error("Queue family indices not set");
    }
    
    // 收集所有需要创建的队列家族索引（去重）
    std::set<uint32_t> uniqueQueueFamilies = {
        queueFamilyIndices_.graphicsIndex.value(),
        queueFamilyIndices_.presentIndex.value()
    };
    
    // 如果有专用传输队列且与图形/呈现队列不同，则添加
    if (queueFamilyIndices_.transferIndex.has_value()) {
        uniqueQueueFamilies.insert(queueFamilyIndices_.transferIndex.value());
    }
    
    // 如果有专用计算队列且与其他队列不同，则添加
    if (queueFamilyIndices_.computeIndex.has_value()) {
        uniqueQueueFamilies.insert(queueFamilyIndices_.computeIndex.value());
    }

    // 为每个唯一的队列家族创建队列创建信息
    for (uint32_t queueFamily : uniqueQueueFamilies) {
        vk::DeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.setQueueFamilyIndex(queueFamily)
                       .setQueueCount(1)
                       .setPQueuePriorities(&properties);
        queueCreateInfos.push_back(queueCreateInfo);
    }
    
    createInfo.setQueueCreateInfos(queueCreateInfos);

    // 配置设备特性
    vk::PhysicalDeviceFeatures deviceFeatures{};
    createInfo.setPEnabledFeatures(&deviceFeatures);

    device_ = phyDevice_.createDevice(createInfo);
    if (!device_) {
        LOG_ERROR("Failed to create Vulkan device");
        throw std::runtime_error("Failed to create Vulkan device");
    }
    
    // 第三步：使用Device更新调度器以获取Device级别的函数
    VULKAN_HPP_DEFAULT_DISPATCHER.init(device_);
    LOG_INFO("Vulkan dispatcher updated with device");
    
    // 第四步：获取队列句柄
    LOG_INFO("Retrieving queue handles");
    graphicsQueue_ = device_.getQueue(queueFamilyIndices_.graphicsIndex.value(), 0);
    presentQueue_ = device_.getQueue(queueFamilyIndices_.presentIndex.value(), 0);
    
    if (queueFamilyIndices_.transferIndex.has_value()) {
        transferQueue_ = device_.getQueue(queueFamilyIndices_.transferIndex.value(), 0);
        LOG_INFO("Dedicated transfer queue retrieved (family {})", queueFamilyIndices_.transferIndex.value());
    } else {
        // 如果没有专用传输队列，回退到图形队列
        transferQueue_ = graphicsQueue_;
        LOG_INFO("Using graphics queue as transfer queue");
    }
    
    if (queueFamilyIndices_.computeIndex.has_value()) {
        computeQueue_ = device_.getQueue(queueFamilyIndices_.computeIndex.value(), 0);
        LOG_INFO("Dedicated compute queue retrieved (family {})", queueFamilyIndices_.computeIndex.value());
    } else {
        // 如果没有专用计算队列，回退到图形队列
        computeQueue_ = graphicsQueue_;
        LOG_INFO("Using graphics queue as compute queue");
    }
    
    LOG_INFO("All queue handles retrieved successfully");
    LOG_INFO("Vulkan device created successfully");
}


void Device::pickPhysicalDevice(vk::SurfaceKHR& surface) {
    LOG_INFO("Enumerating physical devices");
    auto& context = Context::Instance();
    auto devices = context.getInstance().enumeratePhysicalDevices();
    
    LOG_INFO("Found {} physical device(s)", devices.size());
    
    if (devices.empty()) {
        LOG_ERROR("Failed to find GPUs with Vulkan support");
        throw std::runtime_error("Failed to find GPUs with Vulkan support");
    }
    
    for (size_t i = 0; i < devices.size(); ++i) {
        LOG_INFO("Checking device {}", i);
        if (isDeviceSuitable(devices[i], surface)) {
            phyDevice_ = devices[i];
            LOG_INFO("Selected device {}", i);
            break;
        }
    }
    
    if (phyDevice_ == VK_NULL_HANDLE) {
        LOG_ERROR("Failed to find a suitable GPU");
        throw std::runtime_error("Failed to find a suitable GPU");
    }
    
    vk::PhysicalDeviceProperties device_properties = phyDevice_.getProperties();
    LOG_INFO("Selected physical device: {}", device_properties.deviceName);
}

bool Device::isDeviceSuitable(vk::PhysicalDevice device, vk::SurfaceKHR surface) {
    // 查找队列家族
    uint32_t queue_family_count = 0;
    device.getQueueFamilyProperties(&queue_family_count, nullptr);
    std::vector<vk::QueueFamilyProperties> queue_families(queue_family_count);
    device.getQueueFamilyProperties(&queue_family_count, queue_families.data());
    
    for (uint32_t i = 0; i < queue_family_count; i++) {
        const auto& queueFamily = queue_families[i];
        
        // 查找图形队列
        if (queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) {
            queueFamilyIndices_.graphicsIndex = i;
        }
        
        // 查找呈现队列
        vk::Bool32 present_support = false;
        vk::Result result = device.getSurfaceSupportKHR(i, surface, &present_support);
        if (result == vk::Result::eSuccess && present_support) {
            queueFamilyIndices_.presentIndex = i;
        }

        // 查找专用计算队列（支持计算但不支持图形）
        if ((queueFamily.queueFlags & vk::QueueFlagBits::eCompute) &&
            !(queueFamily.queueFlags & vk::QueueFlagBits::eGraphics)) {
            // 优先选择专用计算队列
            if (!queueFamilyIndices_.computeIndex.has_value()) {
                queueFamilyIndices_.computeIndex = i;
                LOG_INFO("Found dedicated compute queue family: {}", i);
            }
        }
        
        // 查找专用传输队列（支持传输但不支持图形和计算）
        if ((queueFamily.queueFlags & vk::QueueFlagBits::eTransfer) &&
            !(queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) &&
            !(queueFamily.queueFlags & vk::QueueFlagBits::eCompute)) {
            // 优先选择专用传输队列
            if (!queueFamilyIndices_.transferIndex.has_value()) {
                queueFamilyIndices_.transferIndex = i;
            }
        }
    }
    
    // 检查扩展支持
    uint32_t extension_count = 0;
    auto result = device.enumerateDeviceExtensionProperties(nullptr, &extension_count, nullptr);
    if (result != vk::Result::eSuccess) {
        LOG_ERROR("Failed to enumerate device extension properties data");
        return false;
    }
    std::vector<vk::ExtensionProperties> available_extensions(extension_count);
    result = device.enumerateDeviceExtensionProperties(nullptr, &extension_count, available_extensions.data());
    if (result != vk::Result::eSuccess) {
        LOG_ERROR("Failed to enumerate device extension properties");
        return false;
    }

    std::set<std::string> required_extensions;
    required_extensions.insert(vk::KHRSwapchainExtensionName);
    
    for (const auto& extension : available_extensions) {
        required_extensions.erase(extension.extensionName);
    }
    bool extensions_supported = required_extensions.empty();
    
    // 检查交换链支持
    bool swapchain_adequate = false;
    if (extensions_supported) {
        auto formats = device.getSurfaceFormatsKHR(surface);
        auto present_modes = device.getSurfacePresentModesKHR(surface);
        swapchain_adequate = !formats.empty() && !present_modes.empty();
    }
    
    // 检查特性支持
    vk::PhysicalDeviceFeatures supported_features = device.getFeatures();
    
    return queueFamilyIndices_ && 
           extensions_supported && swapchain_adequate && 
           supported_features.samplerAnisotropy;
}

void Device::cleanup() {
    if (device_) {
        // 等待所有GPU操作完成
        device_.waitIdle();
        
        // 销毁逻辑设备（会自动清理所有子资源）
        device_.destroy();
        device_ = nullptr;
    }
    phyDevice_ = nullptr;
    graphicsQueue_ = nullptr;
    presentQueue_ = nullptr;
    transferQueue_ = nullptr;
    computeQueue_ = nullptr;
}

vk::CommandPool Device::createCommandPool(uint32_t queueFamilyIndex, vk::CommandPoolCreateFlags flags) {
    vk::CommandPoolCreateInfo poolInfo{};
    poolInfo.setFlags(flags)
            .setQueueFamilyIndex(queueFamilyIndex);
    
    return device_.createCommandPool(poolInfo);
}

void Device::destroyCommandPool(vk::CommandPool commandPool) {
    if (commandPool) {
        device_.destroyCommandPool(commandPool);
    }
}

} // namespace vk_gs
