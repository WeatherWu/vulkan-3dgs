#include "device.hpp"
#include "context.hpp"
#include "utils/logger.hpp"
#include "vulkan/command_pool.hpp"

#include <cstdlib>
#include <cstring>
#include <set>

namespace vulkan3DGS {

namespace {

const char* deviceTypeName(vk::PhysicalDeviceType type) {
    switch (type) {
        case vk::PhysicalDeviceType::eDiscreteGpu:
            return "Discrete GPU";
        case vk::PhysicalDeviceType::eIntegratedGpu:
            return "Integrated GPU";
        case vk::PhysicalDeviceType::eVirtualGpu:
            return "Virtual GPU";
        case vk::PhysicalDeviceType::eCpu:
            return "CPU";
        default:
            return "Other";
    }
}

bool allowCpuVulkanDevice() {
#ifdef _WIN32
    char* value = nullptr;
    size_t valueLength = 0;
    if (_dupenv_s(&value, &valueLength, "VULKAN_3DGS_ALLOW_CPU_VULKAN") != 0 || value == nullptr) {
        return false;
    }
    bool enabled = std::string(value) == "1";
    std::free(value);
    return enabled;
#else
    const char* value = std::getenv("VULKAN_3DGS_ALLOW_CPU_VULKAN");
    return value && std::string(value) == "1";
#endif
}

int deviceScore(vk::PhysicalDeviceType type) {
    switch (type) {
        case vk::PhysicalDeviceType::eDiscreteGpu:
            return 1000;
        case vk::PhysicalDeviceType::eIntegratedGpu:
            return 500;
        case vk::PhysicalDeviceType::eVirtualGpu:
            return 100;
        case vk::PhysicalDeviceType::eCpu:
            return allowCpuVulkanDevice() ? -100 : -10000;
        default:
            return 0;
    }
}

} // namespace

Device::Device(vk::SurfaceKHR surface) : surface_(surface) {}

Device::~Device() {
    cleanup();
}
    
void Device::createDevice() {
    pickPhysicalDevice(surface_);
    
    vk::DeviceCreateInfo createInfo;
    std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
    std::vector<const char*> enabledExtensions = deviceExtensions_;
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

    // 配置设备特性。计算相关扩展按支持情况启用，不作为全局设备筛选条件。
    vk::PhysicalDeviceFeatures deviceFeatures{};
    deviceFeatures.setSamplerAnisotropy(VK_TRUE);

    const auto availableExtensions = phyDevice_.enumerateDeviceExtensionProperties();
    auto supportsExtension = [&availableExtensions](const char* name) {
        return std::any_of(availableExtensions.begin(), availableExtensions.end(),
                           [name](const vk::ExtensionProperties& extension) {
                               return std::strcmp(extension.extensionName.data(), name) == 0;
                           });
    };

    const bool supportsAtomicFloatExtension = supportsExtension(vk::EXTShaderAtomicFloatExtensionName);
    const bool supportsAtomicFloat2Extension = supportsExtension(vk::EXTShaderAtomicFloat2ExtensionName);
    const bool supportsMemoryBudgetExtension = supportsExtension(vk::EXTMemoryBudgetExtensionName);

    if (supportsMemoryBudgetExtension) {
        enabledExtensions.push_back(vk::EXTMemoryBudgetExtensionName);
        LOG_INFO("Enabled optional device extension: {}", vk::EXTMemoryBudgetExtensionName);
    }

    vk::PhysicalDeviceVulkan11Features supportedVulkan11Features{};
    vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT supportedAtomicFloatFeatures{};
    vk::PhysicalDeviceShaderAtomicFloat2FeaturesEXT supportedAtomicFloat2Features{};
    vk::PhysicalDeviceTimelineSemaphoreFeatures supportedTimelineSemaphoreFeatures{};
    vk::PhysicalDeviceFeatures2 supportedFeatures2{};
    supportedFeatures2.setPNext(&supportedVulkan11Features);
    vk::BaseOutStructure* supportedFeatureTail = reinterpret_cast<vk::BaseOutStructure*>(&supportedVulkan11Features);
    auto appendSupportedFeature = [&supportedFeatureTail](auto& feature) {
        supportedFeatureTail->pNext = reinterpret_cast<vk::BaseOutStructure*>(&feature);
        supportedFeatureTail = reinterpret_cast<vk::BaseOutStructure*>(&feature);
    };
    if (supportsAtomicFloatExtension) {
        appendSupportedFeature(supportedAtomicFloatFeatures);
        if (supportsAtomicFloat2Extension) {
            appendSupportedFeature(supportedAtomicFloat2Features);
        }
    } else if (supportsAtomicFloat2Extension) {
        appendSupportedFeature(supportedAtomicFloat2Features);
    }
    appendSupportedFeature(supportedTimelineSemaphoreFeatures);
    phyDevice_.getFeatures2(&supportedFeatures2);

    vk::PhysicalDeviceVulkan11Features vulkan11Features{};
    vulkan11Features.setShaderDrawParameters(VK_TRUE);
    vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT atomicFloatFeatures{};
    vk::PhysicalDeviceShaderAtomicFloat2FeaturesEXT atomicFloat2Features{};
    vk::PhysicalDeviceTimelineSemaphoreFeatures timelineSemaphoreFeatures{};

    vk::BaseOutStructure* featureTail = reinterpret_cast<vk::BaseOutStructure*>(&vulkan11Features);
    auto appendFeature = [&featureTail](auto& feature) {
        featureTail->pNext = reinterpret_cast<vk::BaseOutStructure*>(&feature);
        featureTail = reinterpret_cast<vk::BaseOutStructure*>(&feature);
    };

    if (supportsAtomicFloatExtension &&
        supportedAtomicFloatFeatures.shaderBufferFloat32Atomics &&
        supportedAtomicFloatFeatures.shaderBufferFloat32AtomicAdd) {
        enabledExtensions.push_back(vk::EXTShaderAtomicFloatExtensionName);
        atomicFloatFeatures.setShaderBufferFloat32Atomics(VK_TRUE)
                           .setShaderBufferFloat32AtomicAdd(VK_TRUE);
        appendFeature(atomicFloatFeatures);
        LOG_INFO("Enabled optional device extension: {}", vk::EXTShaderAtomicFloatExtensionName);
    } else {
        LOG_WARN("Optional device extension {} not enabled; training shaders that require float atomic add may be unavailable",
                 vk::EXTShaderAtomicFloatExtensionName);
    }

    if (supportsAtomicFloat2Extension &&
        supportedAtomicFloat2Features.shaderBufferFloat32AtomicMinMax) {
        enabledExtensions.push_back(vk::EXTShaderAtomicFloat2ExtensionName);
        atomicFloat2Features.setShaderBufferFloat32AtomicMinMax(VK_TRUE);
        appendFeature(atomicFloat2Features);
        LOG_INFO("Enabled optional device extension: {}", vk::EXTShaderAtomicFloat2ExtensionName);
    } else {
        LOG_DEBUG("Optional device extension {} not enabled", vk::EXTShaderAtomicFloat2ExtensionName);
    }

    if (supportedTimelineSemaphoreFeatures.timelineSemaphore) {
        timelineSemaphoreFeatures.setTimelineSemaphore(VK_TRUE);
        appendFeature(timelineSemaphoreFeatures);
        LOG_INFO("Enabled timeline semaphores for asynchronous resource uploads");
    } else {
        LOG_WARN("Timeline semaphores are unavailable; resource uploads will use the synchronous fallback");
    }

    createInfo.setPEnabledFeatures(&deviceFeatures);
    createInfo.setPNext(&vulkan11Features);
    createInfo.setPEnabledExtensionNames(enabledExtensions);

    device_ = phyDevice_.createDevice(createInfo);
    if (!device_) {
        LOG_ERROR("Failed to create Vulkan device");
        throw std::runtime_error("Failed to create Vulkan device");
    }
    
    // 第三步：使用Device更新调度器以获取Device级别的函数
    VULKAN_HPP_DEFAULT_DISPATCHER.init(device_);
    LOG_DEBUG("Vulkan dispatcher updated with device");
    
    // 第四步：获取队列句柄
    graphicsQueue_ = device_.getQueue(queueFamilyIndices_.graphicsIndex.value(), 0);
    presentQueue_ = device_.getQueue(queueFamilyIndices_.presentIndex.value(), 0);
    
    if (queueFamilyIndices_.transferIndex.has_value()) {
        transferQueue_ = device_.getQueue(queueFamilyIndices_.transferIndex.value(), 0);
        LOG_DEBUG("Dedicated transfer queue retrieved (family {})", queueFamilyIndices_.transferIndex.value());
    } else {
        // 如果没有专用传输队列，回退到图形队列
        transferQueue_ = graphicsQueue_;
        LOG_DEBUG("Using graphics queue as transfer queue");
    }
    
    if (queueFamilyIndices_.computeIndex.has_value()) {
        computeQueue_ = device_.getQueue(queueFamilyIndices_.computeIndex.value(), 0);
        LOG_DEBUG("Dedicated compute queue retrieved (family {})", queueFamilyIndices_.computeIndex.value());
    } else {
        // 如果没有专用计算队列，回退到图形队列
        computeQueue_ = graphicsQueue_;
        LOG_DEBUG("Using graphics queue as compute queue");
    }
    
    LOG_INFO("Vulkan device created successfully");
}


void Device::pickPhysicalDevice(vk::SurfaceKHR& surface) {
    auto& context = Context::Instance();
    auto devices = context.getInstance().enumeratePhysicalDevices();
    
    LOG_DEBUG("Found {} physical device(s)", devices.size());
    
    if (devices.empty()) {
        LOG_ERROR("Failed to find GPUs with Vulkan support");
        throw std::runtime_error("Failed to find GPUs with Vulkan support");
    }
    
    int bestScore = -10001;
    QueueFamilyIndices bestQueueFamilyIndices;
    size_t bestDeviceIndex = 0;
    
    for (size_t i = 0; i < devices.size(); ++i) {
        auto properties = devices[i].getProperties();
        LOG_DEBUG("Checking physical device {}: {} ({})", i, properties.deviceName, deviceTypeName(properties.deviceType));
        
        if (properties.deviceType == vk::PhysicalDeviceType::eCpu && !allowCpuVulkanDevice()) {
            LOG_WARN("Skipping CPU Vulkan device {}. Set VULKAN_3DGS_ALLOW_CPU_VULKAN=1 to allow software fallback.", properties.deviceName);
            continue;
        }
        
        QueueFamilyIndices candidateQueueFamilyIndices;
        if (isDeviceSuitable(devices[i], surface, candidateQueueFamilyIndices)) {
            int score = deviceScore(properties.deviceType);
            LOG_DEBUG("Physical device {} is suitable with score {}", properties.deviceName, score);
            
            if (score > bestScore) {
                phyDevice_ = devices[i];
                bestQueueFamilyIndices = candidateQueueFamilyIndices;
                bestScore = score;
                bestDeviceIndex = i;
            }
        }
    }
    
    if (phyDevice_ == VK_NULL_HANDLE) {
        LOG_ERROR("Failed to find a suitable GPU");
        throw std::runtime_error("Failed to find a suitable GPU");
    }
    
    vk::PhysicalDeviceProperties device_properties = phyDevice_.getProperties();
    queueFamilyIndices_ = bestQueueFamilyIndices;
    LOG_DEBUG("Selected device {} with score {}", bestDeviceIndex, bestScore);
    LOG_INFO("Selected physical device: {} ({})", device_properties.deviceName, deviceTypeName(device_properties.deviceType));
}

bool Device::isDeviceSuitable(vk::PhysicalDevice device, vk::SurfaceKHR surface, QueueFamilyIndices& queueFamilyIndices) {
    auto properties = device.getProperties();
    
    // 查找队列家族
    uint32_t queue_family_count = 0;
    device.getQueueFamilyProperties(&queue_family_count, nullptr);
    std::vector<vk::QueueFamilyProperties> queue_families(queue_family_count);
    device.getQueueFamilyProperties(&queue_family_count, queue_families.data());
    
    for (uint32_t i = 0; i < queue_family_count; i++) {
        const auto& queueFamily = queue_families[i];
        
        // 查找图形队列
        if (queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) {
            queueFamilyIndices.graphicsIndex = i;
        }
        
        // 查找呈现队列
        vk::Bool32 present_support = false;
        vk::Result result = device.getSurfaceSupportKHR(i, surface, &present_support);
        if (result == vk::Result::eSuccess && present_support) {
            queueFamilyIndices.presentIndex = i;
        }

        // 查找专用计算队列（支持计算但不支持图形）
        if ((queueFamily.queueFlags & vk::QueueFlagBits::eCompute) &&
            !(queueFamily.queueFlags & vk::QueueFlagBits::eGraphics)) {
            // 优先选择专用计算队列
            if (!queueFamilyIndices.computeIndex.has_value()) {
                queueFamilyIndices.computeIndex = i;
                LOG_DEBUG("Found dedicated compute queue family: {}", i);
            }
        }
        
        // 查找专用传输队列（支持传输但不支持图形和计算）
        if ((queueFamily.queueFlags & vk::QueueFlagBits::eTransfer) &&
            !(queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) &&
            !(queueFamily.queueFlags & vk::QueueFlagBits::eCompute)) {
            // 优先选择专用传输队列
            if (!queueFamilyIndices.transferIndex.has_value()) {
                queueFamilyIndices.transferIndex = i;
            }
        }
    }
    
    // 检查扩展支持
    uint32_t extension_count = 0;
    auto result = device.enumerateDeviceExtensionProperties(nullptr, &extension_count, nullptr);
    if (result != vk::Result::eSuccess) {
        LOG_WARN("Device {} rejected: failed to enumerate extension property count", properties.deviceName);
        return false;
    }
    std::vector<vk::ExtensionProperties> available_extensions(extension_count);
    result = device.enumerateDeviceExtensionProperties(nullptr, &extension_count, available_extensions.data());
    if (result != vk::Result::eSuccess) {
        LOG_WARN("Device {} rejected: failed to enumerate extension properties", properties.deviceName);
        return false;
    }

    std::set<std::string> required_extensions;
    for (const char* extension : deviceExtensions_) {
        required_extensions.insert(extension);
    }
    
    for (const auto& extension : available_extensions) {
        required_extensions.erase(extension.extensionName);
    }
    bool extensions_supported = required_extensions.empty();
    if (!extensions_supported) {
        for (const auto& extension : required_extensions) {
            LOG_DEBUG("Device {} missing extension: {}", properties.deviceName, extension);
        }
    }
    
    // 检查交换链支持
    bool swapchain_adequate = false;
    if (extensions_supported && queueFamilyIndices.presentIndex.has_value()) {
        auto formats = device.getSurfaceFormatsKHR(surface);
        auto present_modes = device.getSurfacePresentModesKHR(surface);
        swapchain_adequate = !formats.empty() && !present_modes.empty();
    }
    
    // 检查特性支持
    vk::PhysicalDeviceFeatures supported_features = device.getFeatures();
    vk::PhysicalDeviceVulkan11Features vulkan11Features{};
    vk::PhysicalDeviceFeatures2 features2{};
    features2.setPNext(&vulkan11Features);
    device.getFeatures2(&features2);
    
    bool featuresSupported = supported_features.samplerAnisotropy &&
                             vulkan11Features.shaderDrawParameters;
    
    if (!queueFamilyIndices) {
        LOG_DEBUG("Device {} rejected: missing graphics or present queue", properties.deviceName);
    }
    if (extensions_supported && queueFamilyIndices.presentIndex.has_value() && !swapchain_adequate) {
        LOG_DEBUG("Device {} rejected: inadequate swapchain support", properties.deviceName);
    }
    if (!featuresSupported) {
        LOG_DEBUG("Device {} rejected: missing required Vulkan features", properties.deviceName);
    }
    
    return queueFamilyIndices &&
           extensions_supported && swapchain_adequate &&
           featuresSupported;
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

} // namespace vulkan3DGS
