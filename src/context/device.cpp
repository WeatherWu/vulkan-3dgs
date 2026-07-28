#include "device.hpp"
#include "context.hpp"
#include "device_selection.hpp"
#include "utils/logger.hpp"
#include "vulkan/command_pool.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <set>
#include <sstream>
#include <utility>

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

std::string formatVulkanVersion(uint32_t version) {
    std::ostringstream stream;
    stream << VK_API_VERSION_MAJOR(version) << '.'
           << VK_API_VERSION_MINOR(version) << '.'
           << VK_API_VERSION_PATCH(version);
    return stream.str();
}

std::string formatDriverVersion(uint32_t version) {
    std::ostringstream stream;
    stream << version << " (0x" << std::hex << std::uppercase << version << ')';
    return stream.str();
}

template <typename Uuid>
std::string formatDeviceUuid(const Uuid& uuid) {
    if (std::all_of(uuid.begin(), uuid.end(), [](uint8_t byte) { return byte == 0; })) {
        return {};
    }

    std::ostringstream stream;
    stream << std::hex << std::setfill('0');
    for (size_t index = 0; index < uuid.size(); ++index) {
        if (index == 4 || index == 6 || index == 8 || index == 10) {
            stream << '-';
        }
        stream << std::setw(2) << static_cast<unsigned int>(uuid[index]);
    }
    return stream.str();
}

uint64_t largestDeviceLocalHeap(vk::PhysicalDevice device) {
    const vk::PhysicalDeviceMemoryProperties memory = device.getMemoryProperties();
    uint64_t largestHeap = 0;
    for (uint32_t index = 0; index < memory.memoryHeapCount; ++index) {
        if ((memory.memoryHeaps[index].flags & vk::MemoryHeapFlagBits::eDeviceLocal) !=
            vk::MemoryHeapFlags{}) {
            largestHeap = std::max(largestHeap,
                                   static_cast<uint64_t>(memory.memoryHeaps[index].size));
        }
    }
    return largestHeap;
}

std::string formatMemorySize(uint64_t bytes) {
    constexpr double bytesPerGiB = 1024.0 * 1024.0 * 1024.0;
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(1)
           << static_cast<double>(bytes) / bytesPerGiB << " GiB";
    return stream.str();
}

Device::PhysicalDeviceInfo physicalDeviceInfo(vk::PhysicalDevice device, uint32_t vulkanIndex) {
    vk::PhysicalDeviceIDProperties idProperties{};
    vk::PhysicalDeviceProperties2 properties2{};
    properties2.pNext = &idProperties;
    device.getProperties2(&properties2);

    const vk::PhysicalDeviceProperties& properties = properties2.properties;
    Device::PhysicalDeviceInfo info{};
    info.vulkanIndex = vulkanIndex;
    info.name = properties.deviceName.data();
    info.uuid = formatDeviceUuid(idProperties.deviceUUID);
    info.typeName = deviceTypeName(properties.deviceType);
    info.apiVersion = formatVulkanVersion(properties.apiVersion);
    info.driverVersion = formatDriverVersion(properties.driverVersion);
    info.deviceLocalMemoryBytes = largestDeviceLocalHeap(device);
    return info;
}

std::string suitableDeviceList(const std::vector<Device::PhysicalDeviceInfo>& devices) {
    std::ostringstream stream;
    for (const Device::PhysicalDeviceInfo& device : devices) {
        stream << "\n  [" << device.vulkanIndex << "] " << device.name
               << " (" << device.typeName << ", "
               << formatMemorySize(device.deviceLocalMemoryBytes);
        if (!device.uuid.empty()) {
            stream << ", UUID " << device.uuid;
        }
        stream << ')';
    }
    return stream.str();
}

} // namespace

Device::Device(vk::SurfaceKHR surface,
               std::optional<std::string> gpuSelector,
               DeviceRole role)
    : surface_(surface), gpuSelector_(std::move(gpuSelector)), role_(role) {}

Device::~Device() {
    cleanup();
}
    
void Device::createDevice() {
    pickPhysicalDevice(surface_);
    
    vk::DeviceCreateInfo createInfo;
    std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
    std::vector<const char*> enabledExtensions = role_ == DeviceRole::Training
        ? trainingDeviceExtensions_
        : presentationDeviceExtensions_;
    float properties = 1.0;
    
    if (role_ == DeviceRole::Presentation &&
        (!queueFamilyIndices_.graphicsIndex || !queueFamilyIndices_.presentIndex)) {
        throw std::runtime_error("Presentation queue family indices not set");
    }
    if (role_ == DeviceRole::Training && !queueFamilyIndices_.computeIndex) {
        throw std::runtime_error("Training compute queue family index not set");
    }
    
    // 收集所有需要创建的队列家族索引（去重）
    std::set<uint32_t> uniqueQueueFamilies;
    if (role_ == DeviceRole::Presentation) {
        uniqueQueueFamilies.insert(queueFamilyIndices_.graphicsIndex.value());
        uniqueQueueFamilies.insert(queueFamilyIndices_.presentIndex.value());
    } else {
        uniqueQueueFamilies.insert(queueFamilyIndices_.computeIndex.value());
    }
    
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

    // 显示和训练逻辑设备只启用各自实际使用的核心特性。
    vk::PhysicalDeviceFeatures deviceFeatures{};
    if (role_ == DeviceRole::Presentation) {
        deviceFeatures.setSamplerAnisotropy(VK_TRUE);
    }

    const auto availableExtensions = phyDevice_.enumerateDeviceExtensionProperties();
    auto supportsExtension = [&availableExtensions](const char* name) {
        return std::any_of(availableExtensions.begin(), availableExtensions.end(),
                           [name](const vk::ExtensionProperties& extension) {
                               return std::strcmp(extension.extensionName.data(), name) == 0;
                           });
    };
    auto enableExtension = [&enabledExtensions](const char* name) {
        if (std::find_if(enabledExtensions.begin(), enabledExtensions.end(),
                         [name](const char* enabled) {
                             return std::strcmp(enabled, name) == 0;
                         }) == enabledExtensions.end()) {
            enabledExtensions.push_back(name);
        }
    };

    const bool supportsAtomicFloatExtension = supportsExtension(vk::EXTShaderAtomicFloatExtensionName);
    const bool supportsAtomicFloat2Extension = supportsExtension(vk::EXTShaderAtomicFloat2ExtensionName);
    const bool supportsMemoryBudgetExtension = supportsExtension(vk::EXTMemoryBudgetExtensionName);

    if (supportsMemoryBudgetExtension) {
        enableExtension(vk::EXTMemoryBudgetExtensionName);
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
    if (role_ == DeviceRole::Presentation) {
        vulkan11Features.setShaderDrawParameters(VK_TRUE);
    }
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
        enableExtension(vk::EXTShaderAtomicFloatExtensionName);
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
        enableExtension(vk::EXTShaderAtomicFloat2ExtensionName);
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
    
    // 全局 Vulkan-Hpp dispatcher 始终由 presentation device 持有，避免训练设备
    // 未启用 swapchain 时覆盖显示路径的扩展函数指针。
    if (role_ == DeviceRole::Presentation) {
        VULKAN_HPP_DEFAULT_DISPATCHER.init(device_);
        LOG_DEBUG("Vulkan dispatcher updated with presentation device");
    }
    
    // 第四步：获取队列句柄
    if (role_ == DeviceRole::Presentation) {
        graphicsQueue_ = device_.getQueue(queueFamilyIndices_.graphicsIndex.value(), 0);
        presentQueue_ = device_.getQueue(queueFamilyIndices_.presentIndex.value(), 0);
    }
    
    if (queueFamilyIndices_.computeIndex.has_value()) {
        computeQueue_ = device_.getQueue(queueFamilyIndices_.computeIndex.value(), 0);
        LOG_DEBUG("Dedicated compute queue retrieved (family {})", queueFamilyIndices_.computeIndex.value());
    } else {
        // 如果没有专用计算队列，回退到图形队列
        computeQueue_ = graphicsQueue_;
        LOG_DEBUG("Using graphics queue as compute queue");
    }

    if (queueFamilyIndices_.transferIndex.has_value()) {
        transferQueue_ = device_.getQueue(queueFamilyIndices_.transferIndex.value(), 0);
        LOG_DEBUG("Dedicated transfer queue retrieved (family {})", queueFamilyIndices_.transferIndex.value());
    } else {
        transferQueue_ = role_ == DeviceRole::Training ? computeQueue_ : graphicsQueue_;
        LOG_DEBUG("Using {} queue as transfer queue",
                  role_ == DeviceRole::Training ? "compute" : "graphics");
    }
    
    LOG_INFO("Vulkan {} device created successfully",
             role_ == DeviceRole::Training ? "training" : "presentation");
}


void Device::pickPhysicalDevice(vk::SurfaceKHR& surface) {
    auto& context = Context::Instance();
    auto devices = context.getInstance().enumeratePhysicalDevices();
    
    LOG_DEBUG("Found {} physical device(s) while selecting the {} device",
              devices.size(), role_ == DeviceRole::Training ? "training" : "presentation");
    
    if (devices.empty()) {
        LOG_ERROR("Failed to find GPUs with Vulkan support");
        throw std::runtime_error("Failed to find GPUs with Vulkan support");
    }
    
    struct SuitableDevice {
        vk::PhysicalDevice device;
        QueueFamilyIndices queueFamilyIndices;
        int score = 0;
    };

    availablePhysicalDeviceInfos_.clear();
    std::vector<SuitableDevice> suitableDevices;
    
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

            PhysicalDeviceInfo info = physicalDeviceInfo(devices[i], static_cast<uint32_t>(i));
            LOG_INFO("Suitable {} GPU [{}]: {} ({}), UUID {}, Vulkan {}, driver {}, device-local {}",
                     role_ == DeviceRole::Training ? "training" : "presentation",
                     info.vulkanIndex,
                     info.name,
                     info.typeName,
                     info.uuid.empty() ? "unavailable" : info.uuid,
                     info.apiVersion,
                     info.driverVersion,
                     formatMemorySize(info.deviceLocalMemoryBytes));
            availablePhysicalDeviceInfos_.push_back(std::move(info));
            suitableDevices.push_back({devices[i], candidateQueueFamilyIndices, score});
        }
    }

    if (suitableDevices.empty()) {
        LOG_ERROR("Failed to find a suitable GPU");
        throw std::runtime_error("Failed to find a suitable GPU");
    }

    size_t selectedCandidateIndex = 0;
    if (gpuSelector_.has_value()) {
        std::vector<GpuSelectionCandidate> selectionCandidates;
        selectionCandidates.reserve(availablePhysicalDeviceInfos_.size());
        for (const PhysicalDeviceInfo& info : availablePhysicalDeviceInfos_) {
            selectionCandidates.push_back({info.vulkanIndex, info.name, info.uuid});
        }

        const GpuSelectionResult result = selectGpuCandidate(selectionCandidates, *gpuSelector_);
        if (!result) {
            const std::string error = result.error + ". Suitable GPUs:" +
                                      suitableDeviceList(availablePhysicalDeviceInfos_);
            LOG_ERROR("{}", error);
            throw std::runtime_error(error);
        }
        selectedCandidateIndex = *result.candidateIndex;
    } else {
        for (size_t index = 1; index < suitableDevices.size(); ++index) {
            if (suitableDevices[index].score > suitableDevices[selectedCandidateIndex].score) {
                selectedCandidateIndex = index;
            }
        }
    }

    const SuitableDevice& selectedDevice = suitableDevices[selectedCandidateIndex];
    phyDevice_ = selectedDevice.device;
    queueFamilyIndices_ = selectedDevice.queueFamilyIndices;
    selectedPhysicalDeviceInfo_ = availablePhysicalDeviceInfos_[selectedCandidateIndex];
    LOG_INFO("Selected {} device [{}]: {} ({})",
             role_ == DeviceRole::Training ? "training" : "presentation",
             selectedPhysicalDeviceInfo_.vulkanIndex,
             selectedPhysicalDeviceInfo_.name, selectedPhysicalDeviceInfo_.typeName);
    LOG_INFO("Selected {} GPU details: UUID {}, Vulkan {}, driver {}, device-local {}",
             role_ == DeviceRole::Training ? "training" : "presentation",
             selectedPhysicalDeviceInfo_.uuid.empty() ? "unavailable" : selectedPhysicalDeviceInfo_.uuid,
             selectedPhysicalDeviceInfo_.apiVersion,
             selectedPhysicalDeviceInfo_.driverVersion,
             formatMemorySize(selectedPhysicalDeviceInfo_.deviceLocalMemoryBytes));
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
        
        // 训练设备不需要查询窗口 Surface 的呈现能力。
        if (role_ == DeviceRole::Presentation) {
            vk::Bool32 present_support = false;
            vk::Result result = device.getSurfaceSupportKHR(i, surface, &present_support);
            if (result == vk::Result::eSuccess && present_support) {
                queueFamilyIndices.presentIndex = i;
            }
        }

        // 查找计算队列，优先使用不带 graphics 的专用队列。
        if (queueFamily.queueFlags & vk::QueueFlagBits::eCompute) {
            const bool candidateDedicated =
                !(queueFamily.queueFlags & vk::QueueFlagBits::eGraphics);
            const bool currentDedicated = queueFamilyIndices.computeIndex.has_value() &&
                !(queue_families[*queueFamilyIndices.computeIndex].queueFlags &
                  vk::QueueFlagBits::eGraphics);
            if (!queueFamilyIndices.computeIndex.has_value() ||
                (candidateDedicated && !currentDedicated)) {
                queueFamilyIndices.computeIndex = i;
                if (candidateDedicated) {
                    LOG_DEBUG("Found dedicated compute queue family: {}", i);
                }
            }
        }
        
        // 查找传输队列，优先使用不带 graphics/compute 的专用队列。
        if (queueFamily.queueFlags & vk::QueueFlagBits::eTransfer) {
            const bool candidateDedicated =
                !(queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) &&
                !(queueFamily.queueFlags & vk::QueueFlagBits::eCompute);
            const bool currentDedicated = queueFamilyIndices.transferIndex.has_value() &&
                !(queue_families[*queueFamilyIndices.transferIndex].queueFlags &
                  vk::QueueFlagBits::eGraphics) &&
                !(queue_families[*queueFamilyIndices.transferIndex].queueFlags &
                  vk::QueueFlagBits::eCompute);
            if (!queueFamilyIndices.transferIndex.has_value() ||
                (candidateDedicated && !currentDedicated)) {
                queueFamilyIndices.transferIndex = i;
            }
        }
    }

    if (role_ == DeviceRole::Training && queueFamilyIndices.computeIndex.has_value()) {
        const bool hasDedicatedTransfer = queueFamilyIndices.transferIndex.has_value() &&
            !(queue_families[*queueFamilyIndices.transferIndex].queueFlags &
              vk::QueueFlagBits::eGraphics) &&
            !(queue_families[*queueFamilyIndices.transferIndex].queueFlags &
              vk::QueueFlagBits::eCompute);
        if (!hasDedicatedTransfer) {
            queueFamilyIndices.transferIndex = queueFamilyIndices.computeIndex;
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
    const auto& deviceExtensions = role_ == DeviceRole::Training
        ? trainingDeviceExtensions_
        : presentationDeviceExtensions_;
    for (const char* extension : deviceExtensions) {
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
    bool swapchain_adequate = role_ == DeviceRole::Training;
    if (role_ == DeviceRole::Presentation &&
        extensions_supported && queueFamilyIndices.presentIndex.has_value()) {
        auto formats = device.getSurfaceFormatsKHR(surface);
        auto present_modes = device.getSurfacePresentModesKHR(surface);
        swapchain_adequate = !formats.empty() && !present_modes.empty();
    }
    
    // 检查特性支持
    vk::PhysicalDeviceFeatures supported_features = device.getFeatures();
    vk::PhysicalDeviceVulkan11Features vulkan11Features{};
    vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT atomicFloatFeatures{};
    vk::PhysicalDeviceFeatures2 features2{};
    features2.setPNext(&vulkan11Features);
    vulkan11Features.setPNext(&atomicFloatFeatures);
    device.getFeatures2(&features2);
    
    const bool featuresSupported = role_ == DeviceRole::Training
        ? atomicFloatFeatures.shaderBufferFloat32Atomics &&
              atomicFloatFeatures.shaderBufferFloat32AtomicAdd
        : supported_features.samplerAnisotropy &&
              vulkan11Features.shaderDrawParameters;
    const bool queuesSupported = role_ == DeviceRole::Training
        ? queueFamilyIndices.computeIndex.has_value()
        : static_cast<bool>(queueFamilyIndices);
    
    if (!queuesSupported) {
        LOG_DEBUG("Device {} rejected: missing {} queue",
                  properties.deviceName,
                  role_ == DeviceRole::Training ? "compute" : "graphics or present");
    }
    if (role_ == DeviceRole::Presentation && extensions_supported &&
        queueFamilyIndices.presentIndex.has_value() && !swapchain_adequate) {
        LOG_DEBUG("Device {} rejected: inadequate swapchain support", properties.deviceName);
    }
    if (!featuresSupported) {
        LOG_DEBUG("Device {} rejected: missing required {} Vulkan features",
                  properties.deviceName,
                  role_ == DeviceRole::Training ? "training" : "presentation");
    }
    
    return queuesSupported &&
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
