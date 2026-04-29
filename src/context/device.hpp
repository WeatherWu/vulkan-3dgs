#pragma once

#include <vulkan/vulkan.hpp>
#include <vector>
#include <string>
#include <memory>
#include <optional>

namespace vk_gs {

class Device {
public:
    struct QueueFamilyIndices {
        std::optional<uint32_t> graphicsIndex;
        std::optional<uint32_t> presentIndex;
        std::optional<uint32_t> transferIndex;
        std::optional<uint32_t> computeIndex;
        
        operator bool() const {
            return graphicsIndex && presentIndex;
        }
    };

    Device(vk::SurfaceKHR surface);
    Device() = default;
    ~Device();
    
    void createDevice();
    void pickPhysicalDevice(vk::SurfaceKHR& surface);
    bool isDeviceSuitable(vk::PhysicalDevice device, vk::SurfaceKHR surface);

    void cleanup();

    vk::Device& getDevice() { return device_; }
    vk::PhysicalDevice& getPhysicalDevice() { return phyDevice_; }
    vk::Queue getGraphicsQueue() { return graphicsQueue_; }
    vk::Queue getPresentQueue() { return presentQueue_; }
    vk::Queue getTransferQueue() { return transferQueue_; }
    vk::Queue getComputeQueue() { return computeQueue_; }  // 新增：获取计算队列
    
    const QueueFamilyIndices& getQueueFamilyIndices() const { return queueFamilyIndices_; }
    
    vk::CommandPool createCommandPool(uint32_t queue_family_index, vk::CommandPoolCreateFlags flags = {});
    void destroyCommandPool(vk::CommandPool command_pool);
    
private:
    vk::SurfaceKHR surface_ = nullptr;  // 保存surface供createDevice使用
    vk::PhysicalDevice phyDevice_ = nullptr;
    vk::Device device_ = nullptr;

    vk::Queue graphicsQueue_ = nullptr;
    vk::Queue presentQueue_ = nullptr;
    vk::Queue transferQueue_ = nullptr;
    vk::Queue computeQueue_ = nullptr;  // 新增：计算队列
    QueueFamilyIndices queueFamilyIndices_;
    
    const std::vector<const char*> deviceExtensions_ = {
        vk::KHRSwapchainExtensionName
    };
};

} // namespace vk_gs
