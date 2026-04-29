#pragma once

#include <vulkan/vulkan.hpp>
#include <cstddef>
#include "vulkan/command_pool.hpp"

namespace vk_gs {

class Buffer {
public:
    Buffer() = default;
    ~Buffer();

    // 禁止拷贝
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    // 允许移动
    Buffer(Buffer&& other) noexcept;
    Buffer& operator=(Buffer&& other) noexcept;

    // 创建缓冲区（支持 Staging 上传）
    void create(vk::Device device, 
                vk::PhysicalDevice physicalDevice,
                vk::Queue transferQueue,
                uint32_t transferQueueFamilyIndex,
                const void* data,
                vk::DeviceSize size,
                vk::BufferUsageFlags usage,
                vk::MemoryPropertyFlags properties);

    // 清理资源
    void cleanup();

    // 获取底层对象
    vk::Buffer getBuffer() const { return buffer_; }
    vk::DeviceMemory getMemory() const { return memory_; }
    vk::DescriptorBufferInfo getDescriptorInfo() const;

private:
    vk::Device device_ = nullptr;
    vk::PhysicalDevice physicalDevice_ = nullptr;
    vk::Queue transferQueue_ = nullptr;
    uint32_t transferQueueFamilyIndex_ = 0;
    vk::Buffer buffer_ = nullptr;
    vk::DeviceMemory memory_ = nullptr;
    vk::DeviceSize size_ = 0;

    // 辅助函数：查找合适的内存类型
    static uint32_t findMemoryType(vk::PhysicalDevice physicalDevice, 
                                   uint32_t typeFilter, 
                                   vk::MemoryPropertyFlags properties);
    
    // 内部方法：通过 Staging Buffer 上传数据
    void uploadData(const void* data, vk::DeviceSize size);
};

} // namespace vk_gs
