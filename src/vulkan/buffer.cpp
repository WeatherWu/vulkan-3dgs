#include "buffer.hpp"
#include "context/context.hpp"
#include "vulkan/command_pool.hpp"
#include "utils/logger.hpp"
#include <cstring>
#include <stdexcept>
#include <array>
#include <algorithm>

namespace vulkan3DGS {

Buffer::~Buffer() {
    cleanup();
}

Buffer::Buffer(Buffer&& other) noexcept
    : device_(other.device_),
      physicalDevice_(other.physicalDevice_),
      transferQueue_(other.transferQueue_),
      transferQueueFamilyIndex_(other.transferQueueFamilyIndex_),
      buffer_(other.buffer_),
      memory_(other.memory_),
      size_(other.size_),
      properties_(other.properties_) {
    other.device_ = nullptr;
    other.physicalDevice_ = nullptr;
    other.transferQueue_ = nullptr;
    other.transferQueueFamilyIndex_ = 0;
    other.buffer_ = nullptr;
    other.memory_ = nullptr;
    other.size_ = 0;
    other.properties_ = {};
}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
    if (this != &other) {
        cleanup();
        device_ = other.device_;
        physicalDevice_ = other.physicalDevice_;
        transferQueue_ = other.transferQueue_;
        transferQueueFamilyIndex_ = other.transferQueueFamilyIndex_;
        buffer_ = other.buffer_;
        memory_ = other.memory_;
        size_ = other.size_;
        properties_ = other.properties_;
        other.device_ = nullptr;
        other.physicalDevice_ = nullptr;
        other.transferQueue_ = nullptr;
        other.transferQueueFamilyIndex_ = 0;
        other.buffer_ = nullptr;
        other.memory_ = nullptr;
        other.size_ = 0;
        other.properties_ = {};
    }
    return *this;
}

void Buffer::create(vk::Device device, 
                    vk::PhysicalDevice physicalDevice,
                    vk::Queue transferQueue,
                    uint32_t transferQueueFamilyIndex,
                    const void* data,
                    vk::DeviceSize size,
                    vk::BufferUsageFlags usage,
                    vk::MemoryPropertyFlags properties) {
    LOG_DEBUG("Buffer::create - Starting buffer creation (size: {} bytes)", size);
    
    device_ = device;
    physicalDevice_ = physicalDevice;
    transferQueue_ = transferQueue;
    transferQueueFamilyIndex_ = transferQueueFamilyIndex;
    size_ = size;
    properties_ = properties;

    // 1. 创建目标缓冲区
    LOG_DEBUG("Buffer::create - Creating Vulkan buffer");
    std::array<uint32_t, 3> queueFamilies = {
        transferQueueFamilyIndex_,
        Context::Instance().getDevice().getQueueFamilyIndices().graphicsIndex.value(),
        Context::Instance().getDevice().getQueueFamilyIndices().computeIndex.value_or(
            Context::Instance().getDevice().getQueueFamilyIndices().graphicsIndex.value())
    };
    std::sort(queueFamilies.begin(), queueFamilies.end());
    auto uniqueEnd = std::unique(queueFamilies.begin(), queueFamilies.end());
    uint32_t queueFamilyCount = static_cast<uint32_t>(std::distance(queueFamilies.begin(), uniqueEnd));

    vk::BufferCreateInfo bufferInfo{};
    bufferInfo.setSize(size)
                .setUsage(usage | vk::BufferUsageFlagBits::eTransferDst)
                .setSharingMode(queueFamilyCount > 1 ? vk::SharingMode::eConcurrent : vk::SharingMode::eExclusive)
                .setQueueFamilyIndexCount(queueFamilyCount > 1 ? queueFamilyCount : 0)
                .setPQueueFamilyIndices(queueFamilyCount > 1 ? queueFamilies.data() : nullptr);

    buffer_ = device_.createBuffer(bufferInfo);
    LOG_DEBUG("Buffer::create - Vulkan buffer created");

    // 2. 分配内存
    LOG_DEBUG("Buffer::create - Getting memory requirements");
    vk::MemoryRequirements memRequirements = device_.getBufferMemoryRequirements(buffer_);
    
    LOG_DEBUG("Buffer::create - Finding suitable memory type");
    uint32_t memoryTypeIndex = findMemoryType(physicalDevice_, memRequirements.memoryTypeBits, properties);
    LOG_DEBUG("Buffer::create - Found memory type index: {}", memoryTypeIndex);
    
    vk::MemoryAllocateInfo allocInfo{};
    allocInfo.setAllocationSize(memRequirements.size)
             .setMemoryTypeIndex(memoryTypeIndex);

    LOG_DEBUG("Buffer::create - Allocating device memory");
    memory_ = device_.allocateMemory(allocInfo);
    LOG_DEBUG("Buffer::create - Memory allocated successfully");

    LOG_DEBUG("Buffer::create - Binding buffer to memory");
    device_.bindBufferMemory(buffer_, memory_, 0);
    LOG_DEBUG("Buffer::create - Buffer bound to memory");

    // 3. 如果有初始数据，进行上传
    if (data != nullptr) {
        LOG_DEBUG("Buffer::create - Uploading initial data ({} bytes)", size);
        uploadData(data, size);
        LOG_DEBUG("Buffer::create - Data upload completed");
    }
    
    LOG_DEBUG("Buffer::create - Buffer creation completed successfully");
}

void Buffer::cleanup() {
    if (buffer_) {
        device_.destroyBuffer(buffer_);
        buffer_ = nullptr;
    }
    if (memory_) {
        device_.freeMemory(memory_);
        memory_ = nullptr;
    }
    device_ = nullptr;
    physicalDevice_ = nullptr;
    transferQueue_ = nullptr;
    size_ = 0;
    properties_ = {};
}

void Buffer::upload(const void* data, vk::DeviceSize size, vk::DeviceSize offset) {
    if (!buffer_ || !memory_) {
        throw std::runtime_error("Cannot upload to an uninitialized buffer");
    }
    if (!data) {
        throw std::runtime_error("Cannot upload null data to buffer");
    }
    if (offset > size_ || size > size_ - offset) {
        throw std::runtime_error("Upload size exceeds buffer capacity");
    }

    if ((properties_ & vk::MemoryPropertyFlagBits::eHostVisible) == vk::MemoryPropertyFlagBits::eHostVisible) {
        void* mappedMemory = device_.mapMemory(memory_, 0, VK_WHOLE_SIZE);
        std::memcpy(static_cast<std::byte*>(mappedMemory) + offset, data, static_cast<size_t>(size));
        if ((properties_ & vk::MemoryPropertyFlagBits::eHostCoherent) != vk::MemoryPropertyFlagBits::eHostCoherent) {
            vk::MappedMemoryRange range{};
            range.setMemory(memory_).setOffset(0).setSize(VK_WHOLE_SIZE);
            device_.flushMappedMemoryRanges(range);
        }
        device_.unmapMemory(memory_);
        return;
    }

    uploadData(data, size, offset);
}

void Buffer::download(void* data, vk::DeviceSize size) {
    if (!buffer_ || !memory_) {
        throw std::runtime_error("Cannot download from an uninitialized buffer");
    }
    if (!data) {
        throw std::runtime_error("Cannot download to null data");
    }
    if (size > size_) {
        throw std::runtime_error("Download size exceeds buffer capacity");
    }

    if ((properties_ & vk::MemoryPropertyFlagBits::eHostVisible) == vk::MemoryPropertyFlagBits::eHostVisible) {
        void* mappedMemory = device_.mapMemory(memory_, 0, size);
        std::memcpy(data, mappedMemory, static_cast<size_t>(size));
        device_.unmapMemory(memory_);
        return;
    }

    CommandPool cmdPool;
    cmdPool.create(device_, transferQueueFamilyIndex_, vk::CommandPoolCreateFlagBits::eTransient);

    vk::Buffer stagingBuffer;
    vk::DeviceMemory stagingMemory;

    vk::BufferCreateInfo bufferInfo{};
    bufferInfo.setSize(size)
              .setUsage(vk::BufferUsageFlagBits::eTransferDst)
              .setSharingMode(vk::SharingMode::eExclusive);
    stagingBuffer = device_.createBuffer(bufferInfo);

    vk::MemoryRequirements memRequirements = device_.getBufferMemoryRequirements(stagingBuffer);
    vk::MemoryAllocateInfo allocInfo{};
    allocInfo.setAllocationSize(memRequirements.size)
             .setMemoryTypeIndex(findMemoryType(physicalDevice_,
                                                memRequirements.memoryTypeBits,
                                                vk::MemoryPropertyFlagBits::eHostVisible |
                                                    vk::MemoryPropertyFlagBits::eHostCoherent));
    stagingMemory = device_.allocateMemory(allocInfo);
    device_.bindBufferMemory(stagingBuffer, stagingMemory, 0);

    vk::CommandBuffer commandBuffer = cmdPool.allocateCommandBuffer();
    vk::CommandBufferBeginInfo beginInfo{};
    beginInfo.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    commandBuffer.begin(beginInfo);

    vk::BufferCopy copyRegion{};
    copyRegion.setSize(size);
    commandBuffer.copyBuffer(buffer_, stagingBuffer, copyRegion);
    commandBuffer.end();

    vk::SubmitInfo submitInfo{};
    submitInfo.setCommandBufferCount(1)
              .setPCommandBuffers(&commandBuffer);
    transferQueue_.submit(submitInfo);
    transferQueue_.waitIdle();

    void* mappedMemory = device_.mapMemory(stagingMemory, 0, size);
    std::memcpy(data, mappedMemory, static_cast<size_t>(size));
    device_.unmapMemory(stagingMemory);

    device_.destroyBuffer(stagingBuffer);
    device_.freeMemory(stagingMemory);
}

vk::DescriptorBufferInfo Buffer::getDescriptorInfo() const {
    vk::DescriptorBufferInfo info{};
    info.setBuffer(buffer_)
        .setOffset(0)
        .setRange(size_);
    return info;
}

void Buffer::uploadData(const void* data, vk::DeviceSize size, vk::DeviceSize offset) {
    LOG_DEBUG("Buffer::uploadData - Starting data upload ({} bytes)", size);
    
    // 使用保存的传输队列家族索引
    uint32_t queueFamilyIndex = transferQueueFamilyIndex_;
    LOG_DEBUG("Buffer::uploadData - Queue family index: {}", queueFamilyIndex);
    
    // 创建临时 CommandPool
    LOG_DEBUG("Buffer::uploadData - Creating temporary command pool");
    CommandPool cmdPool;
    cmdPool.create(device_, queueFamilyIndex, vk::CommandPoolCreateFlagBits::eTransient);
    LOG_DEBUG("Buffer::uploadData - Command pool created");

    // 创建 Staging Buffer
    LOG_DEBUG("Buffer::uploadData - Creating staging buffer");
    vk::Buffer stagingBuffer;
    vk::DeviceMemory stagingMemory;

    vk::BufferCreateInfo bufferInfo{};
    bufferInfo.setSize(size)
                .setUsage(vk::BufferUsageFlagBits::eTransferSrc)
                .setSharingMode(vk::SharingMode::eExclusive);

    stagingBuffer = device_.createBuffer(bufferInfo);
    LOG_DEBUG("Buffer::uploadData - Staging buffer created");

    LOG_DEBUG("Buffer::uploadData - Getting staging buffer memory requirements");
    vk::MemoryRequirements memRequirements = device_.getBufferMemoryRequirements(stagingBuffer);
    LOG_DEBUG("Buffer::uploadData - Finding staging buffer memory type");
    vk::MemoryAllocateInfo allocInfo{};
    allocInfo.setAllocationSize(memRequirements.size)
             .setMemoryTypeIndex(findMemoryType(physicalDevice_, 
                                                memRequirements.memoryTypeBits,
                                                vk::MemoryPropertyFlagBits::eHostVisible | 
                                                vk::MemoryPropertyFlagBits::eHostCoherent));
    LOG_DEBUG("Buffer::uploadData - Allocating staging memory");

    stagingMemory = device_.allocateMemory(allocInfo);
    LOG_DEBUG("Buffer::uploadData - Staging memory allocated");
    
    device_.bindBufferMemory(stagingBuffer, stagingMemory, 0);
    LOG_DEBUG("Buffer::uploadData - Staging buffer bound to memory");

    // 映射并拷贝数据
    LOG_DEBUG("Buffer::uploadData - Mapping staging memory");
    void* mappedMemory = device_.mapMemory(stagingMemory, 0, size);
    LOG_DEBUG("Buffer::uploadData - Copying {} bytes to staging buffer", size);
    std::memcpy(mappedMemory, data, static_cast<size_t>(size));
    LOG_DEBUG("Buffer::uploadData - Unmapping staging memory");
    device_.unmapMemory(stagingMemory);
    LOG_DEBUG("Buffer::uploadData - Data copied to staging buffer");

    // 记录拷贝命令
    LOG_DEBUG("Buffer::uploadData - Allocating command buffer");
    vk::CommandBuffer commandBuffer = cmdPool.allocateCommandBuffer();

    vk::CommandBufferBeginInfo beginInfo{};
    beginInfo.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    LOG_DEBUG("Buffer::uploadData - Beginning command buffer recording");
    commandBuffer.begin(beginInfo);

    LOG_DEBUG("Buffer::uploadData - Recording copy command");
    vk::BufferCopy copyRegion{};
    copyRegion.setDstOffset(offset)
              .setSize(size);
    commandBuffer.copyBuffer(stagingBuffer, buffer_, copyRegion);

    LOG_DEBUG("Buffer::uploadData - Ending command buffer recording");
    commandBuffer.end();

    LOG_DEBUG("Buffer::uploadData - Submitting copy command to transfer queue");
    vk::SubmitInfo submitInfo{};
    submitInfo.setCommandBufferCount(1)
              .setPCommandBuffers(&commandBuffer);

    transferQueue_.submit(submitInfo);
    LOG_DEBUG("Buffer::uploadData - Waiting for transfer queue to complete");
    transferQueue_.waitIdle();
    LOG_DEBUG("Buffer::uploadData - Transfer completed");

    // 清理 Staging 资源
    LOG_DEBUG("Buffer::uploadData - Cleaning up staging resources");
    device_.destroyBuffer(stagingBuffer);
    device_.freeMemory(stagingMemory);
    LOG_DEBUG("Buffer::uploadData - Data upload completed successfully");
}

uint32_t Buffer::findMemoryType(vk::PhysicalDevice physicalDevice, 
                                uint32_t typeFilter, 
                                vk::MemoryPropertyFlags properties) {
    vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) &&
            (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw std::runtime_error("Failed to find suitable memory type");
}

} // namespace vulkan3DGS
