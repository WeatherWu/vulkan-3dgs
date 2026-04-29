#include "command_pool.hpp"
#include <stdexcept>

namespace vk_gs {

CommandPool::~CommandPool() {
    cleanup();
}

CommandPool::CommandPool(CommandPool&& other) noexcept
    : device_(other.device_), pool_(other.pool_) {
    other.device_ = nullptr;
    other.pool_ = nullptr;
}

CommandPool& CommandPool::operator=(CommandPool&& other) noexcept {
    if (this != &other) {
        cleanup();
        device_ = other.device_;
        pool_ = other.pool_;
        other.device_ = nullptr;
        other.pool_ = nullptr;
    }
    return *this;
}

void CommandPool::create(vk::Device device, uint32_t queueFamilyIndex, vk::CommandPoolCreateFlags flags) {
    device_ = device;
    vk::CommandPoolCreateInfo poolInfo{};
    poolInfo.setQueueFamilyIndex(queueFamilyIndex)
            .setFlags(flags);
    
    pool_ = device_.createCommandPool(poolInfo);
}

void CommandPool::cleanup() {
    if (pool_) {
        device_.destroyCommandPool(pool_);
        pool_ = nullptr;
        device_ = nullptr;
    }
}

vk::CommandBuffer CommandPool::allocateCommandBuffer(vk::CommandBufferLevel level) {
    vk::CommandBufferAllocateInfo allocInfo{};
    allocInfo.setCommandPool(pool_)
             .setLevel(level)
             .setCommandBufferCount(1);
    
    auto buffers = device_.allocateCommandBuffers(allocInfo);
    return buffers.front();
}

void CommandPool::freeCommandBuffer(vk::CommandBuffer commandBuffer) {
    if (pool_) {
        device_.freeCommandBuffers(pool_, commandBuffer);
    }
}

void CommandPool::reset() {
    if (pool_) {
        device_.resetCommandPool(pool_);
    }
}

} // namespace vk_gs
