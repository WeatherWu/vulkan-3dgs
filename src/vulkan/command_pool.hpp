#pragma once

#include <vulkan/vulkan.hpp>
#include <vector>

namespace vk_gs {

class CommandPool {
public:
    CommandPool() = default;
    ~CommandPool();

    // 禁止拷贝
    CommandPool(const CommandPool&) = delete;
    CommandPool& operator=(const CommandPool&) = delete;

    // 允许移动
    CommandPool(CommandPool&& other) noexcept;
    CommandPool& operator=(CommandPool&& other) noexcept;

    void create(vk::Device device, uint32_t queueFamilyIndex, 
                vk::CommandPoolCreateFlags flags = {});
    void cleanup();

    vk::CommandBuffer allocateCommandBuffer(vk::CommandBufferLevel level = vk::CommandBufferLevel::ePrimary);
    void freeCommandBuffer(vk::CommandBuffer commandBuffer);
    void reset();

    vk::CommandPool getPool() const { return pool_; }

private:
    vk::Device device_ = nullptr;
    vk::CommandPool pool_ = nullptr;
};

} // namespace vk_gs
