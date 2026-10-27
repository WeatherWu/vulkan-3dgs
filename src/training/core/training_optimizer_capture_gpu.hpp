#pragma once
#include "training/core/training_optimizer_capture.hpp"
#include "vulkan/buffer.hpp"

#include <memory>

namespace vulkan3DGS {
// Bounded host-coherent staging on a compute or transfer queue. No live buffer ownership.
// Cross-queue producers require caller-provided semaphore synchronization and valid queue sharing.
class OptimizerCaptureTransfer {
public:
    OptimizerCaptureTransfer(vk::Device device, vk::PhysicalDevice physical, vk::Queue queue,
                             uint32_t family);
    void download(vk::DescriptorBufferInfo source, uint64_t offset, std::span<std::byte> bytes);
    void upload(vk::DescriptorBufferInfo destination, uint64_t offset,
                std::span<const std::byte> bytes);

private:
    vk::Device device_;
    vk::Queue queue_;
    Buffer staging_;
    CommandPool pool_;
    vk::CommandBuffer command_;
    vk::UniqueFence fence_;
};

void checkOptimizerCaptureBudget(vk::PhysicalDevice physical, uint32_t count, uint64_t deviceBytes,
                                 const std::filesystem::path& output = {});

// One-shot resource owner; caller keeps it alive through submission, wait and export.
class TrainingOptimizerCaptureSession {
public:
    TrainingOptimizerCaptureSession(vk::Device device, vk::PhysicalDevice physical, vk::Queue queue,
                                    uint32_t family, OptimizerCaptureHeader header,
                                    const std::filesystem::path& path);
    void recordPre(vk::CommandBuffer command,
                   const std::array<vk::DescriptorBufferInfo, 7>& sources,
                   const TrainingPushConstants& push, const TrainingForwardCamera& camera,
                   uint64_t dispatchedShaderHash);
    void markSubmitted() noexcept {
        submitted_ = true;
    }
    void markPrepareSubmitted() noexcept {
        prepareSubmitted_ = true;
    }
    void markCompleted() noexcept {
        completed_ = true;
    }
    bool submitted() const noexcept {
        return prepareSubmitted_ || submitted_;
    }
    bool optimizerSubmitted() const noexcept {
        return submitted_;
    }
    bool completed() const noexcept {
        return completed_;
    }
    void setFrame(uint32_t frame) noexcept {
        header_.frame = frame;
    }
    void save(const std::array<vk::DescriptorBufferInfo, 3>& post);

private:
    OptimizerCaptureHeader header_;
    std::filesystem::path path_;
    std::array<Buffer, 6> pre_;
    OptimizerCaptureTransfer transfer_;
    bool recorded_ = false;
    bool prepareSubmitted_ = false;
    bool submitted_ = false;
    bool completed_ = false;
};
} // namespace vulkan3DGS
