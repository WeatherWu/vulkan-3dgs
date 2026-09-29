#pragma once

#include "training/core/gaussian_backward_renderer.hpp"
#include "training/core/gaussian_densification_renderer.hpp"
#include "training/core/gaussian_forward_renderer.hpp"
#include "training/core/training_buffers.hpp"
#include "training/core/training_profiling_service.hpp"
#include "training/core/training_validation_service.hpp"
#include "vulkan/command_pool.hpp"

#include <cstdint>

namespace vulkan3DGS {

struct TrainingStepExecutionRequest {
    TrainingBuffers& buffers;
    GaussianForwardRenderer& forward;
    GaussianBackwardRenderer& backward;
    GaussianDensificationRenderer& densification;
    TrainingValidationService& validation;
    TrainingProfilingService& profiling;
    TrainingPushConstants pushConstants{};
    TrainingDensificationPushConstants densificationPushConstants{};
    vk::Semaphore uploadSemaphore = nullptr;
    uint64_t uploadWaitValue = 0;
    uint32_t trainableGaussianCount = 0;
    uint32_t maxGaussianCount = 0;
    uint32_t splitChildren = 2;
    bool runDensification = false;
    bool pruneByScreenSize = false;
};

struct TrainingStepExecutionResult {
    uint32_t tileItemCount = 0;
    uint32_t gaussianCount = 0;
    TrainingDensificationStats densification{};
    bool optimizerUpdated = false;
    bool densificationRan = false;
    bool gpuProfilingDisabled = false;
};

// Owns the Vulkan command resources and the prepare/main execution mechanics
// of one training iteration. It has no scheduling, dataset, UI or model-I/O
// ownership.
class TrainingStepExecutor {
public:
    TrainingStepExecutor() = default;
    ~TrainingStepExecutor() noexcept;

    TrainingStepExecutor(const TrainingStepExecutor&) = delete;
    TrainingStepExecutor& operator=(const TrainingStepExecutor&) = delete;

    void initialize(vk::Device device, vk::Queue computeQueue, uint32_t computeQueueFamilyIndex);
    void cleanup();
    [[nodiscard]] bool initialized() const noexcept {
        return prepareCommandBuffer_ && mainCommandBuffer_;
    }
    [[nodiscard]] TrainingStepExecutionResult execute(TrainingStepExecutionRequest request);

private:
    vk::Device device_ = nullptr;
    vk::Queue computeQueue_ = nullptr;
    uint32_t computeQueueFamilyIndex_ = 0;
    CommandPool commandPool_;
    vk::CommandBuffer prepareCommandBuffer_ = nullptr;
    vk::CommandBuffer mainCommandBuffer_ = nullptr;
    vk::Fence prepareFence_ = nullptr;
    vk::Fence mainFence_ = nullptr;
};

} // namespace vulkan3DGS
