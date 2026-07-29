#pragma once

#include <initializer_list>
#include <vulkan/vulkan.hpp>

#include "gaussian_training/training_buffers.hpp"
#include "gaussian_training/training_types.hpp"
#include "vulkan/compute_pipeline.hpp"

namespace vulkan3DGS {

class GaussianDensificationRenderer {
public:
    void initialize(vk::Device device,
                    vk::PhysicalDevice physicalDevice,
                    vk::Queue computeQueue,
                    uint32_t computeQueueFamilyIndex);
    void cleanup();

    void setTrainingBuffers(const TrainingBuffers& trainingBuffers,
                            vk::CommandBuffer commandBuffer,
                            TrainingDensificationPushConstants pushConstants);
    void setProfilingQueryPool(vk::QueryPool queryPool);
    void densifyAndPrune();

    bool isInitialized() const { return initialized_; }

private:
    void createResources();
    void destroyResources();
    void updateDescriptorSet(vk::DescriptorSet descriptorSet,
                             std::initializer_list<uint32_t> bindings);
    void bindAndDispatch(ComputePipeline& pipeline,
                         vk::DescriptorSet descriptorSet,
                         uint32_t groupCountX);
    void shaderBufferBarrier(std::initializer_list<vk::DescriptorBufferInfo> buffers,
                             vk::AccessFlags dstAccessMask);
    void clearDensificationStates();
    void writeProfilingTimestamp(TrainingGpuProfileStage stage, bool end);

    vk::Device device_ = nullptr;
    vk::PhysicalDevice physicalDevice_ = nullptr;
    vk::Queue computeQueue_ = nullptr;
    uint32_t computeQueueFamilyIndex_ = 0;

    ComputePipeline clearPipeline_;
    ComputePipeline densifyPrunePipeline_;
    ComputePipeline dispatchBuildPipeline_;
    ComputePipeline opacityResetPipeline_;
    vk::DescriptorPool descriptorPool_ = nullptr;
    vk::DescriptorSet clearDescriptorSet_ = nullptr;
    vk::DescriptorSet densifyPruneDescriptorSet_ = nullptr;
    vk::DescriptorSet dispatchBuildDescriptorSet_ = nullptr;
    vk::DescriptorSet opacityResetDescriptorSet_ = nullptr;

    const TrainingBuffers* trainingBuffers_ = nullptr;
    vk::CommandBuffer commandBuffer_ = nullptr;
    vk::QueryPool profilingQueryPool_ = nullptr;
    TrainingDensificationPushConstants pushConstants_{};
    bool initialized_ = false;
};

} // namespace vulkan3DGS
