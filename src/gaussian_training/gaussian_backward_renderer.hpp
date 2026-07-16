#pragma once

#include <initializer_list>
#include <vulkan/vulkan.hpp>

#include "gaussian_training/training_buffers.hpp"
#include "gaussian_training/training_types.hpp"
#include "renderer.hpp"
#include "vulkan/compute_pipeline.hpp"

namespace vulkan3DGS {

class GaussianBackwardRenderer : public BackwardRenderer {
public:
    void initialize(vk::Device device,
                    vk::PhysicalDevice physicalDevice,
                    vk::Queue computeQueue,
                    uint32_t computeQueueFamilyIndex,
                    uint32_t gaussianCount,
                    TrainingExtent extent) override;
    void cleanup() override;

    void setTrainingBuffers(const TrainingBuffers& trainingBuffers,
                            vk::CommandBuffer commandBuffer,
                            TrainingPushConstants pushConstants);
    void setProfilingQueryPool(vk::QueryPool queryPool);
    void backward() override;
    void gradientDescent() override;

    bool isInitialized() const override { return initialized_; }
    uint32_t gaussianCount() const { return gaussianCount_; }
    TrainingExtent extent() const { return extent_; }

private:
    void createBackwardResources();
    void destroyBackwardResources();
    void computeLoss();
    void clearBackwardBuffers();
    void computeLossToPixel();
    void backpropPixelTo2DGS();
    void backprop2DGSTo3DGS();
    void optimizeParameters();
    void finalizeValidation();
    void shaderBufferBarrier(std::initializer_list<vk::DescriptorBufferInfo> buffers,
                             vk::AccessFlags dstAccessMask);
    void writeProfilingTimestamp(TrainingGpuProfileStage stage, bool end);

    vk::Device device_ = nullptr;
    vk::PhysicalDevice physicalDevice_ = nullptr;
    vk::Queue computeQueue_ = nullptr;
    uint32_t computeQueueFamilyIndex_ = 0;
    uint32_t gaussianCount_ = 0;
    TrainingExtent extent_{};
    ComputePipeline backwardClearPipeline_;
    ComputePipeline lossPipeline_;
    ComputePipeline lossToPixelPipeline_;
    ComputePipeline pixelTo2DGSPipeline_;
    ComputePipeline twoDGSTo3DGSPipeline_;
    ComputePipeline optimizerPipeline_;
    ComputePipeline validationFinalizePipeline_;
    vk::DescriptorPool descriptorPool_ = nullptr;
    vk::DescriptorSet backwardClearDescriptorSet_ = nullptr;
    vk::DescriptorSet lossDescriptorSet_ = nullptr;
    vk::DescriptorSet lossToPixelDescriptorSet_ = nullptr;
    vk::DescriptorSet pixelTo2DGSDescriptorSet_ = nullptr;
    vk::DescriptorSet twoDGSTo3DGSDescriptorSet_ = nullptr;
    vk::DescriptorSet optimizerDescriptorSet_ = nullptr;
    vk::DescriptorSet validationFinalizeDescriptorSet_ = nullptr;
    const TrainingBuffers* trainingBuffers_ = nullptr;
    vk::CommandBuffer commandBuffer_ = nullptr;
    vk::QueryPool profilingQueryPool_ = nullptr;
    TrainingPushConstants pushConstants_{};
    bool initialized_ = false;
};

} // namespace vulkan3DGS
