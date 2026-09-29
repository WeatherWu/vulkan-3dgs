#pragma once

#include <initializer_list>
#include <vulkan/vulkan.hpp>

#include "training/core/training_buffers.hpp"
#include "training/core/training_types.hpp"
#include "render/renderer.hpp"
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
    void setPixelTo2DGSMode(TrainingPixelTo2DGSMode mode) { pixelTo2DGSMode_ = mode; }
    bool subgroupPixelTo2DGSSupported() const { return subgroupPixelTo2DGSSupported_; }
    bool tileGaussianPixelTo2DGSSupported() const { return tileGaussianPixelTo2DGSSupported_; }
    bool vkSplatPerSplatSupported() const { return vkSplatPerSplatSupported_; }
    bool vkSplatTensorSupported() const { return vkSplatTensorSupported_; }
    TrainingPixelTo2DGSMode activePixelTo2DGSMode() const;
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
    void backpropTileGaussian();
    void backprop2DGSTo3DGS();
    void backprop2DGSTo3DGSAndOptimize();
    void optimizeParameters();
    void validateGaussians();
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
    ComputePipeline pixelTo2DGSWorkgroupPipeline_;
    ComputePipeline pixelTo2DGSSubgroupPipeline_;
    ComputePipeline pixelTo2DGSAdaptivePipeline_;
    ComputePipeline tileGaussianAtomicPipeline_;
    ComputePipeline vkSplatPerSplatPipeline_;
    ComputePipeline vkSplatTensorPipeline_;
    ComputePipeline twoDGSTo3DGSPipeline_;
    ComputePipeline fusedProjectionOptimizerPipeline_;
    ComputePipeline optimizerPipeline_;
    ComputePipeline gaussianValidationPipeline_;
    ComputePipeline validationFinalizePipeline_;
    vk::DescriptorPool descriptorPool_ = nullptr;
    vk::DescriptorSet backwardClearDescriptorSet_ = nullptr;
    vk::DescriptorSet lossDescriptorSet_ = nullptr;
    vk::DescriptorSet lossToPixelDescriptorSet_ = nullptr;
    vk::DescriptorSet pixelTo2DGSDescriptorSet_ = nullptr;
    vk::DescriptorSet pixelTo2DGSWorkgroupDescriptorSet_ = nullptr;
    vk::DescriptorSet pixelTo2DGSSubgroupDescriptorSet_ = nullptr;
    vk::DescriptorSet pixelTo2DGSAdaptiveDescriptorSet_ = nullptr;
    vk::DescriptorSet tileGaussianAtomicDescriptorSet_ = nullptr;
    vk::DescriptorSet vkSplatPerSplatDescriptorSet_ = nullptr;
    vk::DescriptorSet vkSplatTensorDescriptorSet_ = nullptr;
    vk::DescriptorSet twoDGSTo3DGSDescriptorSet_ = nullptr;
    vk::DescriptorSet fusedProjectionOptimizerDescriptorSet_ = nullptr;
    vk::DescriptorSet optimizerDescriptorSet_ = nullptr;
    vk::DescriptorSet gaussianValidationDescriptorSet_ = nullptr;
    vk::DescriptorSet validationFinalizeDescriptorSet_ = nullptr;
    const TrainingBuffers* trainingBuffers_ = nullptr;
    vk::CommandBuffer commandBuffer_ = nullptr;
    vk::QueryPool profilingQueryPool_ = nullptr;
    TrainingPushConstants pushConstants_{};
    TrainingPixelTo2DGSMode pixelTo2DGSMode_ = TrainingPixelTo2DGSMode::Auto;
    uint32_t subgroupSize_ = 0;
    bool subgroupPixelTo2DGSSupported_ = false;
    bool tileGaussianPixelTo2DGSSupported_ = false;
    bool vkSplatPerSplatSupported_ = false;
    bool vkSplatTensorSupported_ = false;
    bool computeBackwardClear_ = false;
    bool fusedProjectionOptimizerEnabled_ = true;
    bool initialized_ = false;
};

} // namespace vulkan3DGS
