#pragma once

#include "training/core/backward/backward_command_utils.hpp"
#include "training/core/backward/backward_pass_context.hpp"
#include "training/core/backward/vksplat_loss_pass.hpp"
#include "vulkan/compute_pipeline.hpp"

namespace vulkan3DGS {

class BackwardLossPass {
public:
    void initialize(vk::Device device);
    void configureVkSplat(vk::PhysicalDevice physical, vk::Queue queue, uint32_t family) {
        physical_ = physical;
        queue_ = queue;
        family_ = family;
    }
    void setVkSplat(bool requested) {
        vkSplatRequested_ = requested;
        if (!requested) vkSplat_.reset();
    }
    bool vkSplatActive() const {
        return vkSplatActive_;
    }
    void cleanup();
    void setSsimMode(TrainingSsimMode mode) { ssimMode_ = mode; }
    [[nodiscard]] TrainingSsimMode activeSsimMode() const { return ssimMode_; }

    void recordLoss(const BackwardPassContext& context);
    void recordClear(const BackwardPassContext& context, bool fusedProjectionOptimizerEnabled);
    void recordLossToPixel(const BackwardPassContext& context);

private:
    void recordSsimHorizontal(const BackwardPassContext& context, bool coefficients);
    void recordScratchReadComplete(const BackwardPassContext& context);
    vk::Device device_ = nullptr;
    ComputePipeline clearPipeline_;
    ComputePipeline lossPipeline_;
    ComputePipeline lossToPixelPipeline_;
    ComputePipeline ssimMomentsHorizontalPipeline_;
    ComputePipeline ssimCoefficientsHorizontalPipeline_;
    ComputePipeline separableLossPipeline_;
    ComputePipeline separableLossToPixelPipeline_;
    BackwardDescriptorPool descriptorPool_;
    vk::DescriptorSet clearDescriptorSet_ = nullptr;
    vk::DescriptorSet lossDescriptorSet_ = nullptr;
    vk::DescriptorSet lossToPixelDescriptorSet_ = nullptr;
    vk::DescriptorSet ssimMomentsHorizontalSet_ = nullptr;
    vk::DescriptorSet ssimCoefficientsHorizontalSet_ = nullptr;
    vk::DescriptorSet separableLossSet_ = nullptr;
    vk::DescriptorSet separableLossToPixelSet_ = nullptr;
    TrainingSsimMode ssimMode_ = TrainingSsimMode::TwoDimensional;
    bool computeClear_ = false;
    vk::PhysicalDevice physical_ = nullptr;
    vk::Queue queue_ = nullptr;
    uint32_t family_ = 0;
    bool vkSplatRequested_ = false;
    bool vkSplatActive_ = false;
    std::unique_ptr<VkSplatLossPass> vkSplat_;
};

} // namespace vulkan3DGS
