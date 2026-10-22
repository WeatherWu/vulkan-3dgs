#pragma once

#include "training/core/backward/backward_command_utils.hpp"
#include "training/core/backward/backward_pass_context.hpp"
#include "training/core/training_tile_layout.hpp"
#include "vulkan/compute_pipeline.hpp"

#include <array>

namespace vulkan3DGS {

struct PixelTo2DGSCapabilities {
    uint32_t subgroupSize = 0;
    bool subgroup = false;
    bool tileGaussian = false;
    bool vkSplatPerSplat = false;
    bool vkSplatTensor = false;
};

[[nodiscard]] TrainingPixelTo2DGSMode
resolvePixelTo2DGSMode(TrainingPixelTo2DGSMode requested,
                       const PixelTo2DGSCapabilities& capabilities) noexcept;
[[nodiscard]] bool supportsTensorSubgroupReduction(
    const PixelTo2DGSCapabilities& capabilities) noexcept;
[[nodiscard]] bool useTensorSubgroupReduction(
    bool requested, TrainingPixelTo2DGSMode activeMode, TrainingTileMode tileMode,
    const PixelTo2DGSCapabilities& capabilities) noexcept;
[[nodiscard]] bool useTensorGlobalPixelGrad(
    bool requested, bool subgroupRequested, TrainingPixelTo2DGSMode activeMode,
    TrainingTileMode tileMode, const PixelTo2DGSCapabilities& capabilities) noexcept;

class PixelTo2DGSDispatcher {
public:
    void initialize(vk::Device device, vk::PhysicalDevice physicalDevice);
    void cleanup();

    void setRequestedMode(TrainingPixelTo2DGSMode mode) {
        requestedMode_ = mode;
    }

    void setTensorSubgroupReduction(bool enabled) noexcept {
        tensorSubgroupReduction_ = enabled;
    }
    void setTensorGlobalPixelGrad(bool enabled) noexcept {
        tensorGlobalPixelGrad_ = enabled;
    }

    [[nodiscard]] bool tensorGlobalPixelGradActive(uint32_t tileSize) const noexcept {
        const auto tileMode = tileSize == kTrainingTileSize16 ? TrainingTileMode::Fixed16
                                             : TrainingTileMode::Fixed8;
        return useTensorGlobalPixelGrad(tensorGlobalPixelGrad_, tensorSubgroupReduction_,
                                         activeMode(), tileMode, capabilities_);
    }
    [[nodiscard]] bool tensorSubgroupReductionActive(uint32_t tileSize) const noexcept {
        const auto tileMode = tileSize == kTrainingTileSize16 ? TrainingTileMode::Fixed16
                                             : TrainingTileMode::Fixed8;
        return useTensorSubgroupReduction(tensorSubgroupReduction_, activeMode(), tileMode,
                                           capabilities_);
    }
    [[nodiscard]] TrainingPixelTo2DGSMode activeMode() const noexcept;

    [[nodiscard]] const PixelTo2DGSCapabilities& capabilities() const noexcept {
        return capabilities_;
    }
    void record(const BackwardPassContext& context);

private:
    struct TilePipelineVariant {
        uint32_t tileSize = kTrainingTileSize16;
        ComputePipeline directPipeline;
        ComputePipeline workgroupPipeline;
        ComputePipeline subgroupPipeline;
        ComputePipeline adaptivePipeline;
        ComputePipeline tileGaussianPipeline;
        ComputePipeline vkSplatPerSplatPipeline;
        ComputePipeline vkSplatTensorPipeline;
        vk::DescriptorSet directDescriptorSet = nullptr;
        vk::DescriptorSet workgroupDescriptorSet = nullptr;
        vk::DescriptorSet subgroupDescriptorSet = nullptr;
        vk::DescriptorSet adaptiveDescriptorSet = nullptr;
        vk::DescriptorSet tileGaussianDescriptorSet = nullptr;
        vk::DescriptorSet vkSplatPerSplatDescriptorSet = nullptr;
        vk::DescriptorSet vkSplatTensorDescriptorSet = nullptr;
    };

    void initializeVariant(TilePipelineVariant& variant, uint32_t tileSize,
                           const ComputePipelineConfig& config);
    void cleanupVariant(TilePipelineVariant& variant);
    [[nodiscard]] TilePipelineVariant& variantForTileSize(uint32_t tileSize);
    void recordTileGaussian(const BackwardPassContext& context, TilePipelineVariant& variant);

    vk::Device device_ = nullptr;
    std::array<TilePipelineVariant, 2> tileVariants_{};
    ComputePipeline tensorSubgroupPipeline_;
    vk::DescriptorSet tensorSubgroupDescriptorSet_ = nullptr;
    ComputePipeline tensorGlobalPixelGradPipeline_;
    vk::DescriptorSet tensorGlobalPixelGradDescriptorSet_ = nullptr;
    BackwardDescriptorPool descriptorPool_;
    TrainingPixelTo2DGSMode requestedMode_ = TrainingPixelTo2DGSMode::Auto;
    bool tensorSubgroupReduction_ = false;
    bool tensorGlobalPixelGrad_ = false;
    PixelTo2DGSCapabilities capabilities_{};
};

} // namespace vulkan3DGS
