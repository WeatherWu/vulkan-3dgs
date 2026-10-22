#pragma once

#include "training/core/backward/backward_command_utils.hpp"
#include "training/core/backward/backward_pass_context.hpp"
#include "training/core/training_tile_layout.hpp"
#include "vulkan/compute_pipeline.hpp"
#include "training/core/backward/checkpoint_state_pass.hpp"
#include "training/core/backward/checkpoint_backward_pass.hpp"
#include <memory>

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
[[nodiscard]] bool
supportsAdaptiveSpatialBackward(TrainingPixelTo2DGSMode requested,
                                const PixelTo2DGSCapabilities& capabilities) noexcept;
[[nodiscard]] bool supportsTensorSubgroupReduction(
    const PixelTo2DGSCapabilities& capabilities) noexcept;
[[nodiscard]] bool useTensorSubgroupReduction(
    bool requested, TrainingPixelTo2DGSMode activeMode, TrainingTileMode tileMode,
    const PixelTo2DGSCapabilities& capabilities) noexcept;
[[nodiscard]] bool useTensorGlobalPixelGrad(
    bool requested, bool subgroupRequested, TrainingPixelTo2DGSMode activeMode,
    TrainingTileMode tileMode, const PixelTo2DGSCapabilities& capabilities) noexcept;
[[nodiscard]] bool useTensorRgbLatencyHiding(
    bool requested, bool globalPixelGradRequested, bool subgroupRequested,
    TrainingPixelTo2DGSMode activeMode, TrainingTileMode tileMode,
    const PixelTo2DGSCapabilities& capabilities) noexcept;

class PixelTo2DGSDispatcher {
public:
    void initialize(vk::Device device, vk::PhysicalDevice physicalDevice);
    void cleanup();

    void setRequestedMode(TrainingPixelTo2DGSMode mode) {
        requestedMode_ = mode;
    }
    void setAdaptiveSpatialTiles(bool enabled) noexcept {
        adaptiveSpatialTiles_ = enabled;
    }
    void setAdaptiveTensorChildBatch(uint32_t batch) noexcept {
        adaptiveTensorChildBatch_ = normalizeAdaptiveTensorChildBatch(batch);
    }
    void setTensorSubgroupReduction(bool enabled) noexcept {
        tensorSubgroupReduction_ = enabled;
    }
    void setTensorGlobalPixelGrad(bool enabled) noexcept {
        tensorGlobalPixelGrad_ = enabled;
    }
    void setTensorRgbLatencyHiding(bool enabled) noexcept {
        tensorRgbLatencyHiding_ = enabled;
    }
    [[nodiscard]] bool tensorRgbLatencyHidingActive(uint32_t tileSize) const noexcept {
        const auto tileMode = adaptiveSpatialTiles_ ? TrainingTileMode::AdaptiveSpatial
            : tileSize == kTrainingTileSize16 ? TrainingTileMode::Fixed16
                                             : TrainingTileMode::Fixed8;
        return useTensorRgbLatencyHiding(tensorRgbLatencyHiding_, tensorGlobalPixelGrad_,
                                          tensorSubgroupReduction_, activeMode(), tileMode,
                                          capabilities_);
    }
    [[nodiscard]] bool tensorGlobalPixelGradActive(uint32_t tileSize) const noexcept {
        const auto tileMode = adaptiveSpatialTiles_ ? TrainingTileMode::AdaptiveSpatial
            : tileSize == kTrainingTileSize16 ? TrainingTileMode::Fixed16
                                             : TrainingTileMode::Fixed8;
        return useTensorGlobalPixelGrad(tensorGlobalPixelGrad_, tensorSubgroupReduction_,
                                         activeMode(), tileMode, capabilities_);
    }
    [[nodiscard]] bool tensorSubgroupReductionActive(uint32_t tileSize) const noexcept {
        const auto tileMode = adaptiveSpatialTiles_ ? TrainingTileMode::AdaptiveSpatial
            : tileSize == kTrainingTileSize16 ? TrainingTileMode::Fixed16
                                             : TrainingTileMode::Fixed8;
        return useTensorSubgroupReduction(tensorSubgroupReduction_, activeMode(), tileMode,
                                           capabilities_);
    }
    [[nodiscard]] TrainingPixelTo2DGSMode activeMode() const noexcept;
    [[nodiscard]] bool adaptiveSpatialModeSupported() const noexcept {
        return supportsAdaptiveSpatialBackward(requestedMode_, capabilities_);
    }
    [[nodiscard]] const PixelTo2DGSCapabilities& capabilities() const noexcept {
        return capabilities_;
    }
    void record(const BackwardPassContext& context);
    [[nodiscard]] bool tensorEarlyCullSupported() const noexcept {
        return static_cast<bool>(tensorEarlyCullPipeline_.getPipeline());
    }
    void prepareCheckpointDiagnostics(TrainingBuffers& buffers, uint32_t itemCount, bool warp = false);
    void recordCheckpointDiagnostics(const BackwardPassContext& context, bool resetRun);

private:
    void recordSelected(const BackwardPassContext& context, vk::Buffer indirect = nullptr,
                        vk::DeviceSize offset = 0u);
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
    void recordAdaptiveSpatial(const BackwardPassContext& context, TrainingPixelTo2DGSMode mode);
    void recordAdaptiveSpatialDispatch(const BackwardPassContext& context,
                                       ComputePipeline& pipeline, vk::DescriptorSet descriptorSet,
                                       uint32_t tileSize);

    vk::Device device_ = nullptr;
    vk::PhysicalDevice physicalDevice_ = nullptr;
    std::unique_ptr<CheckpointStatePass> checkpointState_;
    std::unique_ptr<CheckpointBackwardPass> checkpointBackward_;
    std::array<TilePipelineVariant, 2> tileVariants_{};
    ComputePipeline tensorSubgroupPipeline_;
    vk::DescriptorSet tensorSubgroupDescriptorSet_ = nullptr;
    ComputePipeline tensorGlobalPixelGradPipeline_;
    vk::DescriptorSet tensorGlobalPixelGradDescriptorSet_ = nullptr;
    ComputePipeline tensorRgbLatencyPipeline_;
    vk::DescriptorSet tensorRgbLatencyDescriptorSet_ = nullptr;
    ComputePipeline tensorEarlyCullPipeline_;
    vk::DescriptorSet tensorEarlyCullDescriptorSet_ = nullptr;
    std::array<ComputePipeline, 2> adaptiveSpatialPerSplatPipelines_{};
    std::array<ComputePipeline, 2> adaptiveSpatialTensorPipelines_{};
    std::array<ComputePipeline, 2> adaptiveTensorLargeChildPipelines_{};
    BackwardDescriptorPool descriptorPool_;
    std::array<vk::DescriptorSet, 2> adaptiveSpatialPerSplatDescriptorSets_{};
    std::array<vk::DescriptorSet, 2> adaptiveSpatialTensorDescriptorSets_{};
    std::array<vk::DescriptorSet, 2> adaptiveTensorLargeChildDescriptorSets_{};
    uint32_t adaptiveTensorChildBatch_ = 8u;
    TrainingPixelTo2DGSMode requestedMode_ = TrainingPixelTo2DGSMode::Auto;
    bool adaptiveSpatialTiles_ = false;
    bool tensorSubgroupReduction_ = false;
    bool tensorGlobalPixelGrad_ = false;
    bool tensorRgbLatencyHiding_ = false;
    PixelTo2DGSCapabilities capabilities_{};
};

} // namespace vulkan3DGS
