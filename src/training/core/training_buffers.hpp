#pragma once

#include <array>
#include <optional>
#include <vector>
#include <vulkan/vulkan.hpp>

#include "training/core/training_types.hpp"
#include "vulkan/buffer.hpp"

namespace vulkan3DGS {

class TrainingBuffers {
public:
    void initialize(vk::Device device,
                    vk::PhysicalDevice physicalDevice,
                    vk::Queue transferQueue,
                    uint32_t transferQueueFamilyIndex,
                    uint32_t computeQueueFamilyIndex);
    void cleanup();

    void resize(uint32_t gaussianCount, TrainingExtent extent);
    void resizeTileItems(uint32_t tileItemCapacity);
    void ensureTileSortStorage(uint32_t tileItemCapacity, vk::DeviceSize sortStorageSize, vk::BufferUsageFlags sortStorageUsage);
    void ensureDensificationCapacity(uint32_t gaussianCapacity);
    void adoptDensifiedGaussians(uint32_t gaussianCount);
    void uploadGaussianParams(const GaussianTrainParam* params, uint32_t gaussianCount);
    std::vector<GaussianTrainParam> downloadGaussianParams(uint32_t gaussianCount);
    std::vector<ProjectedGaussianGrad> downloadProjectedGrads(uint32_t gaussianCount);
    std::vector<TrainingAdaptiveTileDiagnostic> downloadAdaptiveTileDiagnostics(
        uint32_t parentItemCount);
    std::vector<uint32_t> downloadAdaptiveChildMasks(uint32_t parentItemCount);
    std::vector<uint8_t> downloadAdaptiveChildMembership(uint32_t gaussianCount,
                                                          uint32_t parentItemCount,
                                                          uint32_t childItemCount);
    std::vector<float> downloadLoss(uint32_t pixelCount);
    std::vector<glm::vec4> downloadRenderedColor(uint32_t pixelCount);
    void uploadTargetColor(const uint8_t* pixels, uint32_t width, uint32_t height);
    void setTargetColorDescriptor(vk::DescriptorBufferInfo descriptor) {
        targetColorDescriptorOverride_ = descriptor;
    }
    void clearTargetColorDescriptor() {
        targetColorDescriptorOverride_.reset();
    }
    void uploadCamera(const TrainingForwardCamera& camera);
    void recordTileItemCountReadback(vk::CommandBuffer commandBuffer);
    void recordDensificationStatsReadback(vk::CommandBuffer commandBuffer);
    uint32_t requiredTileItemCount();
    [[nodiscard]] std::array<uint32_t, 2> adaptiveChildStats() const;
    uint32_t densifiedGaussianCount();
    TrainingDensificationStats densificationStats();

    uint32_t gaussianCapacity() const { return gaussianCapacity_; }
    uint32_t densificationCapacity() const { return densificationCapacity_; }
    uint32_t tileItemCapacity() const { return tileItemCapacity_; }
    TrainingExtent extent() const { return extent_; }

    vk::DescriptorBufferInfo gaussianParamsInfo() const { return gaussianParams_.getDescriptorInfo(); }
    vk::DescriptorBufferInfo gaussianGradsInfo() const { return gaussianGrads_.getDescriptorInfo(); }
    vk::DescriptorBufferInfo adamStatesInfo() const { return adamStates_.getDescriptorInfo(); }
    vk::DescriptorBufferInfo projectedInfo() const { return projected_.getDescriptorInfo(); }
    vk::DescriptorBufferInfo tileItemsInfo() const { return tileItemsSorted_.getDescriptorInfo(); }
    vk::DescriptorBufferInfo tileItemsUnsortedInfo() const { return tileItems_.getDescriptorInfo(); }
    vk::DescriptorBufferInfo tileKeyLowInfo() const { return tileKeyLow_.getDescriptorInfo(); }
    vk::DescriptorBufferInfo tileKeyHighInfo() const { return tileKeyHigh_.getDescriptorInfo(); }
    vk::DescriptorBufferInfo tileSortIndicesInfo() const { return tileSortIndices_.getDescriptorInfo(); }
    vk::DescriptorBufferInfo tileSortScratchInfo() const { return tileSortScratch_.getDescriptorInfo(); }
    vk::DescriptorBufferInfo tileItemsSortedInfo() const { return tileItemsSorted_.getDescriptorInfo(); }
    vk::Buffer tileItemsBuffer() const { return tileItems_.getBuffer(); }
    vk::Buffer tileKeyLowBuffer() const { return tileKeyLow_.getBuffer(); }
    vk::Buffer tileKeyHighBuffer() const { return tileKeyHigh_.getBuffer(); }
    vk::Buffer tileSortIndicesBuffer() const { return tileSortIndices_.getBuffer(); }
    vk::Buffer tileSortScratchBuffer() const { return tileSortScratch_.getBuffer(); }
    vk::Buffer tileSortStorageBuffer() const { return tileSortStorage_.getBuffer(); }
    vk::DescriptorBufferInfo tileRangesInfo() const { return tileRanges_.getDescriptorInfo(); }
    vk::DescriptorBufferInfo gaussianTileRangesInfo() const {
        return gaussianTileRanges_.getDescriptorInfo();
    }
    vk::DescriptorBufferInfo gaussianTilePrefixScratchInfo() const {
        return gaussianTilePrefixScratch_.getDescriptorInfo();
    }
    vk::DescriptorBufferInfo adaptiveChildCountsInfo() const {
        return adaptiveChildCounts_.getDescriptorInfo();
    }
    vk::DescriptorBufferInfo adaptiveTileSplitFlagsInfo() const {
        return adaptiveTileSplitFlags_.getDescriptorInfo();
    }
    vk::DescriptorBufferInfo adaptiveChildMasksInfo() const {
        return adaptiveChildMasks_.getDescriptorInfo();
    }
    vk::DescriptorBufferInfo adaptiveChildOffsetsInfo() const {
        return adaptiveChildOffsets_.getDescriptorInfo();
    }
    vk::DescriptorBufferInfo adaptiveChildPrefixInfo() const {
        return adaptiveChildPrefix_.getDescriptorInfo();
    }
    vk::DescriptorBufferInfo adaptiveChildBlockSumsInfo() const {
        return adaptiveChildBlockSums_.getDescriptorInfo();
    }
    vk::DescriptorBufferInfo adaptiveActiveParentsInfo() const {
        return adaptiveActiveParents_.getDescriptorInfo();
    }
    vk::DescriptorBufferInfo adaptiveChildDispatchInfo() const {
        return adaptiveChildDispatch_.getDescriptorInfo();
    }
    vk::Buffer adaptiveChildDispatchBuffer() const { return adaptiveChildDispatch_.getBuffer(); }
    vk::DescriptorBufferInfo renderedColorInfo() const {
        return renderedColor_.getDescriptorInfo();
    }
    vk::DescriptorBufferInfo targetColorInfo() const {
        return targetColorDescriptorOverride_.value_or(targetColor_.getDescriptorInfo());
    }
    vk::DescriptorBufferInfo pixelGradsInfo() const { return pixelGrads_.getDescriptorInfo(); }
    vk::DescriptorBufferInfo ssimBackwardStatesInfo() const { return ssimBackwardStates_.getDescriptorInfo(); }
    vk::DescriptorBufferInfo pixelValidationPartialsInfo() const { return pixelValidationPartials_.getDescriptorInfo(); }
    vk::DescriptorBufferInfo gaussianValidationPartialsInfo() const { return gaussianValidationPartials_.getDescriptorInfo(); }
    vk::DescriptorBufferInfo densifiedGaussianValidationPartialsInfo() const { return densifiedGaussianValidationPartials_.getDescriptorInfo(); }
    vk::DescriptorBufferInfo validationFinalResultInfo() const { return validationFinalResult_.getDescriptorInfo(); }
    vk::Buffer validationFinalResultBuffer() const { return validationFinalResult_.getBuffer(); }
    vk::DescriptorBufferInfo pixelBlendStatesInfo() const { return pixelBlendStates_.getDescriptorInfo(); }
    vk::DescriptorBufferInfo densificationStatesInfo() const { return densificationStates_.getDescriptorInfo(); }
    vk::Buffer densificationStatesBuffer() const { return densificationStates_.getBuffer(); }
    vk::DescriptorBufferInfo densifiedParamsInfo() const { return densifiedParams_.getDescriptorInfo(); }
    vk::DescriptorBufferInfo densifiedAdamStatesInfo() const { return densifiedAdamStates_.getDescriptorInfo(); }
    vk::DescriptorBufferInfo densificationCountersInfo() const { return densificationCounters_.getDescriptorInfo(); }
    vk::Buffer densificationCountersBuffer() const { return densificationCounters_.getBuffer(); }
    vk::DescriptorBufferInfo projectedGradsInfo() const { return projectedGrads_.getDescriptorInfo(); }
    vk::DescriptorBufferInfo lossInfo() const { return loss_.getDescriptorInfo(); }
    vk::DescriptorBufferInfo countersInfo() const { return counters_.getDescriptorInfo(); }
    vk::DescriptorBufferInfo previewInstancesInfo() const { return previewInstances_.getDescriptorInfo(); }
    vk::DescriptorBufferInfo cameraInfo() const { return camera_.getDescriptorInfo(); }

private:
    void createStorageBuffer(Buffer& buffer, vk::DeviceSize size);
    void createUniformBuffer(Buffer& buffer, vk::DeviceSize size);
    void createZeroedStorageBuffer(Buffer& buffer, vk::DeviceSize size);

    vk::Device device_ = nullptr;
    vk::PhysicalDevice physicalDevice_ = nullptr;
    vk::Queue transferQueue_ = nullptr;
    uint32_t transferQueueFamilyIndex_ = 0;
    uint32_t computeQueueFamilyIndex_ = 0;

    uint32_t gaussianCapacity_ = 0;
    uint32_t gaussianWorkspaceCapacity_ = 0;
    uint32_t densificationCapacity_ = 0;
    uint32_t tileItemCapacity_ = 0;
    TrainingExtent extent_{};

    Buffer gaussianParams_;
    Buffer gaussianGrads_;
    Buffer adamStates_;
    Buffer projected_;
    Buffer tileItems_;
    Buffer tileKeyLow_;
    Buffer tileKeyHigh_;
    Buffer tileSortIndices_;
    Buffer tileSortScratch_;
    Buffer tileItemsSorted_;
    Buffer tileSortStorage_;
    Buffer tileRanges_;
    Buffer gaussianTileRanges_;
    Buffer gaussianTilePrefixScratch_;
    Buffer adaptiveChildCounts_;
    Buffer adaptiveTileSplitFlags_;
    Buffer adaptiveChildMasks_;
    Buffer adaptiveChildOffsets_;
    Buffer adaptiveChildPrefix_;
    Buffer adaptiveChildBlockSums_;
    Buffer adaptiveActiveParents_;
    Buffer adaptiveChildDispatch_;
    Buffer renderedColor_;
    Buffer targetColor_;
    std::optional<vk::DescriptorBufferInfo> targetColorDescriptorOverride_;
    Buffer pixelGrads_;
    Buffer ssimBackwardStates_;
    Buffer pixelValidationPartials_;
    Buffer gaussianValidationPartials_;
    Buffer densifiedGaussianValidationPartials_;
    Buffer validationFinalResult_;
    Buffer pixelBlendStates_;
    Buffer densificationStates_;
    Buffer densifiedParams_;
    Buffer densifiedAdamStates_;
    Buffer densificationCounters_;
    Buffer projectedGrads_;
    Buffer loss_;
    Buffer counters_;
    Buffer tileItemCountReadback_;
    void* tileItemCountReadbackMapped_ = nullptr;
    Buffer densificationStatsReadback_;
    void* densificationStatsReadbackMapped_ = nullptr;
    Buffer previewInstances_;
    Buffer camera_;
};

} // namespace vulkan3DGS
