#pragma once

#include <initializer_list>
#include <vulkan/vulkan.hpp>

#include "training/core/training_buffers.hpp"
#include "training/core/training_tile_layout.hpp"
#include "training/core/training_types.hpp"
#include "training/core/training_faster_sort.hpp"
#include "render/renderer.hpp"
#include "vulkan/compute_pipeline.hpp"
#include <vk_radix_sort.h>
#include <array>

namespace vulkan3DGS {

enum class TrainingSortTimestamp : uint32_t {
    CountBegin, CountEnd, PrefixEnd, EmitBegin, EmitEnd, DepthEnd,
    HighGatherEnd, TileEnd, ItemGatherEnd, RangesEnd,
    RangeBoundaryDispatchEnd, RangeBoundaryBarrierEnd, RangeFinalizeDispatchEnd, Count
};
inline constexpr uint32_t kTrainingSortTimestampCount = uint32_t(TrainingSortTimestamp::Count);

class GaussianForwardRenderer : public ForwardTrainingRenderer {
public:
    void initialize(vk::Device device,
                    vk::PhysicalDevice physicalDevice,
                    vk::Queue computeQueue,
                    uint32_t computeQueueFamilyIndex,
                    uint32_t gaussianCount,
                    TrainingExtent extent) override;
    void cleanup() override;

    void setTrainingBuffers(TrainingBuffers& trainingBuffers,
                            vk::CommandBuffer commandBuffer,
                            TrainingPushConstants pushConstants);
    void setProfilingQueryPool(vk::QueryPool queryPool);
    void setCompositeMode(TrainingForwardCompositeMode mode) { compositeMode_ = mode; }
    void forward() override;
    void prepareTileItems();
    void renderPreparedTiles(uint32_t tileItemCount);
    // Private replay callers own the query pool and restore inputs before recording.
    void setSortDiagnosticQueryPool(vk::QueryPool pool) { sortDiagnosticQueries_ = pool; }
    void prepareProjectedTileItems();
    void emitPreparedTileItems(uint32_t tileItemCount);
    void buildPreparedTileLists(uint32_t tileItemCount);
    // Change only while idle, before recording the next prepare submission.
    void setFasterSortEnabled(bool enabled) { fasterSortRequested_ = enabled; }
    bool fasterSortActive() const { return fasterSortPrepared_; }
    uint32_t fasterSortParticipatingCount() const { return fasterSortPrepared_ && fasterSort_ ? fasterSort_->participatingCount() : 0u; }

    bool isInitialized() const override { return initialized_; }

private:
    struct TilePipelineVariant {
        uint32_t tileSize = kTrainingTileSize16;
        ComputePipeline tileClearPipeline;
        ComputePipeline tileCountPipeline;
        ComputePipeline tileEmitPipeline;
        ComputePipeline tileRangeBoundariesPipeline;
        ComputePipeline tileRangeBuildPipeline;
        ComputePipeline forwardPipeline;
        ComputePipeline forwardWorkgroupPipeline;
        vk::DescriptorSet tileClearDescriptorSet = nullptr;
        vk::DescriptorSet tileCountDescriptorSet = nullptr;
        vk::DescriptorSet tileEmitDescriptorSet = nullptr;
        vk::DescriptorSet tileRangeBoundariesDescriptorSet = nullptr;
        vk::DescriptorSet tileRangeBuildDescriptorSet = nullptr;
        vk::DescriptorSet forwardDescriptorSet = nullptr;
        vk::DescriptorSet forwardWorkgroupDescriptorSet = nullptr;
    };

    struct AdaptiveTilePipelines {
        ComputePipeline countChildren;
        ComputePipeline prefixChildren;
        ComputePipeline prefixChildBlocks;
        ComputePipeline selectChildren;
        ComputePipeline dispatchChildren;
        ComputePipeline scatterChildren;
        ComputePipeline buildParentBoundaries;
        ComputePipeline clearLeafRanges;
        ComputePipeline buildLeafRanges;
        ComputePipeline installChildRanges;
        ComputePipeline forwardDirect;
        std::array<ComputePipeline, 2> forwardWorkgroup;
        vk::DescriptorSet countChildrenSet = nullptr;
        vk::DescriptorSet prefixChildrenSet = nullptr;
        vk::DescriptorSet prefixChildBlocksSet = nullptr;
        vk::DescriptorSet selectChildrenSet = nullptr;
        vk::DescriptorSet dispatchChildrenSet = nullptr;
        vk::DescriptorSet scatterChildrenSet = nullptr;
        vk::DescriptorSet buildParentBoundariesSet = nullptr;
        vk::DescriptorSet clearLeafRangesSet = nullptr;
        vk::DescriptorSet buildLeafRangesSet = nullptr;
        vk::DescriptorSet installChildRangesSet = nullptr;
        vk::DescriptorSet forwardDirectSet = nullptr;
        std::array<vk::DescriptorSet, 2> forwardWorkgroupSets{};
    };

    void createForwardResources();
    void destroyForwardResources();
    void initializeTileVariant(TilePipelineVariant& variant, uint32_t tileSize);
    void cleanupTileVariant(TilePipelineVariant& variant);
    void initializeAdaptivePipelines();
    void cleanupAdaptivePipelines();
    [[nodiscard]] TilePipelineVariant& activeTileVariant();
    [[nodiscard]] const TilePipelineVariant& activeTileVariant() const;
    void projectGaussians();
    void buildAdaptiveChildRanges();
    void clearTileRanges();
    void countTileCoverage();
    void prefixGaussianTileRanges();
    void emitTileItems();
    void sortTileItems(uint32_t tileItemCount);
    void gatherHighTileKeys(uint32_t tileItemCount);
    void gatherSortedTileItems(uint32_t tileItemCount);
    void rebuildTileRanges(uint32_t tileItemCount);
    void ensureTileSortResources(uint32_t tileItemCount);
    void compositePixels();
    void bindAndDispatch(ComputePipeline& pipeline, vk::DescriptorSet descriptorSet,
                         uint32_t groupCountX, uint32_t groupCountY = 1, uint32_t groupCountZ = 1);
    void bindAndDispatchChildren(ComputePipeline& pipeline, vk::DescriptorSet descriptorSet);
    void bindAndDispatchPrefix(ComputePipeline& pipeline, vk::DescriptorSet descriptorSet,
                               const TrainingPrefixPushConstants& prefixConstants,
                               uint32_t groupCountX);
    void shaderMemoryBarrier();
    void tileSortScratchReuseBarrier();
    void shaderBufferBarrier(std::initializer_list<vk::DescriptorBufferInfo> buffers,
                             vk::AccessFlags dstAccessMask);
    void updateDescriptorSet(vk::DescriptorSet descriptorSet,
                             std::initializer_list<uint32_t> bindings);
    void writeProfilingTimestamp(TrainingGpuProfileStage stage, bool end);
    void writeSortTimestamp(TrainingSortTimestamp point);

    vk::Device device_ = nullptr;
    vk::PhysicalDevice physicalDevice_ = nullptr;
    vk::Queue computeQueue_ = nullptr;
    uint32_t computeQueueFamilyIndex_ = 0;
    uint32_t gaussianCount_ = 0;
    TrainingExtent extent_{};

    ComputePipeline projectPipeline_;
    ComputePipeline gaussianPrefixRangesPipeline_;
    ComputePipeline gaussianPrefixScratchPipeline_;
    ComputePipeline gaussianPrefixTopPipeline_;
    ComputePipeline gaussianPrefixAddScratchPipeline_;
    ComputePipeline gaussianPrefixAddRangesPipeline_;
    ComputePipeline tileGatherHighPipeline_;
    ComputePipeline tileGatherItemsPipeline_;
    std::array<TilePipelineVariant, 2> tileVariants_{};
    AdaptiveTilePipelines adaptivePipelines_{};
    VrdxSorter radixSorter_ = VK_NULL_HANDLE;

    vk::DescriptorPool descriptorPool_ = nullptr;
    vk::DescriptorSet projectDescriptorSet_ = nullptr;
    vk::DescriptorSet gaussianPrefixRangesDescriptorSet_ = nullptr;
    vk::DescriptorSet gaussianPrefixScratchDescriptorSet_ = nullptr;
    vk::DescriptorSet gaussianPrefixTopDescriptorSet_ = nullptr;
    vk::DescriptorSet gaussianPrefixAddScratchDescriptorSet_ = nullptr;
    vk::DescriptorSet gaussianPrefixAddRangesDescriptorSet_ = nullptr;
    vk::DescriptorSet tileGatherHighDescriptorSet_ = nullptr;
    vk::DescriptorSet tileGatherItemsDescriptorSet_ = nullptr;

    TrainingBuffers* trainingBuffers_ = nullptr;
    vk::CommandBuffer commandBuffer_ = nullptr;
    vk::QueryPool profilingQueryPool_ = nullptr;
    vk::QueryPool sortDiagnosticQueries_ = nullptr;
    TrainingPushConstants pushConstants_{};
    uint32_t adaptiveSplitThreshold_ = 256u;
    TrainingForwardCompositeMode compositeMode_ = TrainingForwardCompositeMode::WorkgroupShared;
    bool initialized_ = false;
    std::unique_ptr<TrainingFasterSort> fasterSort_;
    bool fasterSortRequested_ = false, fasterSortPrepared_ = false;
    int fasterSortLoggedState_ = -1;
};

} // namespace vulkan3DGS
