#pragma once

#include <initializer_list>
#include <vulkan/vulkan.hpp>

#include "training/core/training_buffers.hpp"
#include "training/core/training_types.hpp"
#include "render/renderer.hpp"
#include "vulkan/compute_pipeline.hpp"
#include <vk_radix_sort.h>

namespace vulkan3DGS {

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

    bool isInitialized() const override { return initialized_; }

private:
    void createForwardResources();
    void destroyForwardResources();
    void projectGaussians();
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
    void bindAndDispatch(ComputePipeline& pipeline,
                         vk::DescriptorSet descriptorSet,
                         uint32_t groupCountX,
                         uint32_t groupCountY = 1,
                         uint32_t groupCountZ = 1);
    void bindAndDispatchPrefix(ComputePipeline& pipeline,
                               vk::DescriptorSet descriptorSet,
                               const TrainingPrefixPushConstants& prefixConstants,
                               uint32_t groupCountX);
    void shaderMemoryBarrier();
    void shaderBufferBarrier(std::initializer_list<vk::DescriptorBufferInfo> buffers,
                             vk::AccessFlags dstAccessMask);
    void updateDescriptorSet(vk::DescriptorSet descriptorSet,
                             std::initializer_list<uint32_t> bindings);
    void writeProfilingTimestamp(TrainingGpuProfileStage stage, bool end);

    vk::Device device_ = nullptr;
    vk::PhysicalDevice physicalDevice_ = nullptr;
    vk::Queue computeQueue_ = nullptr;
    uint32_t computeQueueFamilyIndex_ = 0;
    uint32_t gaussianCount_ = 0;
    TrainingExtent extent_{};

    ComputePipeline projectPipeline_;
    ComputePipeline tileClearPipeline_;
    ComputePipeline tileCountPipeline_;
    ComputePipeline gaussianPrefixRangesPipeline_;
    ComputePipeline gaussianPrefixScratchPipeline_;
    ComputePipeline gaussianPrefixTopPipeline_;
    ComputePipeline gaussianPrefixAddScratchPipeline_;
    ComputePipeline gaussianPrefixAddRangesPipeline_;
    ComputePipeline tileEmitPipeline_;
    ComputePipeline tileGatherHighPipeline_;
    ComputePipeline tileGatherItemsPipeline_;
    ComputePipeline tileRangeBoundariesPipeline_;
    ComputePipeline tileRangeBuildPipeline_;
    ComputePipeline forwardPipeline_;
    ComputePipeline forwardWorkgroupPipeline_;
    VrdxSorter radixSorter_ = VK_NULL_HANDLE;

    vk::DescriptorPool descriptorPool_ = nullptr;
    vk::DescriptorSet projectDescriptorSet_ = nullptr;
    vk::DescriptorSet tileClearDescriptorSet_ = nullptr;
    vk::DescriptorSet tileCountDescriptorSet_ = nullptr;
    vk::DescriptorSet gaussianPrefixRangesDescriptorSet_ = nullptr;
    vk::DescriptorSet gaussianPrefixScratchDescriptorSet_ = nullptr;
    vk::DescriptorSet gaussianPrefixTopDescriptorSet_ = nullptr;
    vk::DescriptorSet gaussianPrefixAddScratchDescriptorSet_ = nullptr;
    vk::DescriptorSet gaussianPrefixAddRangesDescriptorSet_ = nullptr;
    vk::DescriptorSet tileEmitDescriptorSet_ = nullptr;
    vk::DescriptorSet tileGatherHighDescriptorSet_ = nullptr;
    vk::DescriptorSet tileGatherItemsDescriptorSet_ = nullptr;
    vk::DescriptorSet tileRangeBoundariesDescriptorSet_ = nullptr;
    vk::DescriptorSet tileRangeBuildDescriptorSet_ = nullptr;
    vk::DescriptorSet forwardDescriptorSet_ = nullptr;
    vk::DescriptorSet forwardWorkgroupDescriptorSet_ = nullptr;

    TrainingBuffers* trainingBuffers_ = nullptr;
    vk::CommandBuffer commandBuffer_ = nullptr;
    vk::QueryPool profilingQueryPool_ = nullptr;
    TrainingPushConstants pushConstants_{};
    TrainingForwardCompositeMode compositeMode_ = TrainingForwardCompositeMode::WorkgroupShared;
    bool initialized_ = false;
};

} // namespace vulkan3DGS
