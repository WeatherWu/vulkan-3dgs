#pragma once

#include <initializer_list>
#include <vulkan/vulkan.hpp>

#include "gaussian_training/training_buffers.hpp"
#include "gaussian_training/training_types.hpp"
#include "renderer.hpp"
#include "vulkan/compute_pipeline.hpp"
#include <vk_radix_sort.h>

namespace vk_gs {

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
    void forward() override;
    void prepareTileItems();
    void renderPreparedTiles();

    bool isInitialized() const override { return initialized_; }

private:
    void createForwardResources();
    void destroyForwardResources();
    void clearForwardBuffers();
    void projectGaussians();
    void clearTileRanges();
    void countTileCoverage();
    void prefixTileRanges();
    void emitTileItems();
    void sortTileItems();
    void gatherHighTileKeys(uint32_t tileItemCount);
    void gatherSortedTileItems(uint32_t tileItemCount);
    void rebuildTileRanges();
    void ensureTileSortResources(uint32_t tileItemCount);
    void compositePixels();
    void bindAndDispatch(ComputePipeline& pipeline,
                         vk::DescriptorSet descriptorSet,
                         uint32_t groupCountX,
                         uint32_t groupCountY = 1,
                         uint32_t groupCountZ = 1);
    void shaderMemoryBarrier();
    void shaderBufferBarrier(std::initializer_list<vk::DescriptorBufferInfo> buffers,
                             vk::AccessFlags dstAccessMask);
    void updateDescriptorSet(vk::DescriptorSet descriptorSet,
                             std::initializer_list<uint32_t> bindings);

    vk::Device device_ = nullptr;
    vk::PhysicalDevice physicalDevice_ = nullptr;
    vk::Queue computeQueue_ = nullptr;
    uint32_t computeQueueFamilyIndex_ = 0;
    uint32_t gaussianCount_ = 0;
    TrainingExtent extent_{};

    ComputePipeline clearPipeline_;
    ComputePipeline projectPipeline_;
    ComputePipeline tileClearPipeline_;
    ComputePipeline tileCountPipeline_;
    ComputePipeline tilePrefixPipeline_;
    ComputePipeline tileEmitPipeline_;
    ComputePipeline tileGatherHighPipeline_;
    ComputePipeline tileGatherItemsPipeline_;
    ComputePipeline tileRangeBuildPipeline_;
    ComputePipeline forwardPipeline_;
    VrdxSorter radixSorter_ = VK_NULL_HANDLE;

    vk::DescriptorPool descriptorPool_ = nullptr;
    vk::DescriptorSet clearDescriptorSet_ = nullptr;
    vk::DescriptorSet projectDescriptorSet_ = nullptr;
    vk::DescriptorSet tileClearDescriptorSet_ = nullptr;
    vk::DescriptorSet tileCountDescriptorSet_ = nullptr;
    vk::DescriptorSet tilePrefixDescriptorSet_ = nullptr;
    vk::DescriptorSet tileEmitDescriptorSet_ = nullptr;
    vk::DescriptorSet tileGatherHighDescriptorSet_ = nullptr;
    vk::DescriptorSet tileGatherItemsDescriptorSet_ = nullptr;
    vk::DescriptorSet tileRangeBuildDescriptorSet_ = nullptr;
    vk::DescriptorSet forwardDescriptorSet_ = nullptr;

    TrainingBuffers* trainingBuffers_ = nullptr;
    vk::CommandBuffer commandBuffer_ = nullptr;
    TrainingPushConstants pushConstants_{};
    bool initialized_ = false;
};

} // namespace vk_gs
