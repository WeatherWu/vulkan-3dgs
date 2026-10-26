#include "training/core/gaussian_forward_renderer.hpp"
#include "training/core/training_sort_synchronization.hpp"

#include "utils/logger.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace vulkan3DGS {

namespace {

bool fasterSortEnvironment() {
#ifdef _WIN32
    char* value = nullptr;
    size_t length = 0;
    if (_dupenv_s(&value, &length, "VULKAN_3DGS_FASTER_SORT") != 0 || !value) return false;
    const std::unique_ptr<char, decltype(&std::free)> owner(value, &std::free);
#else
    const char* value = std::getenv("VULKAN_3DGS_FASTER_SORT");
#endif
    if (!value) return false;
    if (std::strcmp(value, "0") && std::strcmp(value, "1"))
        throw std::runtime_error("VULKAN_3DGS_FASTER_SORT must be 0 or 1");
    return std::strcmp(value, "1") == 0;
}

vk::DescriptorSetLayoutBinding storageBinding(uint32_t binding) {
    vk::DescriptorSetLayoutBinding layoutBinding{};
    layoutBinding.setBinding(binding)
                 .setDescriptorType(vk::DescriptorType::eStorageBuffer)
                 .setDescriptorCount(1)
                 .setStageFlags(vk::ShaderStageFlagBits::eCompute);
    return layoutBinding;
}

vk::DescriptorSetLayoutBinding uniformBinding(uint32_t binding) {
    vk::DescriptorSetLayoutBinding layoutBinding{};
    layoutBinding.setBinding(binding)
                 .setDescriptorType(vk::DescriptorType::eUniformBuffer)
                 .setDescriptorCount(1)
                 .setStageFlags(vk::ShaderStageFlagBits::eCompute);
    return layoutBinding;
}

ComputePipelineConfig pipelineConfig(std::initializer_list<vk::DescriptorSetLayoutBinding> bindings) {
    ComputePipelineConfig config{};
    config.descriptorBindings.assign(bindings.begin(), bindings.end());
    config.pushConstantSize = sizeof(TrainingPushConstants);
    return config;
}

ComputePipelineConfig prefixPipelineConfig(
    std::initializer_list<vk::DescriptorSetLayoutBinding> bindings) {
    ComputePipelineConfig config{};
    config.descriptorBindings.assign(bindings.begin(), bindings.end());
    config.pushConstantSize = sizeof(TrainingPrefixPushConstants);
    return config;
}

uint32_t ceilDiv(uint32_t value, uint32_t divisor) {
    return (value + divisor - 1u) / divisor;
}

bool supportsShaderBufferFloat32AtomicMinMax(vk::PhysicalDevice physicalDevice) {
    auto extensions = physicalDevice.enumerateDeviceExtensionProperties();
    const bool hasAtomicFloat2 = std::any_of(extensions.begin(), extensions.end(),
        [](const vk::ExtensionProperties& extension) {
            return std::strcmp(extension.extensionName.data(), vk::EXTShaderAtomicFloat2ExtensionName) == 0;
        });
    if (!hasAtomicFloat2) {
        return false;
    }

    vk::PhysicalDeviceShaderAtomicFloat2FeaturesEXT atomicFloat2Features{};
    vk::PhysicalDeviceFeatures2 features2{};
    features2.setPNext(&atomicFloat2Features);
    physicalDevice.getFeatures2(&features2);
    return atomicFloat2Features.shaderBufferFloat32AtomicMinMax;
}

} // namespace

void GaussianForwardRenderer::initialize(vk::Device device,
                                         vk::PhysicalDevice physicalDevice,
                                         vk::Queue computeQueue,
                                         uint32_t computeQueueFamilyIndex,
                                         uint32_t gaussianCount,
                                         TrainingExtent extent) {
    device_ = device;
    physicalDevice_ = physicalDevice;
    computeQueue_ = computeQueue;
    computeQueueFamilyIndex_ = computeQueueFamilyIndex;
    gaussianCount_ = gaussianCount;
    extent_ = extent;
    fasterSortRequested_ = fasterSortEnvironment();
    createForwardResources();
    initialized_ = true;

    LOG_INFO("GaussianForwardRenderer initialized ({} gaussians, {}x{})",
             gaussianCount_, extent_.width, extent_.height);
}

void GaussianForwardRenderer::cleanup() {
    fasterSort_.reset();
    destroyForwardResources();
    device_ = nullptr;
    physicalDevice_ = nullptr;
    computeQueue_ = nullptr;
    computeQueueFamilyIndex_ = 0;
    gaussianCount_ = 0;
    extent_ = {};
    trainingBuffers_ = nullptr;
    commandBuffer_ = nullptr;
    profilingQueryPool_ = nullptr;
    sortDiagnosticQueries_ = nullptr;
    pushConstants_ = {};
    initialized_ = false;
    fasterSortRequested_ = fasterSortPrepared_ = false;
    fasterSortLoggedState_ = -1;
}

void GaussianForwardRenderer::setTrainingBuffers(TrainingBuffers& trainingBuffers,
                                                  vk::CommandBuffer commandBuffer,
                                                  TrainingPushConstants pushConstants) {
    trainingBuffers_ = &trainingBuffers;
    commandBuffer_ = commandBuffer;
    pushConstants_ = pushConstants;
}

void GaussianForwardRenderer::setProfilingQueryPool(vk::QueryPool queryPool) {
    profilingQueryPool_ = queryPool;
}

void GaussianForwardRenderer::forward() {
    if (fasterSortRequested_ && pushConstants_.tileSize == kTrainingTileSize16)
        throw std::logic_error("Faster sort requires separate prepare/fence/main submissions");
    if (!initialized_) {
        LOG_WARN("GaussianForwardRenderer::forward called before initialization");
        return;
    }
    if (!trainingBuffers_) {
        LOG_WARN("GaussianForwardRenderer::forward called without training buffers");
        return;
    }
    if (!commandBuffer_) {
        LOG_WARN("GaussianForwardRenderer::forward called without a command buffer");
        return;
    }

    prepareTileItems();
    renderPreparedTiles(pushConstants_.tileItemCount);
}

void GaussianForwardRenderer::prepareTileItems() {
    if (pushConstants_.tileSize == 32u) {
        adaptiveSplitThreshold_ = std::max(pushConstants_.tileItemCount, 1u);
    }
    writeProfilingTimestamp(TrainingGpuProfileStage::GaussianProjection, false);
    projectGaussians();
    writeProfilingTimestamp(TrainingGpuProfileStage::GaussianProjection, true);

    prepareProjectedTileItems();
}

void GaussianForwardRenderer::prepareProjectedTileItems() {
    writeSortTimestamp(TrainingSortTimestamp::CountBegin);
    writeProfilingTimestamp(TrainingGpuProfileStage::TileCoverageCount, false);
    clearTileRanges();
    countTileCoverage();
    writeProfilingTimestamp(TrainingGpuProfileStage::TileCoverageCount, true);
    writeSortTimestamp(TrainingSortTimestamp::CountEnd);

    writeProfilingTimestamp(TrainingGpuProfileStage::TilePrefix, false);
    prefixGaussianTileRanges();
    fasterSortPrepared_ = fasterSortRequested_ && pushConstants_.tileSize == kTrainingTileSize16;
    const int sortState = fasterSortRequested_ ? (fasterSortPrepared_ ? 2 : 1) : 0;
    if (fasterSortLoggedState_ != sortState) {
        LOG_INFO("Faster-GS sorting: requested={}, active={}, tile_size={}{}", fasterSortRequested_,
                 fasterSortPrepared_, pushConstants_.tileSize,
                 fasterSortRequested_ && !fasterSortPrepared_ ? " (original sorting fallback)" : "");
        fasterSortLoggedState_ = sortState;
    }
    if (fasterSortPrepared_) {
        if (!fasterSort_) fasterSort_ = std::make_unique<TrainingFasterSort>(device_, physicalDevice_, computeQueue_, computeQueueFamilyIndex_);
        fasterSort_->prepare(commandBuffer_, *trainingBuffers_, pushConstants_);
    }
    writeProfilingTimestamp(TrainingGpuProfileStage::TilePrefix, true);
    writeSortTimestamp(TrainingSortTimestamp::PrefixEnd);
}

void GaussianForwardRenderer::renderPreparedTiles(uint32_t tileItemCount) {
    buildPreparedTileLists(tileItemCount);
    writeProfilingTimestamp(TrainingGpuProfileStage::Composite, false);
    compositePixels();
    writeProfilingTimestamp(TrainingGpuProfileStage::Composite, true);
}

void GaussianForwardRenderer::emitPreparedTileItems(uint32_t tileItemCount) {
    pushConstants_.tileItemCount = tileItemCount;
    writeSortTimestamp(TrainingSortTimestamp::EmitBegin);
    writeProfilingTimestamp(TrainingGpuProfileStage::TileEmit, false);
    emitTileItems();
    writeProfilingTimestamp(TrainingGpuProfileStage::TileEmit, true);
    writeSortTimestamp(TrainingSortTimestamp::EmitEnd);
}

void GaussianForwardRenderer::buildPreparedTileLists(uint32_t tileItemCount) {
    if (fasterSortPrepared_) {
        pushConstants_.tileItemCount = tileItemCount;
        writeProfilingTimestamp(TrainingGpuProfileStage::TileSortAndRanges, false);
        fasterSort_->build(commandBuffer_, *trainingBuffers_, pushConstants_, radixSorter_);
        writeProfilingTimestamp(TrainingGpuProfileStage::TileSortAndRanges, true);
        writeSortTimestamp(TrainingSortTimestamp::RangesEnd);
        return;
    }
    emitPreparedTileItems(tileItemCount);
    writeProfilingTimestamp(TrainingGpuProfileStage::TileSortAndRanges, false);
    sortTileItems(tileItemCount);
    rebuildTileRanges(tileItemCount);
    writeProfilingTimestamp(TrainingGpuProfileStage::TileSortAndRanges, true);
    writeSortTimestamp(TrainingSortTimestamp::RangesEnd);
}

void GaussianForwardRenderer::writeSortTimestamp(TrainingSortTimestamp point) {
    if (sortDiagnosticQueries_ && commandBuffer_)
        commandBuffer_.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands,
                                      sortDiagnosticQueries_, uint32_t(point));
}

void GaussianForwardRenderer::writeProfilingTimestamp(TrainingGpuProfileStage stage, bool end) {
    if (profilingQueryPool_ && commandBuffer_) {
        commandBuffer_.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands,
                                      profilingQueryPool_,
                                      trainingGpuTimestampQuery(stage, end));
    }
}

void GaussianForwardRenderer::initializeTileVariant(TilePipelineVariant& variant,
                                                    uint32_t tileSize) {
    variant.tileSize = tileSize;
    const bool tile8 = tileSize == kTrainingTileSize8;
    variant.tileClearPipeline.initialize(
        device_,
        tile8 ? "shaders/train_forward_tile_clear_tile8.comp.spv"
              : "shaders/train_forward_tile_clear.comp.spv",
        pipelineConfig({storageBinding(5)}));
    variant.tileCountPipeline.initialize(
        device_,
        tile8 ? "shaders/train_forward_tile_count_tile8.comp.spv"
              : "shaders/train_forward_tile_count.comp.spv",
        pipelineConfig({storageBinding(3), storageBinding(16), storageBinding(32)}));
    variant.tileEmitPipeline.initialize(
        device_,
        tile8 ? "shaders/train_forward_tile_emit_tile8.comp.spv"
              : "shaders/train_forward_tile_emit.comp.spv",
        pipelineConfig({storageBinding(3), storageBinding(4), storageBinding(20),
                        storageBinding(21), storageBinding(22), storageBinding(32)}));
    variant.tileRangeBoundariesPipeline.initialize(
        device_,
        tile8 ? "shaders/train_forward_tile_range_boundaries_tile8.comp.spv"
              : "shaders/train_forward_tile_range_boundaries.comp.spv",
        pipelineConfig({storageBinding(5), storageBinding(21), storageBinding(22)}));
    variant.tileRangeBuildPipeline.initialize(
        device_,
        tile8 ? "shaders/train_forward_tile_sort_tile8.comp.spv"
              : "shaders/train_forward_tile_sort.comp.spv",
        pipelineConfig({storageBinding(5)}));
    variant.forwardPipeline.initialize(
        device_,
        tile8 ? "shaders/train_forward_tile8.comp.spv"
              : "shaders/train_forward.comp.spv",
        pipelineConfig({storageBinding(3), storageBinding(4), storageBinding(5),
                        storageBinding(6), storageBinding(14)}));
    variant.forwardWorkgroupPipeline.initialize(
        device_,
        tile8 ? "shaders/train_forward_workgroup_tile8.comp.spv"
              : "shaders/train_forward_workgroup.comp.spv",
        pipelineConfig({storageBinding(3), storageBinding(4), storageBinding(5),
                        storageBinding(6), storageBinding(14)}));
}

void GaussianForwardRenderer::cleanupTileVariant(TilePipelineVariant& variant) {
    variant.forwardWorkgroupPipeline.cleanup();
    variant.forwardPipeline.cleanup();
    variant.tileRangeBuildPipeline.cleanup();
    variant.tileRangeBoundariesPipeline.cleanup();
    variant.tileEmitPipeline.cleanup();
    variant.tileCountPipeline.cleanup();
    variant.tileClearPipeline.cleanup();
    variant.tileClearDescriptorSet = nullptr;
    variant.tileCountDescriptorSet = nullptr;
    variant.tileEmitDescriptorSet = nullptr;
    variant.tileRangeBoundariesDescriptorSet = nullptr;
    variant.tileRangeBuildDescriptorSet = nullptr;
    variant.forwardDescriptorSet = nullptr;
    variant.forwardWorkgroupDescriptorSet = nullptr;
}

void GaussianForwardRenderer::initializeAdaptivePipelines() {
    adaptivePipelines_.countChildren.initialize(
        device_, "shaders/train_forward_adaptive_child_count.comp.spv",
        pipelineConfig({storageBinding(3), storageBinding(5), storageBinding(24),
                        storageBinding(35), storageBinding(36)}));
    adaptivePipelines_.prefixChildren.initialize(
        device_, "shaders/train_forward_adaptive_child_prefix.comp.spv",
        pipelineConfig({storageBinding(5), storageBinding(35), storageBinding(38),
                        storageBinding(39)}));
    adaptivePipelines_.prefixChildBlocks.initialize(
        device_, "shaders/train_forward_adaptive_child_block_prefix.comp.spv",
        pipelineConfig({storageBinding(9), storageBinding(39)}));
    adaptivePipelines_.selectChildren.initialize(
        device_, "shaders/train_forward_adaptive_child_select.comp.spv",
        pipelineConfig({storageBinding(5), storageBinding(9), storageBinding(34),
                        storageBinding(35), storageBinding(37), storageBinding(38),
                        storageBinding(39), storageBinding(40)}));
    adaptivePipelines_.dispatchChildren.initialize(
        device_, "shaders/train_forward_adaptive_child_dispatch.comp.spv",
        pipelineConfig({storageBinding(9), storageBinding(41)}));
    adaptivePipelines_.scatterChildren.initialize(
        device_, "shaders/train_forward_adaptive_child_scatter.comp.spv",
        pipelineConfig({storageBinding(5), storageBinding(24), storageBinding(34),
                        storageBinding(36), storageBinding(37)}));
    adaptivePipelines_.buildParentBoundaries.initialize(
        device_, "shaders/train_forward_adaptive_parent_range_boundaries.comp.spv",
        pipelineConfig({storageBinding(5), storageBinding(21), storageBinding(22)}));
    adaptivePipelines_.clearLeafRanges.initialize(
        device_, "shaders/train_forward_tile_clear_adaptive.comp.spv",
        pipelineConfig({storageBinding(5)}));
    adaptivePipelines_.buildLeafRanges.initialize(
        device_, "shaders/train_forward_tile_sort_adaptive.comp.spv",
        pipelineConfig({storageBinding(5)}));
    adaptivePipelines_.installChildRanges.initialize(
        device_, "shaders/train_forward_adaptive_child_ranges.comp.spv",
        pipelineConfig({storageBinding(5), storageBinding(34), storageBinding(35),
                        storageBinding(37)}));
    adaptivePipelines_.forwardDirect.initialize(
        device_, "shaders/train_forward_adaptive.comp.spv",
        pipelineConfig({storageBinding(3), storageBinding(4), storageBinding(5), storageBinding(6),
                        storageBinding(14), storageBinding(34)}));
    for (uint32_t index = 0u; index < adaptivePipelines_.forwardWorkgroup.size(); ++index) {
        const uint32_t tileSize = index == 0u ? kTrainingTileSize8 : kTrainingTileSize16;
        adaptivePipelines_.forwardWorkgroup[index].initialize(
            device_, "shaders/train_forward_workgroup_adaptive_tile" +
                         std::to_string(tileSize) + ".comp.spv",
            pipelineConfig({storageBinding(3), storageBinding(4), storageBinding(5),
                            storageBinding(6), storageBinding(14), storageBinding(34),
                            storageBinding(40), storageBinding(41)}));
    }
}

void GaussianForwardRenderer::cleanupAdaptivePipelines() {
    for (uint32_t index = 0u; index < adaptivePipelines_.forwardWorkgroup.size(); ++index) {
        adaptivePipelines_.forwardWorkgroup[index].cleanup();
        adaptivePipelines_.forwardWorkgroupSets[index] = nullptr;
    }
    adaptivePipelines_.forwardDirect.cleanup();
    adaptivePipelines_.installChildRanges.cleanup();
    adaptivePipelines_.buildLeafRanges.cleanup();
    adaptivePipelines_.clearLeafRanges.cleanup();
    adaptivePipelines_.buildParentBoundaries.cleanup();
    adaptivePipelines_.scatterChildren.cleanup();
    adaptivePipelines_.selectChildren.cleanup();
    adaptivePipelines_.dispatchChildren.cleanup();
    adaptivePipelines_.prefixChildBlocks.cleanup();
    adaptivePipelines_.prefixChildren.cleanup();
    adaptivePipelines_.countChildren.cleanup();
    adaptivePipelines_.countChildrenSet = nullptr;
    adaptivePipelines_.prefixChildrenSet = nullptr;
    adaptivePipelines_.prefixChildBlocksSet = nullptr;
    adaptivePipelines_.selectChildrenSet = nullptr;
    adaptivePipelines_.dispatchChildrenSet = nullptr;
    adaptivePipelines_.scatterChildrenSet = nullptr;
    adaptivePipelines_.buildParentBoundariesSet = nullptr;
    adaptivePipelines_.clearLeafRangesSet = nullptr;
    adaptivePipelines_.buildLeafRangesSet = nullptr;
    adaptivePipelines_.installChildRangesSet = nullptr;
    adaptivePipelines_.forwardDirectSet = nullptr;
}

GaussianForwardRenderer::TilePipelineVariant& GaussianForwardRenderer::activeTileVariant() {
    return pushConstants_.tileSize == kTrainingTileSize8 ? tileVariants_[0] : tileVariants_[1];
}

const GaussianForwardRenderer::TilePipelineVariant&
GaussianForwardRenderer::activeTileVariant() const {
    return pushConstants_.tileSize == kTrainingTileSize8 ? tileVariants_[0]
                                                         : tileVariants_[1];
}

void GaussianForwardRenderer::createForwardResources() {
    projectPipeline_.initialize(device_, "shaders/train_forward_project.comp.spv",
                                 pipelineConfig({storageBinding(0), storageBinding(3), uniformBinding(11)}));
    initializeTileVariant(tileVariants_[0], kTrainingTileSize8);
    initializeTileVariant(tileVariants_[1], kTrainingTileSize16);
    initializeAdaptivePipelines();
    gaussianPrefixRangesPipeline_.initialize(
        device_, "shaders/train_forward_gaussian_prefix_ranges.comp.spv",
        prefixPipelineConfig({storageBinding(32), storageBinding(33)}));
    gaussianPrefixScratchPipeline_.initialize(
        device_, "shaders/train_forward_gaussian_prefix_scratch.comp.spv",
        prefixPipelineConfig({storageBinding(33)}));
    gaussianPrefixTopPipeline_.initialize(
        device_, "shaders/train_forward_gaussian_prefix_top.comp.spv",
        prefixPipelineConfig({storageBinding(9), storageBinding(33)}));
    gaussianPrefixAddScratchPipeline_.initialize(
        device_, "shaders/train_forward_gaussian_prefix_add_scratch.comp.spv",
        prefixPipelineConfig({storageBinding(33)}));
    gaussianPrefixAddRangesPipeline_.initialize(
        device_, "shaders/train_forward_gaussian_prefix_add_ranges.comp.spv",
        prefixPipelineConfig({storageBinding(32), storageBinding(33)}));
    tileGatherHighPipeline_.initialize(device_, "shaders/train_forward_tile_gather_high.comp.spv",
                                       pipelineConfig({storageBinding(21), storageBinding(22), storageBinding(23)}));
    tileGatherItemsPipeline_.initialize(device_, "shaders/train_forward_tile_gather_items.comp.spv",
                                        pipelineConfig({storageBinding(4), storageBinding(22), storageBinding(24)}));

    VrdxSorterCreateInfo sorterInfo{};
    sorterInfo.physicalDevice = physicalDevice_;
    sorterInfo.device = device_;
    sorterInfo.pipelineCache = VK_NULL_HANDLE;
    vrdxCreateSorter(&sorterInfo, &radixSorter_);

    std::array<vk::DescriptorPoolSize, 2> poolSizes{};
    poolSizes[0].setType(vk::DescriptorType::eStorageBuffer)
                .setDescriptorCount(224);
    poolSizes[1].setType(vk::DescriptorType::eUniformBuffer)
                .setDescriptorCount(1);

    vk::DescriptorPoolCreateInfo poolInfo{};
    poolInfo.setPoolSizeCount(static_cast<uint32_t>(poolSizes.size()))
            .setPPoolSizes(poolSizes.data())
            .setMaxSets(35);
    descriptorPool_ = device_.createDescriptorPool(poolInfo);

    std::vector<vk::DescriptorSetLayout> layouts;
    layouts.reserve(35u);
    layouts.push_back(projectPipeline_.getDescriptorSetLayout());
    layouts.push_back(gaussianPrefixRangesPipeline_.getDescriptorSetLayout());
    layouts.push_back(gaussianPrefixScratchPipeline_.getDescriptorSetLayout());
    layouts.push_back(gaussianPrefixTopPipeline_.getDescriptorSetLayout());
    layouts.push_back(gaussianPrefixAddScratchPipeline_.getDescriptorSetLayout());
    layouts.push_back(gaussianPrefixAddRangesPipeline_.getDescriptorSetLayout());
    layouts.push_back(tileGatherHighPipeline_.getDescriptorSetLayout());
    layouts.push_back(tileGatherItemsPipeline_.getDescriptorSetLayout());
    for (const TilePipelineVariant& variant : tileVariants_) {
        layouts.push_back(variant.tileClearPipeline.getDescriptorSetLayout());
        layouts.push_back(variant.tileCountPipeline.getDescriptorSetLayout());
        layouts.push_back(variant.tileEmitPipeline.getDescriptorSetLayout());
        layouts.push_back(variant.tileRangeBoundariesPipeline.getDescriptorSetLayout());
        layouts.push_back(variant.tileRangeBuildPipeline.getDescriptorSetLayout());
        layouts.push_back(variant.forwardPipeline.getDescriptorSetLayout());
        layouts.push_back(variant.forwardWorkgroupPipeline.getDescriptorSetLayout());
    }
    layouts.push_back(adaptivePipelines_.countChildren.getDescriptorSetLayout());
    layouts.push_back(adaptivePipelines_.prefixChildren.getDescriptorSetLayout());
    layouts.push_back(adaptivePipelines_.prefixChildBlocks.getDescriptorSetLayout());
    layouts.push_back(adaptivePipelines_.selectChildren.getDescriptorSetLayout());
    layouts.push_back(adaptivePipelines_.dispatchChildren.getDescriptorSetLayout());
    layouts.push_back(adaptivePipelines_.scatterChildren.getDescriptorSetLayout());
    layouts.push_back(adaptivePipelines_.buildParentBoundaries.getDescriptorSetLayout());
    layouts.push_back(adaptivePipelines_.clearLeafRanges.getDescriptorSetLayout());
    layouts.push_back(adaptivePipelines_.buildLeafRanges.getDescriptorSetLayout());
    layouts.push_back(adaptivePipelines_.installChildRanges.getDescriptorSetLayout());
    layouts.push_back(adaptivePipelines_.forwardDirect.getDescriptorSetLayout());
    for (const ComputePipeline& pipeline : adaptivePipelines_.forwardWorkgroup) {
        layouts.push_back(pipeline.getDescriptorSetLayout());
    }

    vk::DescriptorSetAllocateInfo allocInfo{};
    allocInfo.setDescriptorPool(descriptorPool_)
             .setDescriptorSetCount(static_cast<uint32_t>(layouts.size()))
             .setPSetLayouts(layouts.data());

    std::vector<vk::DescriptorSet> sets = device_.allocateDescriptorSets(allocInfo);
    size_t setIndex = 0u;
    projectDescriptorSet_ = sets[setIndex++];
    gaussianPrefixRangesDescriptorSet_ = sets[setIndex++];
    gaussianPrefixScratchDescriptorSet_ = sets[setIndex++];
    gaussianPrefixTopDescriptorSet_ = sets[setIndex++];
    gaussianPrefixAddScratchDescriptorSet_ = sets[setIndex++];
    gaussianPrefixAddRangesDescriptorSet_ = sets[setIndex++];
    tileGatherHighDescriptorSet_ = sets[setIndex++];
    tileGatherItemsDescriptorSet_ = sets[setIndex++];
    for (TilePipelineVariant& variant : tileVariants_) {
        variant.tileClearDescriptorSet = sets[setIndex++];
        variant.tileCountDescriptorSet = sets[setIndex++];
        variant.tileEmitDescriptorSet = sets[setIndex++];
        variant.tileRangeBoundariesDescriptorSet = sets[setIndex++];
        variant.tileRangeBuildDescriptorSet = sets[setIndex++];
        variant.forwardDescriptorSet = sets[setIndex++];
        variant.forwardWorkgroupDescriptorSet = sets[setIndex++];
    }
    adaptivePipelines_.countChildrenSet = sets[setIndex++];
    adaptivePipelines_.prefixChildrenSet = sets[setIndex++];
    adaptivePipelines_.prefixChildBlocksSet = sets[setIndex++];
    adaptivePipelines_.selectChildrenSet = sets[setIndex++];
    adaptivePipelines_.dispatchChildrenSet = sets[setIndex++];
    adaptivePipelines_.scatterChildrenSet = sets[setIndex++];
    adaptivePipelines_.buildParentBoundariesSet = sets[setIndex++];
    adaptivePipelines_.clearLeafRangesSet = sets[setIndex++];
    adaptivePipelines_.buildLeafRangesSet = sets[setIndex++];
    adaptivePipelines_.installChildRangesSet = sets[setIndex++];
    adaptivePipelines_.forwardDirectSet = sets[setIndex++];
    for (vk::DescriptorSet& descriptorSet : adaptivePipelines_.forwardWorkgroupSets) {
        descriptorSet = sets[setIndex++];
    }
}

void GaussianForwardRenderer::destroyForwardResources() {
    if (descriptorPool_) {
        device_.destroyDescriptorPool(descriptorPool_);
        descriptorPool_ = nullptr;
        projectDescriptorSet_ = nullptr;
        gaussianPrefixRangesDescriptorSet_ = nullptr;
        gaussianPrefixScratchDescriptorSet_ = nullptr;
        gaussianPrefixTopDescriptorSet_ = nullptr;
        gaussianPrefixAddScratchDescriptorSet_ = nullptr;
        gaussianPrefixAddRangesDescriptorSet_ = nullptr;
        tileGatherHighDescriptorSet_ = nullptr;
        tileGatherItemsDescriptorSet_ = nullptr;
        for (TilePipelineVariant& variant : tileVariants_) {
            variant.tileClearDescriptorSet = nullptr;
            variant.tileCountDescriptorSet = nullptr;
            variant.tileEmitDescriptorSet = nullptr;
            variant.tileRangeBoundariesDescriptorSet = nullptr;
            variant.tileRangeBuildDescriptorSet = nullptr;
            variant.forwardDescriptorSet = nullptr;
            variant.forwardWorkgroupDescriptorSet = nullptr;
        }
    }

    if (radixSorter_) {
        vrdxDestroySorter(radixSorter_);
        radixSorter_ = VK_NULL_HANDLE;
    }

    for (TilePipelineVariant& variant : tileVariants_) {
        cleanupTileVariant(variant);
    }
    cleanupAdaptivePipelines();
    tileGatherItemsPipeline_.cleanup();
    tileGatherHighPipeline_.cleanup();
    gaussianPrefixAddRangesPipeline_.cleanup();
    gaussianPrefixAddScratchPipeline_.cleanup();
    gaussianPrefixTopPipeline_.cleanup();
    gaussianPrefixScratchPipeline_.cleanup();
    gaussianPrefixRangesPipeline_.cleanup();
    projectPipeline_.cleanup();
}

void GaussianForwardRenderer::projectGaussians() {
    updateDescriptorSet(projectDescriptorSet_, {0, 3, 11});
    bindAndDispatch(projectPipeline_, projectDescriptorSet_, ceilDiv(pushConstants_.gaussianCount, 256u));
    shaderBufferBarrier({trainingBuffers_->projectedInfo()},
                        vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
}

void GaussianForwardRenderer::clearTileRanges() {
    if (pushConstants_.tileSize == 32u) {
        const uint32_t rangeCapacity = trainingAdaptiveRangeCapacity(extent_);
        updateDescriptorSet(adaptivePipelines_.clearLeafRangesSet, {5});
        bindAndDispatch(adaptivePipelines_.clearLeafRanges,
                        adaptivePipelines_.clearLeafRangesSet,
                        ceilDiv(rangeCapacity, 256u));
        shaderBufferBarrier({trainingBuffers_->tileRangesInfo()},
                            vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
        return;
    }
    TilePipelineVariant& variant = activeTileVariant();
    const TrainingTileLayout layout = makeTrainingTileLayout(extent_, variant.tileSize);
    updateDescriptorSet(variant.tileClearDescriptorSet, {5});
    bindAndDispatch(variant.tileClearPipeline, variant.tileClearDescriptorSet,
                    ceilDiv(layout.tileCount, 256u));
    shaderBufferBarrier({trainingBuffers_->tileRangesInfo()},
                        vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
}

void GaussianForwardRenderer::countTileCoverage() {
    TilePipelineVariant& variant = activeTileVariant();
    updateDescriptorSet(variant.tileCountDescriptorSet, {3, 16, 32});
    bindAndDispatch(variant.tileCountPipeline, variant.tileCountDescriptorSet,
                    ceilDiv(pushConstants_.gaussianCount, 256u));
    shaderBufferBarrier({trainingBuffers_->gaussianTileRangesInfo(),
                         trainingBuffers_->densificationStatesInfo()},
                        vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
}

void GaussianForwardRenderer::prefixGaussianTileRanges() {
    struct PrefixLevel {
        uint32_t offset = 0u;
        uint32_t count = 0u;
    };

    std::vector<PrefixLevel> levels;
    uint32_t levelCount = ceilDiv(pushConstants_.gaussianCount, 256u);
    uint32_t nextOffset = 0u;
    levels.push_back({nextOffset, levelCount});
    nextOffset += levelCount;

    updateDescriptorSet(gaussianPrefixRangesDescriptorSet_, {32, 33});
    TrainingPrefixPushConstants prefixConstants{};
    prefixConstants.elementCount = pushConstants_.gaussianCount;
    prefixConstants.outputOffset = levels[0].offset;
    bindAndDispatchPrefix(gaussianPrefixRangesPipeline_,
                          gaussianPrefixRangesDescriptorSet_,
                          prefixConstants,
                          levelCount);
    shaderBufferBarrier({trainingBuffers_->gaussianTileRangesInfo(),
                         trainingBuffers_->gaussianTilePrefixScratchInfo()},
                        vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);

    while (levels.back().count > 256u) {
        const PrefixLevel current = levels.back();
        const uint32_t parentCount = ceilDiv(current.count, 256u);
        const PrefixLevel parent{nextOffset, parentCount};
        levels.push_back(parent);
        nextOffset += parentCount;

        updateDescriptorSet(gaussianPrefixScratchDescriptorSet_, {33});
        prefixConstants = {};
        prefixConstants.elementCount = current.count;
        prefixConstants.inputOffset = current.offset;
        prefixConstants.outputOffset = parent.offset;
        bindAndDispatchPrefix(gaussianPrefixScratchPipeline_,
                              gaussianPrefixScratchDescriptorSet_,
                              prefixConstants,
                              parentCount);
        shaderBufferBarrier({trainingBuffers_->gaussianTilePrefixScratchInfo()},
                            vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
    }

    const PrefixLevel top = levels.back();
    updateDescriptorSet(gaussianPrefixTopDescriptorSet_, {9, 33});
    prefixConstants = {};
    prefixConstants.elementCount = top.count;
    prefixConstants.inputOffset = top.offset;
    bindAndDispatchPrefix(gaussianPrefixTopPipeline_,
                          gaussianPrefixTopDescriptorSet_,
                          prefixConstants,
                          1u);
    shaderBufferBarrier({trainingBuffers_->gaussianTilePrefixScratchInfo(),
                         trainingBuffers_->countersInfo()},
                        vk::AccessFlagBits::eShaderRead |
                            vk::AccessFlagBits::eShaderWrite |
                            vk::AccessFlagBits::eTransferRead);

    for (size_t levelIndex = levels.size(); levelIndex > 1u; --levelIndex) {
        const PrefixLevel child = levels[levelIndex - 2u];
        const PrefixLevel parent = levels[levelIndex - 1u];
        updateDescriptorSet(gaussianPrefixAddScratchDescriptorSet_, {33});
        prefixConstants = {};
        prefixConstants.elementCount = child.count;
        prefixConstants.inputOffset = child.offset;
        prefixConstants.parentOffset = parent.offset;
        bindAndDispatchPrefix(gaussianPrefixAddScratchPipeline_,
                              gaussianPrefixAddScratchDescriptorSet_,
                              prefixConstants,
                              ceilDiv(child.count, 256u));
        shaderBufferBarrier({trainingBuffers_->gaussianTilePrefixScratchInfo()},
                            vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
    }

    updateDescriptorSet(gaussianPrefixAddRangesDescriptorSet_, {32, 33});
    prefixConstants = {};
    prefixConstants.elementCount = pushConstants_.gaussianCount;
    prefixConstants.inputOffset = levels[0].offset;
    bindAndDispatchPrefix(gaussianPrefixAddRangesPipeline_,
                          gaussianPrefixAddRangesDescriptorSet_,
                          prefixConstants,
                          ceilDiv(pushConstants_.gaussianCount, 256u));
    shaderBufferBarrier({trainingBuffers_->gaussianTileRangesInfo(),
                         trainingBuffers_->countersInfo()},
                        vk::AccessFlagBits::eShaderRead |
                            vk::AccessFlagBits::eShaderWrite |
                            vk::AccessFlagBits::eTransferRead);
}

void GaussianForwardRenderer::emitTileItems() {
    TilePipelineVariant& variant = activeTileVariant();
    updateDescriptorSet(variant.tileEmitDescriptorSet, {3, 4, 20, 21, 22, 32});
    bindAndDispatch(variant.tileEmitPipeline, variant.tileEmitDescriptorSet,
                    ceilDiv(pushConstants_.gaussianCount, 256u));
    shaderBufferBarrier({trainingBuffers_->tileItemsUnsortedInfo(),
                         trainingBuffers_->tileKeyLowInfo(),
                         trainingBuffers_->tileKeyHighInfo(),
                         trainingBuffers_->tileSortIndicesInfo()},
                        vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
}

void GaussianForwardRenderer::sortTileItems(uint32_t tileItemCount) {
    if (tileItemCount == 0u) {
        for (auto point : {TrainingSortTimestamp::DepthEnd, TrainingSortTimestamp::HighGatherEnd,
                           TrainingSortTimestamp::TileEnd, TrainingSortTimestamp::ItemGatherEnd})
            writeSortTimestamp(point);
        return;
    }
    if (tileItemCount == 1u) {
        for (auto point : {TrainingSortTimestamp::DepthEnd, TrainingSortTimestamp::HighGatherEnd,
                           TrainingSortTimestamp::TileEnd})
            writeSortTimestamp(point);
        gatherSortedTileItems(tileItemCount);
        shaderBufferBarrier({trainingBuffers_->tileItemsInfo()},
                            vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
        writeSortTimestamp(TrainingSortTimestamp::ItemGatherEnd);
        return;
    }

    LOG_DEBUG("Preparing radix sort for {} training tile items", tileItemCount);
    ensureTileSortResources(tileItemCount);
    LOG_DEBUG("Training radix sort storage is ready for {} tile items", tileItemCount);
    LOG_DEBUG("Recording low-key training radix sort");
    tileSortScratchReuseBarrier();
    vrdxCmdSortKeyValue(commandBuffer_,
                        radixSorter_,
                        tileItemCount,
                        trainingBuffers_->tileKeyLowBuffer(),
                        0,
                        trainingBuffers_->tileSortIndicesBuffer(),
                        0,
                        trainingBuffers_->tileSortStorageBuffer(),
                        0,
                        VK_NULL_HANDLE,
                        0);
    LOG_DEBUG("Recorded low-key training radix sort");

    shaderMemoryBarrier();
    writeSortTimestamp(TrainingSortTimestamp::DepthEnd);
    gatherHighTileKeys(tileItemCount);
    shaderMemoryBarrier();
    writeSortTimestamp(TrainingSortTimestamp::HighGatherEnd);

    LOG_DEBUG("Recording high-key training radix sort");
    tileSortScratchReuseBarrier();
    vrdxCmdSortKeyValue(commandBuffer_,
                        radixSorter_,
                        tileItemCount,
                        trainingBuffers_->tileSortScratchBuffer(),
                        0,
                        trainingBuffers_->tileSortIndicesBuffer(),
                        0,
                        trainingBuffers_->tileSortStorageBuffer(),
                        0,
                        VK_NULL_HANDLE,
                        0);
    LOG_DEBUG("Recorded high-key training radix sort");

    shaderMemoryBarrier();
    writeSortTimestamp(TrainingSortTimestamp::TileEnd);
    gatherSortedTileItems(tileItemCount);
    shaderBufferBarrier({trainingBuffers_->tileItemsInfo()},
                        vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
    writeSortTimestamp(TrainingSortTimestamp::ItemGatherEnd);
}

void GaussianForwardRenderer::gatherHighTileKeys(uint32_t tileItemCount) {
    updateDescriptorSet(tileGatherHighDescriptorSet_, {21, 22, 23});
    bindAndDispatch(tileGatherHighPipeline_, tileGatherHighDescriptorSet_, ceilDiv(tileItemCount, 256u));
}

void GaussianForwardRenderer::gatherSortedTileItems(uint32_t tileItemCount) {
    updateDescriptorSet(tileGatherItemsDescriptorSet_, {4, 22, 24});
    bindAndDispatch(tileGatherItemsPipeline_, tileGatherItemsDescriptorSet_, ceilDiv(tileItemCount, 256u));
}

void GaussianForwardRenderer::rebuildTileRanges(uint32_t tileItemCount) {
    if (pushConstants_.tileSize == 32u) {
        updateDescriptorSet(adaptivePipelines_.buildParentBoundariesSet, {5, 21, 22});
        bindAndDispatch(adaptivePipelines_.buildParentBoundaries,
                        adaptivePipelines_.buildParentBoundariesSet,
                        ceilDiv(tileItemCount + 1u, 256u));
        shaderBufferBarrier({trainingBuffers_->tileRangesInfo()},
                            vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);

        const uint32_t rangeCapacity = trainingAdaptiveRangeCapacity(extent_);
        updateDescriptorSet(adaptivePipelines_.buildLeafRangesSet, {5});
        bindAndDispatch(adaptivePipelines_.buildLeafRanges,
                        adaptivePipelines_.buildLeafRangesSet,
                        ceilDiv(rangeCapacity, 256u));
        shaderBufferBarrier({trainingBuffers_->tileRangesInfo()},
                            vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
        buildAdaptiveChildRanges();
        return;
    }

    TilePipelineVariant& variant = activeTileVariant();
    const TrainingTileLayout layout = makeTrainingTileLayout(extent_, variant.tileSize);
    updateDescriptorSet(variant.tileRangeBoundariesDescriptorSet, {5, 21, 22});
    bindAndDispatch(variant.tileRangeBoundariesPipeline,
                    variant.tileRangeBoundariesDescriptorSet,
                    ceilDiv(tileItemCount + 1u, 256u));
    writeSortTimestamp(TrainingSortTimestamp::RangeBoundaryDispatchEnd);
    shaderBufferBarrier({trainingBuffers_->tileRangesInfo()},
                        vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
    writeSortTimestamp(TrainingSortTimestamp::RangeBoundaryBarrierEnd);

    updateDescriptorSet(variant.tileRangeBuildDescriptorSet, {5});
    bindAndDispatch(variant.tileRangeBuildPipeline, variant.tileRangeBuildDescriptorSet,
                    ceilDiv(layout.tileCount, 256u));
    writeSortTimestamp(TrainingSortTimestamp::RangeFinalizeDispatchEnd);
    shaderBufferBarrier({trainingBuffers_->tileRangesInfo()},
                        vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);

}

void GaussianForwardRenderer::buildAdaptiveChildRanges() {
    const TrainingTileLayout parentLayout = makeTrainingTileLayout(extent_, kTrainingTileSize16);
    const uint32_t parentItemCount = pushConstants_.tileItemCount;

    // Only child counting needs the configured coarse threshold. The other
    // passes use tileItemCount as the sorted parent-list length and append base.
    writeProfilingTimestamp(TrainingGpuProfileStage::AdaptiveChildCount, false);
    pushConstants_.tileItemCount = adaptiveSplitThreshold_;
    updateDescriptorSet(adaptivePipelines_.countChildrenSet, {3, 5, 24, 35, 36});
    bindAndDispatch(adaptivePipelines_.countChildren, adaptivePipelines_.countChildrenSet,
                    parentLayout.tileCountX, parentLayout.tileCountY);
    pushConstants_.tileItemCount = parentItemCount;
    shaderBufferBarrier(
        {trainingBuffers_->adaptiveChildCountsInfo(), trainingBuffers_->adaptiveChildMasksInfo()},
        vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
    writeProfilingTimestamp(TrainingGpuProfileStage::AdaptiveChildCount, true);

    writeProfilingTimestamp(TrainingGpuProfileStage::AdaptiveChildSelect, false);
    const uint32_t parentCount = parentLayout.tileCount;
    const uint32_t blockCount = (parentCount + 255u) / 256u;
    updateDescriptorSet(adaptivePipelines_.prefixChildrenSet, {5, 35, 38, 39});
    bindAndDispatch(adaptivePipelines_.prefixChildren,
                    adaptivePipelines_.prefixChildrenSet, blockCount);
    shaderBufferBarrier({trainingBuffers_->adaptiveChildPrefixInfo(),
                         trainingBuffers_->adaptiveChildBlockSumsInfo()},
                        vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);

    updateDescriptorSet(adaptivePipelines_.prefixChildBlocksSet, {9, 39});
    bindAndDispatch(adaptivePipelines_.prefixChildBlocks,
                    adaptivePipelines_.prefixChildBlocksSet, 1u);
    shaderBufferBarrier({trainingBuffers_->adaptiveChildBlockSumsInfo(),
                         trainingBuffers_->countersInfo()},
                        vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);

    updateDescriptorSet(adaptivePipelines_.selectChildrenSet,
                        {5, 9, 34, 35, 37, 38, 39, 40});
    bindAndDispatch(adaptivePipelines_.selectChildren,
                    adaptivePipelines_.selectChildrenSet,
                    ceilDiv(parentCount, 256u));
    shaderBufferBarrier({trainingBuffers_->adaptiveTileSplitFlagsInfo(),
                         trainingBuffers_->adaptiveChildOffsetsInfo(),
                         trainingBuffers_->adaptiveActiveParentsInfo(),
                         trainingBuffers_->countersInfo()},
                        vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite |
                            vk::AccessFlagBits::eTransferRead);
    updateDescriptorSet(adaptivePipelines_.dispatchChildrenSet, {9, 41});
    bindAndDispatch(adaptivePipelines_.dispatchChildren,
                    adaptivePipelines_.dispatchChildrenSet, 1u);
    shaderBufferBarrier({trainingBuffers_->adaptiveChildDispatchInfo()},
                        vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eIndirectCommandRead);
    writeProfilingTimestamp(TrainingGpuProfileStage::AdaptiveChildSelect, true);

    writeProfilingTimestamp(TrainingGpuProfileStage::AdaptiveChildScatter, false);
    updateDescriptorSet(adaptivePipelines_.scatterChildrenSet, {5, 24, 34, 36, 37});
    bindAndDispatch(adaptivePipelines_.scatterChildren, adaptivePipelines_.scatterChildrenSet,
                    parentLayout.tileCountX, parentLayout.tileCountY);
    shaderBufferBarrier({trainingBuffers_->tileItemsInfo()},
                        vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
    writeProfilingTimestamp(TrainingGpuProfileStage::AdaptiveChildScatter, true);

    writeProfilingTimestamp(TrainingGpuProfileStage::AdaptiveChildRangeInstall, false);
    updateDescriptorSet(adaptivePipelines_.installChildRangesSet, {5, 34, 35, 37});
    bindAndDispatch(adaptivePipelines_.installChildRanges,
                    adaptivePipelines_.installChildRangesSet,
                    ceilDiv(parentLayout.tileCount, 256u));
    shaderBufferBarrier({trainingBuffers_->tileRangesInfo()},
                        vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
    writeProfilingTimestamp(TrainingGpuProfileStage::AdaptiveChildRangeInstall, true);
}

void GaussianForwardRenderer::ensureTileSortResources(uint32_t tileItemCount) {
    if (!radixSorter_) {
        throw std::runtime_error("Training tile radix sorter is not initialized");
    }

    VrdxSorterStorageRequirements sorterRequirements{};
    vrdxGetSorterKeyValueStorageRequirements(radixSorter_, tileItemCount, &sorterRequirements);
    LOG_DEBUG("Training radix sort requires {} bytes with usage flags 0x{:x}",
              sorterRequirements.size,
              sorterRequirements.usage);
    trainingBuffers_->ensureTileSortStorage(tileItemCount,
                                            sorterRequirements.size,
                                            vk::BufferUsageFlags(sorterRequirements.usage));
}

void GaussianForwardRenderer::compositePixels() {
    if (pushConstants_.tileSize == 32u) {
        if (compositeMode_ == TrainingForwardCompositeMode::WorkgroupShared) {
            for (uint32_t index : {1u, 0u}) {
                const uint32_t tileSize = index == 0u ? kTrainingTileSize8 : kTrainingTileSize16;
                const vk::DescriptorSet descriptorSet =
                    adaptivePipelines_.forwardWorkgroupSets[index];
                updateDescriptorSet(descriptorSet, {3, 4, 5, 6, 14, 34, 40, 41});
                if (tileSize == kTrainingTileSize8) {
                    bindAndDispatchChildren(adaptivePipelines_.forwardWorkgroup[index], descriptorSet);
                } else {
                    bindAndDispatch(adaptivePipelines_.forwardWorkgroup[index], descriptorSet,
                                    ceilDiv(extent_.width, tileSize),
                                    ceilDiv(extent_.height, tileSize));
                }
            }
        } else {
            updateDescriptorSet(adaptivePipelines_.forwardDirectSet, {3, 4, 5, 6, 14, 34});
            bindAndDispatch(adaptivePipelines_.forwardDirect,
                            adaptivePipelines_.forwardDirectSet,
                            ceilDiv(extent_.width, kTrainingTileSize16),
                            ceilDiv(extent_.height, kTrainingTileSize16));
        }
        shaderBufferBarrier({trainingBuffers_->renderedColorInfo(),
                             trainingBuffers_->pixelBlendStatesInfo()},
                            vk::AccessFlagBits::eShaderRead |
                                vk::AccessFlagBits::eShaderWrite);
        return;
    }
    TilePipelineVariant& variant = activeTileVariant();
    ComputePipeline& pipeline = compositeMode_ == TrainingForwardCompositeMode::WorkgroupShared
        ? variant.forwardWorkgroupPipeline
        : variant.forwardPipeline;
    vk::DescriptorSet descriptorSet = compositeMode_ == TrainingForwardCompositeMode::WorkgroupShared
        ? variant.forwardWorkgroupDescriptorSet
        : variant.forwardDescriptorSet;
    updateDescriptorSet(descriptorSet, {3, 4, 5, 6, 14});
    bindAndDispatch(pipeline, descriptorSet,
                    ceilDiv(extent_.width, variant.tileSize),
                    ceilDiv(extent_.height, variant.tileSize));
    shaderBufferBarrier({trainingBuffers_->renderedColorInfo(),
                         trainingBuffers_->pixelBlendStatesInfo()},
                        vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
}

void GaussianForwardRenderer::bindAndDispatch(ComputePipeline& pipeline,
                                              vk::DescriptorSet descriptorSet,
                                              uint32_t groupCountX,
                                              uint32_t groupCountY,
                                              uint32_t groupCountZ) {
    if (groupCountX == 0 || groupCountY == 0 || groupCountZ == 0) {
        return;
    }

    commandBuffer_.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.getPipeline());
    commandBuffer_.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                      pipeline.getPipelineLayout(),
                                      0,
                                      1,
                                      &descriptorSet,
                                      0,
                                      nullptr);
    commandBuffer_.pushConstants(pipeline.getPipelineLayout(),
                                 vk::ShaderStageFlagBits::eCompute,
                                 0,
                                 sizeof(TrainingPushConstants),
                                 &pushConstants_);
    commandBuffer_.dispatch(groupCountX, groupCountY, groupCountZ);
}

void GaussianForwardRenderer::bindAndDispatchChildren(ComputePipeline& pipeline,
                                                       vk::DescriptorSet descriptorSet) {
    commandBuffer_.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.getPipeline());
    commandBuffer_.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                      pipeline.getPipelineLayout(), 0, descriptorSet, {});
    commandBuffer_.pushConstants(pipeline.getPipelineLayout(), vk::ShaderStageFlagBits::eCompute,
                                 0, sizeof(TrainingPushConstants), &pushConstants_);
    commandBuffer_.dispatchIndirect(trainingBuffers_->adaptiveChildDispatchBuffer(), 0u);
}

void GaussianForwardRenderer::bindAndDispatchPrefix(
    ComputePipeline& pipeline,
    vk::DescriptorSet descriptorSet,
    const TrainingPrefixPushConstants& prefixConstants,
    uint32_t groupCountX) {
    if (groupCountX == 0u) {
        return;
    }
    commandBuffer_.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.getPipeline());
    commandBuffer_.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                      pipeline.getPipelineLayout(),
                                      0,
                                      1,
                                      &descriptorSet,
                                      0,
                                      nullptr);
    commandBuffer_.pushConstants(pipeline.getPipelineLayout(),
                                 vk::ShaderStageFlagBits::eCompute,
                                 0,
                                 sizeof(TrainingPrefixPushConstants),
                                 &prefixConstants);
    commandBuffer_.dispatch(groupCountX, 1u, 1u);
}

void GaussianForwardRenderer::shaderBufferBarrier(std::initializer_list<vk::DescriptorBufferInfo> buffers,
                                                  vk::AccessFlags dstAccessMask) {
    std::vector<vk::BufferMemoryBarrier> barriers;
    barriers.reserve(buffers.size());
    for (const auto& buffer : buffers) {
        if (!buffer.buffer) {
            continue;
        }
        vk::BufferMemoryBarrier barrier{};
        barrier.setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
               .setDstAccessMask(dstAccessMask)
               .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
               .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
               .setBuffer(buffer.buffer)
               .setOffset(buffer.offset)
               .setSize(buffer.range);
        barriers.push_back(barrier);
    }

    if (barriers.empty()) {
        return;
    }

    vk::PipelineStageFlags dstStages = vk::PipelineStageFlagBits::eComputeShader;
    if (dstAccessMask & (vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite)) {
        dstStages |= vk::PipelineStageFlagBits::eTransfer;
    }
    if (dstAccessMask & vk::AccessFlagBits::eIndirectCommandRead) {
        dstStages |= vk::PipelineStageFlagBits::eDrawIndirect;
    }

    commandBuffer_.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                   dstStages,
                                   vk::DependencyFlagBits{},
                                   0,
                                   nullptr,
                                   static_cast<uint32_t>(barriers.size()),
                                   barriers.data(),
                                   0,
                                   nullptr);
}

void GaussianForwardRenderer::tileSortScratchReuseBarrier() {
    const auto barrier = trainingSortScratchReuseBarrier(trainingBuffers_->tileSortStorageBuffer());
    commandBuffer_.pipelineBarrier(kTrainingSortScratchReuseSourceStages,
                                   kTrainingSortScratchReuseDestinationStages, {},
                                   0, nullptr, 1, &barrier, 0, nullptr);
}

void GaussianForwardRenderer::shaderMemoryBarrier() {
    vk::MemoryBarrier barrier{};
    barrier.setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
           .setDstAccessMask(vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
    commandBuffer_.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                   vk::PipelineStageFlagBits::eComputeShader,
                                   vk::DependencyFlagBits{},
                                   1,
                                   &barrier,
                                   0,
                                   nullptr,
                                   0,
                                   nullptr);
}

void GaussianForwardRenderer::updateDescriptorSet(vk::DescriptorSet descriptorSet,
                                                  std::initializer_list<uint32_t> bindings) {
    auto gaussianParamsInfo = trainingBuffers_->gaussianParamsInfo();
    auto gaussianGradsInfo = trainingBuffers_->gaussianGradsInfo();
    auto projectedInfo = trainingBuffers_->projectedInfo();
    const bool tileEmitSet = std::any_of(
        tileVariants_.begin(), tileVariants_.end(),
        [descriptorSet](const TilePipelineVariant& variant) {
            return descriptorSet == variant.tileEmitDescriptorSet;
        });
    auto tileItemsInfo = (tileEmitSet || descriptorSet == tileGatherItemsDescriptorSet_)
        ? trainingBuffers_->tileItemsUnsortedInfo()
        : trainingBuffers_->tileItemsInfo();
    auto tileRangesInfo = trainingBuffers_->tileRangesInfo();
    auto renderedColorInfo = trainingBuffers_->renderedColorInfo();
    auto lossInfo = trainingBuffers_->lossInfo();
    auto countersInfo = trainingBuffers_->countersInfo();
    auto cameraInfo = trainingBuffers_->cameraInfo();
    auto pixelBlendStatesInfo = trainingBuffers_->pixelBlendStatesInfo();
    auto densificationStatesInfo = trainingBuffers_->densificationStatesInfo();

    std::array<vk::DescriptorBufferInfo, 42> infos{};
    infos[0] = gaussianParamsInfo;
    infos[1] = gaussianGradsInfo;
    infos[3] = projectedInfo;
    infos[4] = tileItemsInfo;
    infos[5] = tileRangesInfo;
    infos[6] = renderedColorInfo;
    infos[8] = lossInfo;
    infos[9] = countersInfo;
    infos[11] = cameraInfo;
    infos[14] = pixelBlendStatesInfo;
    infos[16] = densificationStatesInfo;
    infos[20] = trainingBuffers_->tileKeyLowInfo();
    infos[21] = trainingBuffers_->tileKeyHighInfo();
    infos[22] = trainingBuffers_->tileSortIndicesInfo();
    infos[23] = trainingBuffers_->tileSortScratchInfo();
    infos[24] = trainingBuffers_->tileItemsSortedInfo();
    infos[32] = trainingBuffers_->gaussianTileRangesInfo();
    infos[33] = trainingBuffers_->gaussianTilePrefixScratchInfo();
    infos[34] = trainingBuffers_->adaptiveTileSplitFlagsInfo();
    infos[35] = trainingBuffers_->adaptiveChildCountsInfo();
    infos[36] = trainingBuffers_->adaptiveChildMasksInfo();
    infos[37] = trainingBuffers_->adaptiveChildOffsetsInfo();
    infos[38] = trainingBuffers_->adaptiveChildPrefixInfo();
    infos[39] = trainingBuffers_->adaptiveChildBlockSumsInfo();
    infos[40] = trainingBuffers_->adaptiveActiveParentsInfo();
    infos[41] = trainingBuffers_->adaptiveChildDispatchInfo();

    std::array<vk::WriteDescriptorSet, 20> writes{};
    uint32_t writeCount = 0;
    for (uint32_t binding : bindings) {
        if (!infos[binding].buffer) {
            continue;
        }
        writes[writeCount]
            .setDstSet(descriptorSet)
            .setDstBinding(binding)
            .setDescriptorCount(1)
            .setDescriptorType(binding == 11 ? vk::DescriptorType::eUniformBuffer
                                             : vk::DescriptorType::eStorageBuffer)
            .setPBufferInfo(&infos[binding]);
        ++writeCount;
    }

    device_.updateDescriptorSets(writeCount, writes.data(), 0, nullptr);
}

} // namespace vulkan3DGS
