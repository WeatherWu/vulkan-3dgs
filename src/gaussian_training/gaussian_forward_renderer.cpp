#include "gaussian_training/gaussian_forward_renderer.hpp"

#include "utils/logger.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace vulkan3DGS {

namespace {

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

uint32_t ceilDiv(uint32_t value, uint32_t divisor) {
    return (value + divisor - 1u) / divisor;
}

uint32_t tileCount(TrainingExtent extent) {
    constexpr uint32_t tileSize = 16;
    return ceilDiv(extent.width, tileSize) * ceilDiv(extent.height, tileSize);
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
    createForwardResources();
    initialized_ = true;

    LOG_INFO("GaussianForwardRenderer initialized ({} gaussians, {}x{})",
             gaussianCount_, extent_.width, extent_.height);
}

void GaussianForwardRenderer::cleanup() {
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
    pushConstants_ = {};
    initialized_ = false;
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
    writeProfilingTimestamp(TrainingGpuProfileStage::PrepareTileItems, false);
    clearForwardBuffers();
    projectGaussians();
    clearTileRanges();
    countTileCoverage();
    prefixTileRanges();
    writeProfilingTimestamp(TrainingGpuProfileStage::PrepareTileItems, true);
}

void GaussianForwardRenderer::renderPreparedTiles(uint32_t tileItemCount) {
    pushConstants_.tileItemCount = tileItemCount;
    writeProfilingTimestamp(TrainingGpuProfileStage::TileEmit, false);
    emitTileItems();
    writeProfilingTimestamp(TrainingGpuProfileStage::TileEmit, true);

    writeProfilingTimestamp(TrainingGpuProfileStage::TileSortAndRanges, false);
    sortTileItems(tileItemCount);
    rebuildTileRanges();
    writeProfilingTimestamp(TrainingGpuProfileStage::TileSortAndRanges, true);

    writeProfilingTimestamp(TrainingGpuProfileStage::Composite, false);
    compositePixels();
    writeProfilingTimestamp(TrainingGpuProfileStage::Composite, true);
}

void GaussianForwardRenderer::writeProfilingTimestamp(TrainingGpuProfileStage stage, bool end) {
    if (profilingQueryPool_ && commandBuffer_) {
        commandBuffer_.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands,
                                      profilingQueryPool_,
                                      trainingGpuTimestampQuery(stage, end));
    }
}

void GaussianForwardRenderer::createForwardResources() {
    clearPipeline_.initialize(device_, "shaders/train_clear.comp.spv",
                              pipelineConfig({storageBinding(1), storageBinding(6), storageBinding(8), storageBinding(9), storageBinding(14), storageBinding(15)}));
    projectPipeline_.initialize(device_, "shaders/train_forward_project.comp.spv",
                                 pipelineConfig({storageBinding(0), storageBinding(3), uniformBinding(11), storageBinding(16)}));
    tileClearPipeline_.initialize(device_, "shaders/train_forward_tile_clear.comp.spv",
                                  pipelineConfig({storageBinding(5)}));
    tileCountPipeline_.initialize(device_, "shaders/train_forward_tile_count.comp.spv",
                                  pipelineConfig({storageBinding(3), storageBinding(5)}));
    tilePrefixPipeline_.initialize(device_, "shaders/train_forward_tile_prefix.comp.spv",
                                   pipelineConfig({storageBinding(5), storageBinding(9)}));
    tileEmitPipeline_.initialize(device_, "shaders/train_forward_tile_emit.comp.spv",
                                 pipelineConfig({storageBinding(3), storageBinding(4), storageBinding(5), storageBinding(20), storageBinding(21), storageBinding(22)}));
    tileGatherHighPipeline_.initialize(device_, "shaders/train_forward_tile_gather_high.comp.spv",
                                       pipelineConfig({storageBinding(21), storageBinding(22), storageBinding(23)}));
    tileGatherItemsPipeline_.initialize(device_, "shaders/train_forward_tile_gather_items.comp.spv",
                                        pipelineConfig({storageBinding(4), storageBinding(22), storageBinding(24)}));
    tileRangeBuildPipeline_.initialize(device_, "shaders/train_forward_tile_sort.comp.spv",
                                       pipelineConfig({storageBinding(4), storageBinding(5), storageBinding(21), storageBinding(22)}));
    const bool useUintRadiusFallback = !supportsShaderBufferFloat32AtomicMinMax(physicalDevice_);
    const char* forwardShader = useUintRadiusFallback
        ? "shaders/train_forward_uint_radius.comp.spv"
        : "shaders/train_forward.comp.spv";
    if (useUintRadiusFallback) {
        LOG_INFO("Using uint maxScreenRadius fallback training forward shader");
    }

    forwardPipeline_.initialize(device_, forwardShader,
                                pipelineConfig({storageBinding(3), storageBinding(4), storageBinding(5), storageBinding(6), storageBinding(14), storageBinding(15)}));

    VrdxSorterCreateInfo sorterInfo{};
    sorterInfo.physicalDevice = physicalDevice_;
    sorterInfo.device = device_;
    sorterInfo.pipelineCache = VK_NULL_HANDLE;
    vrdxCreateSorter(&sorterInfo, &radixSorter_);

    std::array<vk::DescriptorPoolSize, 2> poolSizes{};
    poolSizes[0].setType(vk::DescriptorType::eStorageBuffer)
                .setDescriptorCount(40);
    poolSizes[1].setType(vk::DescriptorType::eUniformBuffer)
                .setDescriptorCount(1);

    vk::DescriptorPoolCreateInfo poolInfo{};
    poolInfo.setPoolSizeCount(static_cast<uint32_t>(poolSizes.size()))
            .setPPoolSizes(poolSizes.data())
            .setMaxSets(10);
    descriptorPool_ = device_.createDescriptorPool(poolInfo);

    std::array<vk::DescriptorSetLayout, 10> layouts = {
        clearPipeline_.getDescriptorSetLayout(),
        projectPipeline_.getDescriptorSetLayout(),
        tileClearPipeline_.getDescriptorSetLayout(),
        tileCountPipeline_.getDescriptorSetLayout(),
        tilePrefixPipeline_.getDescriptorSetLayout(),
        tileEmitPipeline_.getDescriptorSetLayout(),
        tileGatherHighPipeline_.getDescriptorSetLayout(),
        tileGatherItemsPipeline_.getDescriptorSetLayout(),
        tileRangeBuildPipeline_.getDescriptorSetLayout(),
        forwardPipeline_.getDescriptorSetLayout(),
    };

    vk::DescriptorSetAllocateInfo allocInfo{};
    allocInfo.setDescriptorPool(descriptorPool_)
             .setDescriptorSetCount(static_cast<uint32_t>(layouts.size()))
             .setPSetLayouts(layouts.data());

    std::vector<vk::DescriptorSet> sets = device_.allocateDescriptorSets(allocInfo);
    clearDescriptorSet_ = sets[0];
    projectDescriptorSet_ = sets[1];
    tileClearDescriptorSet_ = sets[2];
    tileCountDescriptorSet_ = sets[3];
    tilePrefixDescriptorSet_ = sets[4];
    tileEmitDescriptorSet_ = sets[5];
    tileGatherHighDescriptorSet_ = sets[6];
    tileGatherItemsDescriptorSet_ = sets[7];
    tileRangeBuildDescriptorSet_ = sets[8];
    forwardDescriptorSet_ = sets[9];
}

void GaussianForwardRenderer::destroyForwardResources() {
    if (descriptorPool_) {
        device_.destroyDescriptorPool(descriptorPool_);
        descriptorPool_ = nullptr;
        clearDescriptorSet_ = nullptr;
        projectDescriptorSet_ = nullptr;
        tileClearDescriptorSet_ = nullptr;
        tileCountDescriptorSet_ = nullptr;
        tilePrefixDescriptorSet_ = nullptr;
        tileEmitDescriptorSet_ = nullptr;
        tileGatherHighDescriptorSet_ = nullptr;
        tileGatherItemsDescriptorSet_ = nullptr;
        tileRangeBuildDescriptorSet_ = nullptr;
        forwardDescriptorSet_ = nullptr;
    }

    if (radixSorter_) {
        vrdxDestroySorter(radixSorter_);
        radixSorter_ = VK_NULL_HANDLE;
    }

    forwardPipeline_.cleanup();
    tileRangeBuildPipeline_.cleanup();
    tileGatherItemsPipeline_.cleanup();
    tileGatherHighPipeline_.cleanup();
    tileEmitPipeline_.cleanup();
    tilePrefixPipeline_.cleanup();
    tileCountPipeline_.cleanup();
    tileClearPipeline_.cleanup();
    projectPipeline_.cleanup();
    clearPipeline_.cleanup();
}

void GaussianForwardRenderer::clearForwardBuffers() {
    updateDescriptorSet(clearDescriptorSet_, {1, 6, 8, 9, 14, 15});
    bindAndDispatch(clearPipeline_, clearDescriptorSet_, std::max(ceilDiv(pushConstants_.pixelCount, 256u),
                                                                  ceilDiv(pushConstants_.gaussianCount, 256u)));
    shaderBufferBarrier({trainingBuffers_->gaussianGradsInfo(),
                         trainingBuffers_->renderedColorInfo(),
                         trainingBuffers_->pixelBlendStatesInfo(),
                         trainingBuffers_->gaussianVisibilityInfo(),
                         trainingBuffers_->lossInfo(),
                         trainingBuffers_->countersInfo()},
                        vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
}

void GaussianForwardRenderer::projectGaussians() {
    updateDescriptorSet(projectDescriptorSet_, {0, 3, 11, 16});
    bindAndDispatch(projectPipeline_, projectDescriptorSet_, ceilDiv(pushConstants_.gaussianCount, 256u));
    shaderBufferBarrier({trainingBuffers_->projectedInfo(),
                         trainingBuffers_->densificationStatesInfo()},
                        vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
}

void GaussianForwardRenderer::clearTileRanges() {
    updateDescriptorSet(tileClearDescriptorSet_, {5});
    bindAndDispatch(tileClearPipeline_, tileClearDescriptorSet_, ceilDiv(tileCount(extent_), 256u));
    shaderBufferBarrier({trainingBuffers_->tileRangesInfo()},
                        vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
}

void GaussianForwardRenderer::countTileCoverage() {
    updateDescriptorSet(tileCountDescriptorSet_, {3, 5});
    bindAndDispatch(tileCountPipeline_, tileCountDescriptorSet_, ceilDiv(pushConstants_.gaussianCount, 256u));
    shaderBufferBarrier({trainingBuffers_->tileRangesInfo()},
                        vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
}

void GaussianForwardRenderer::prefixTileRanges() {
    updateDescriptorSet(tilePrefixDescriptorSet_, {5, 9});
    bindAndDispatch(tilePrefixPipeline_, tilePrefixDescriptorSet_, 1);
    shaderBufferBarrier({trainingBuffers_->tileRangesInfo(), trainingBuffers_->countersInfo()},
                        vk::AccessFlagBits::eShaderRead |
                            vk::AccessFlagBits::eShaderWrite |
                            vk::AccessFlagBits::eTransferRead);
}

void GaussianForwardRenderer::emitTileItems() {
    updateDescriptorSet(tileEmitDescriptorSet_, {3, 4, 5, 20, 21, 22});
    bindAndDispatch(tileEmitPipeline_, tileEmitDescriptorSet_, ceilDiv(pushConstants_.gaussianCount, 256u));
    shaderBufferBarrier({trainingBuffers_->tileItemsUnsortedInfo(),
                         trainingBuffers_->tileKeyLowInfo(),
                         trainingBuffers_->tileKeyHighInfo(),
                         trainingBuffers_->tileSortIndicesInfo(),
                         trainingBuffers_->tileRangesInfo()},
                        vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
}

void GaussianForwardRenderer::sortTileItems(uint32_t tileItemCount) {
    if (tileItemCount <= 1) {
        return;
    }

    LOG_DEBUG("Preparing radix sort for {} training tile items", tileItemCount);
    ensureTileSortResources(tileItemCount);
    LOG_DEBUG("Training radix sort storage is ready for {} tile items", tileItemCount);
    LOG_DEBUG("Recording low-key training radix sort");
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
    gatherHighTileKeys(tileItemCount);
    shaderMemoryBarrier();

    LOG_DEBUG("Recording high-key training radix sort");
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
    gatherSortedTileItems(tileItemCount);
    shaderBufferBarrier({trainingBuffers_->tileItemsInfo()},
                        vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
}

void GaussianForwardRenderer::gatherHighTileKeys(uint32_t tileItemCount) {
    updateDescriptorSet(tileGatherHighDescriptorSet_, {21, 22, 23});
    bindAndDispatch(tileGatherHighPipeline_, tileGatherHighDescriptorSet_, ceilDiv(tileItemCount, 256u));
}

void GaussianForwardRenderer::gatherSortedTileItems(uint32_t tileItemCount) {
    updateDescriptorSet(tileGatherItemsDescriptorSet_, {4, 22, 24});
    bindAndDispatch(tileGatherItemsPipeline_, tileGatherItemsDescriptorSet_, ceilDiv(tileItemCount, 256u));
}

void GaussianForwardRenderer::rebuildTileRanges() {
    updateDescriptorSet(tileRangeBuildDescriptorSet_, {4, 5, 21, 22});
    bindAndDispatch(tileRangeBuildPipeline_, tileRangeBuildDescriptorSet_, ceilDiv(tileCount(extent_), 256u));
    shaderBufferBarrier({trainingBuffers_->tileRangesInfo()},
                        vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
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
    updateDescriptorSet(forwardDescriptorSet_, {3, 4, 5, 6, 14, 15});
    bindAndDispatch(forwardPipeline_, forwardDescriptorSet_, ceilDiv(extent_.width, 16u), ceilDiv(extent_.height, 16u));
    shaderBufferBarrier({trainingBuffers_->renderedColorInfo(),
                         trainingBuffers_->pixelBlendStatesInfo(),
                         trainingBuffers_->gaussianVisibilityInfo()},
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
    auto tileItemsInfo = (descriptorSet == tileEmitDescriptorSet_ ||
                          descriptorSet == tileGatherItemsDescriptorSet_)
        ? trainingBuffers_->tileItemsUnsortedInfo()
        : trainingBuffers_->tileItemsInfo();
    auto tileRangesInfo = trainingBuffers_->tileRangesInfo();
    auto renderedColorInfo = trainingBuffers_->renderedColorInfo();
    auto lossInfo = trainingBuffers_->lossInfo();
    auto countersInfo = trainingBuffers_->countersInfo();
    auto cameraInfo = trainingBuffers_->cameraInfo();
    auto pixelBlendStatesInfo = trainingBuffers_->pixelBlendStatesInfo();
    auto gaussianVisibilityInfo = trainingBuffers_->gaussianVisibilityInfo();
    auto densificationStatesInfo = trainingBuffers_->densificationStatesInfo();

    std::array<vk::DescriptorBufferInfo, 25> infos{};
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
    infos[15] = gaussianVisibilityInfo;
    infos[16] = densificationStatesInfo;
    infos[20] = trainingBuffers_->tileKeyLowInfo();
    infos[21] = trainingBuffers_->tileKeyHighInfo();
    infos[22] = trainingBuffers_->tileSortIndicesInfo();
    infos[23] = trainingBuffers_->tileSortScratchInfo();
    infos[24] = trainingBuffers_->tileItemsSortedInfo();

    std::array<vk::WriteDescriptorSet, 16> writes{};
    uint32_t writeCount = 0;
    for (uint32_t binding : bindings) {
        if (!infos[binding].buffer) {
            continue;
        }
        writes[writeCount].setDstSet(descriptorSet)
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
