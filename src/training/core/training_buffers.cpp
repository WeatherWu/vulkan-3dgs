#include "training/core/training_buffers.hpp"
#include "training/core/training_tile_layout.hpp"
#include "vulkan/radix_key_value.hpp"

#include "utils/logger.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <glm/glm.hpp>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace vulkan3DGS {

namespace {

uint32_t roundedGaussianCapacity(uint32_t requestedCapacity) {
    constexpr uint32_t capacityChunk = 16384u;
    const uint64_t requested = std::max<uint64_t>(requestedCapacity, 1u);
    const uint64_t rounded = ((requested + capacityChunk - 1u) / capacityChunk) * capacityChunk;
    return static_cast<uint32_t>(std::min<uint64_t>(rounded,
                                                    std::numeric_limits<uint32_t>::max()));
}

uint32_t gaussianPrefixScratchElementCount(uint32_t gaussianCapacity) {
    uint64_t total = 0u;
    uint64_t levelCount = (std::max<uint64_t>(gaussianCapacity, 1u) + 255u) / 256u;
    while (true) {
        total += levelCount;
        if (levelCount <= 256u) {
            break;
        }
        levelCount = (levelCount + 255u) / 256u;
    }
    return static_cast<uint32_t>(std::min<uint64_t>(total,
                                                    std::numeric_limits<uint32_t>::max()));
}

} // namespace

void TrainingBuffers::initialize(vk::Device device, vk::PhysicalDevice physicalDevice,
                                 vk::Queue transferQueue, uint32_t transferQueueFamilyIndex,
                                 uint32_t computeQueueFamilyIndex, bool storage16Enabled) {
    storage16Enabled_ = storage16Enabled;
    device_ = device;
    physicalDevice_ = physicalDevice;
    transferQueue_ = transferQueue;
    transferQueueFamilyIndex_ = transferQueueFamilyIndex;
    computeQueueFamilyIndex_ = computeQueueFamilyIndex;
    try {
        tileItemCountReadback_.create(
            device_,
            physicalDevice_,
            transferQueue_,
            transferQueueFamilyIndex_,
            nullptr,
            sizeof(glm::uvec4),
            vk::BufferUsageFlagBits::eTransferDst,
            vk::MemoryPropertyFlagBits::eHostVisible |
                vk::MemoryPropertyFlagBits::eHostCoherent,
            {transferQueueFamilyIndex_, computeQueueFamilyIndex_});
        tileItemCountReadbackMapped_ = device_.mapMemory(
            tileItemCountReadback_.getMemory(), 0, sizeof(glm::uvec4));
        densificationStatsReadback_.create(
            device_,
            physicalDevice_,
            transferQueue_,
            transferQueueFamilyIndex_,
            nullptr,
            sizeof(uint32_t) * 16u,
            vk::BufferUsageFlagBits::eTransferDst,
            vk::MemoryPropertyFlagBits::eHostVisible |
                vk::MemoryPropertyFlagBits::eHostCoherent,
            {transferQueueFamilyIndex_, computeQueueFamilyIndex_});
        densificationStatsReadbackMapped_ = device_.mapMemory(
            densificationStatsReadback_.getMemory(), 0, sizeof(uint32_t) * 16u);
    } catch (...) {
        if (densificationStatsReadbackMapped_ && densificationStatsReadback_.getMemory()) {
            device_.unmapMemory(densificationStatsReadback_.getMemory());
        }
        densificationStatsReadbackMapped_ = nullptr;
        densificationStatsReadback_.cleanup();
        if (tileItemCountReadbackMapped_ && tileItemCountReadback_.getMemory()) {
            device_.unmapMemory(tileItemCountReadback_.getMemory());
        }
        tileItemCountReadback_.cleanup();
        tileItemCountReadbackMapped_ = nullptr;
        throw;
    }
}

void TrainingBuffers::ensureCheckpointCapacity(uint32_t tileItemCount) {
    const auto allocationStart = std::chrono::steady_clock::now();
    auto capacity = makeTrainingCheckpointCapacity(tileItemCount, extent_.width, extent_.height);
    const auto limits = physicalDevice_.getProperties().limits;
    if (!capacity.fitsDescriptorRange(limits.maxStorageBufferRange) ||
        capacity.slotCapacity * kCheckpointTilePixels > std::numeric_limits<uint32_t>::max())
        throw std::runtime_error("Checkpoint workspace exceeds descriptor/index limits");
    if (checkpoint_ && checkpoint_->capacity.tileCount == capacity.tileCount &&
        checkpoint_->capacity.slotCapacity >= capacity.slotCapacity &&
        checkpoint_->capacity.taskCapacity >= capacity.taskCapacity) return;
    if (checkpoint_) {
        capacity.slotCapacity = std::max(capacity.slotCapacity, checkpoint_->capacity.slotCapacity);
        capacity.taskCapacity = std::max(capacity.taskCapacity, checkpoint_->capacity.taskCapacity);
        capacity.bankElements = capacity.slotCapacity * kCheckpointBankPixels;
        capacity.bankBytes = capacity.bankElements * 16u;
        capacity.taskBytes = capacity.taskCapacity * sizeof(TrainingCheckpointTask);
    }
    const uint64_t scratchBytes =
        uint64_t(checkpointScratchElements(static_cast<uint32_t>(capacity.tileCount))) * 8u;
    const uint64_t bytes = std::max<uint64_t>(capacity.bankBytes, 16u) * 4u +
                           std::max<uint64_t>(capacity.metadataBytes, 32u) +
                           std::max<uint64_t>(capacity.taskBytes, 16u) +
                           std::max<uint64_t>(scratchBytes, 8u) +
                           2u * sizeof(TrainingCheckpointControl);
    // Account for existing allocations (including the old workspace during a
    // transactional replacement) through the driver's current heap usage.
    bool supportsBudget = false;
    for (const auto& extension : physicalDevice_.enumerateDeviceExtensionProperties())
        supportsBudget |= std::string_view(extension.extensionName.data()) ==
                          VK_EXT_MEMORY_BUDGET_EXTENSION_NAME;
    if (supportsBudget) {
        vk::PhysicalDeviceMemoryBudgetPropertiesEXT budget{};
        vk::PhysicalDeviceMemoryProperties2 memory{};
        memory.pNext = &budget;
        physicalDevice_.getMemoryProperties2(&memory);
        uint64_t bestAvailable = 0;
        for (uint32_t i = 0; i < memory.memoryProperties.memoryHeapCount; ++i) {
            if (!(memory.memoryProperties.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal))
                continue;
            const uint64_t free = budget.heapBudget[i] > budget.heapUsage[i]
                                      ? budget.heapBudget[i] - budget.heapUsage[i] : 0u;
            const uint64_t reserve = std::max<uint64_t>(512u * 1024u * 1024u,
                                                       budget.heapBudget[i] * 15u / 100u);
            bestAvailable = std::max(bestAvailable, free > reserve ? free - reserve : 0u);
        }
        if (bytes > bestAvailable)
            throw std::runtime_error("Checkpoint diagnostic workspace exceeds available VRAM budget");
    }
    auto next = std::make_unique<CheckpointWorkspace>();
    next->capacity = capacity;
    createStorageBuffer(next->metadata, std::max<uint64_t>(capacity.metadataBytes, 32u));
    createStorageBuffer(next->scratch, std::max<uint64_t>(scratchBytes, 8u));
    createStorageBuffer(next->tasks, std::max<uint64_t>(capacity.taskBytes, 16u));
    for (auto& bank : next->banks)
        createStorageBuffer(bank, std::max<uint64_t>(capacity.bankBytes, 16u));
    next->control.create(device_, physicalDevice_, transferQueue_, transferQueueFamilyIndex_, nullptr,
                         sizeof(TrainingCheckpointControl),
                         vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc |
                             vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndirectBuffer,
                         vk::MemoryPropertyFlagBits::eDeviceLocal,
                         {transferQueueFamilyIndex_, computeQueueFamilyIndex_});
    next->readback.create(device_, physicalDevice_, transferQueue_, transferQueueFamilyIndex_, nullptr,
                          sizeof(TrainingCheckpointControl), vk::BufferUsageFlagBits::eTransferDst,
                          vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                          {transferQueueFamilyIndex_, computeQueueFamilyIndex_});
    checkpoint_ = std::move(next);
    LOG_INFO("Checkpoint P1 diagnostic workspace: {} logical buffer bytes, {} slots, {} tasks (layout 2), CPU allocation {} ms",
             bytes, capacity.slotCapacity, capacity.taskCapacity,
             std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - allocationStart).count());
}

TrainingCheckpointResources TrainingBuffers::checkpointResources() const {
    if (!checkpoint_) throw std::runtime_error("Checkpoint diagnostic workspace is unavailable");
    TrainingCheckpointResources resources{};
    resources.capacity = checkpoint_->capacity;
    resources.descriptors = {projectedInfo(), tileItemsInfo(), tileRangesInfo(), pixelBlendStatesInfo(),
        checkpoint_->metadata.getDescriptorInfo(), checkpoint_->scratch.getDescriptorInfo(),
        checkpoint_->banks[0].getDescriptorInfo(), checkpoint_->banks[1].getDescriptorInfo(),
        checkpoint_->banks[2].getDescriptorInfo(), checkpoint_->banks[3].getDescriptorInfo(),
        checkpoint_->tasks.getDescriptorInfo(), checkpoint_->control.getDescriptorInfo()};
    return resources;
}

void TrainingBuffers::recordCheckpointReadback(vk::CommandBuffer command) {
    if (!checkpoint_) return;
    command.copyBuffer(checkpoint_->control.getBuffer(), checkpoint_->readback.getBuffer(),
                        vk::BufferCopy{0, 0, sizeof(TrainingCheckpointControl)});
    vk::MemoryBarrier memory{vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eHostRead};
    command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost,
                            {}, memory, {}, {});
}

TrainingCheckpointControl TrainingBuffers::checkpointDiagnostics() {
    TrainingCheckpointControl result{};
    if (checkpoint_) checkpoint_->readback.download(&result, sizeof(result));
    return result;
}

void TrainingBuffers::cleanup() {
    storage16Enabled_ = false;
    releaseCheckpointWorkspace();
    clearTargetColorDescriptor();
    if (densificationStatsReadbackMapped_ && densificationStatsReadback_.getMemory() && device_) {
        device_.unmapMemory(densificationStatsReadback_.getMemory());
    }
    densificationStatsReadbackMapped_ = nullptr;
    densificationStatsReadback_.cleanup();
    if (tileItemCountReadbackMapped_ && tileItemCountReadback_.getMemory() && device_) {
        device_.unmapMemory(tileItemCountReadback_.getMemory());
    }
    tileItemCountReadbackMapped_ = nullptr;
    tileItemCountReadback_.cleanup();
    camera_.cleanup();
    cameraIntrinsics_.cleanup();
    exactIntrinsics_ = false;
    previewInstances_.cleanup();
    counters_.cleanup();
    loss_.cleanup();
    validationFinalResult_.cleanup();
    densifiedGaussianValidationPartials_.cleanup();
    gaussianValidationPartials_.cleanup();
    pixelValidationPartials_.cleanup();
    projectedGrads_.cleanup();
    densificationCounters_.cleanup();
    densifiedAdamStates_.cleanup();
    densifiedParams_.cleanup();
    densificationStates_.cleanup();
    pixelBlendStates_.cleanup();
    pixelGrads_.cleanup();
    ssimBackwardStates_.cleanup();
    ssimScratch_.cleanup();
    targetColor_.cleanup();
    renderedColor_.cleanup();
    gaussianTilePrefixScratch_.cleanup();
    adaptiveTileSplitFlags_.cleanup();
    adaptiveChildCounts_.cleanup();
    adaptiveChildMasks_.cleanup();
    adaptiveChildOffsets_.cleanup();
    adaptiveChildPrefix_.cleanup();
    adaptiveChildBlockSums_.cleanup();
    adaptiveActiveParents_.cleanup();
    adaptiveChildDispatch_.cleanup();
    gaussianTileRanges_.cleanup();
    tileRanges_.cleanup();
    tileSortStorage_.cleanup();
    tileItemsSorted_.cleanup();
    tileSortScratch_.cleanup();
    tileSortIndices_.cleanup();
    tileKeyHigh_.cleanup();
    tileKeyLow_.cleanup();
    tileItems_.cleanup();
    projected_.cleanup();
    adamStates_.cleanup();
    gaussianGrads_.cleanup();
    gaussianParams_.cleanup();

    gaussianCapacity_ = 0;
    gaussianWorkspaceCapacity_ = 0;
    densificationCapacity_ = 0;
    tileItemCapacity_ = 0;
    extent_ = {};
    device_ = nullptr;
    physicalDevice_ = nullptr;
    transferQueue_ = nullptr;
    transferQueueFamilyIndex_ = 0;
    computeQueueFamilyIndex_ = 0;
}

void TrainingBuffers::resize(uint32_t gaussianCount, TrainingExtent extent) {
    if (extent.width != extent_.width || extent.height != extent_.height)
        releaseCheckpointWorkspace();
    clearTargetColorDescriptor();
    if (!device_) {
        throw std::runtime_error("TrainingBuffers must be initialized before resize");
    }

    const uint32_t safeGaussianCount = std::max(gaussianCount, 1u);
    const uint32_t initialTileCapacity = radixKeyValueCapacity(
        static_cast<uint32_t>(std::min<uint64_t>(uint64_t(safeGaussianCount) * 8u,
                                               std::numeric_limits<uint32_t>::max())),
        physicalDevice_.getProperties().limits.minStorageBufferOffsetAlignment);
    const uint32_t safeWidth = std::max(extent.width, 1u);
    const uint32_t safeHeight = std::max(extent.height, 1u);
    const uint64_t pixelCount = static_cast<uint64_t>(safeWidth) * static_cast<uint64_t>(safeHeight);
    const uint64_t parentTileCount = makeTrainingTileLayout(
        TrainingExtent{safeWidth, safeHeight}, kTrainingTileSize16).tileCount;
    const uint64_t tileRangeCapacity = trainingAdaptiveRangeCapacity(
        TrainingExtent{safeWidth, safeHeight});
    if (parentTileCount * 4u > std::numeric_limits<uint32_t>::max() ||
        parentTileCount * 4u > static_cast<uint64_t>(kAdaptiveTrainingChildDispatchWidth) *
                                   kAdaptiveTrainingChildDispatchWidth) {
        throw std::runtime_error("Adaptive child dispatch exceeds the supported workgroup grid");
    }

    gaussianParams_.cleanup();
    gaussianGrads_.cleanup();
    adamStates_.cleanup();
    projected_.cleanup();
    tileItems_.cleanup();
    tileKeyLow_.cleanup();
    tileKeyHigh_.cleanup();
    tileSortIndices_.cleanup();
    tileSortScratch_.cleanup();
    tileItemsSorted_.cleanup();
    tileSortStorage_.cleanup();
    tileRanges_.cleanup();
    gaussianTileRanges_.cleanup();
    gaussianTilePrefixScratch_.cleanup();
    adaptiveChildCounts_.cleanup();
    adaptiveTileSplitFlags_.cleanup();
    adaptiveChildMasks_.cleanup();
    adaptiveChildOffsets_.cleanup();
    adaptiveChildPrefix_.cleanup();
    adaptiveChildBlockSums_.cleanup();
    adaptiveActiveParents_.cleanup();
    adaptiveChildDispatch_.cleanup();
    renderedColor_.cleanup();
    targetColor_.cleanup();
    pixelGrads_.cleanup();
    ssimBackwardStates_.cleanup();
    ssimScratch_.cleanup();
    pixelValidationPartials_.cleanup();
    gaussianValidationPartials_.cleanup();
    densifiedGaussianValidationPartials_.cleanup();
    validationFinalResult_.cleanup();
    pixelBlendStates_.cleanup();
    projectedGrads_.cleanup();
    densificationCounters_.cleanup();
    densifiedAdamStates_.cleanup();
    densifiedParams_.cleanup();
    densificationStates_.cleanup();
    loss_.cleanup();
    counters_.cleanup();
    previewInstances_.cleanup();
    camera_.cleanup();
    cameraIntrinsics_.cleanup();
    exactIntrinsics_ = false;

    createStorageBuffer(gaussianParams_, sizeof(GaussianTrainParam) * safeGaussianCount);
    createStorageBuffer(gaussianGrads_, sizeof(GaussianGrad) * safeGaussianCount);
    std::vector<AdamState> zeroAdamStates(safeGaussianCount);
    adamStates_.create(device_, physicalDevice_, transferQueue_, transferQueueFamilyIndex_,
                       zeroAdamStates.data(), sizeof(AdamState) * safeGaussianCount,
                       vk::BufferUsageFlagBits::eStorageBuffer |
                           vk::BufferUsageFlagBits::eTransferSrc |
                           vk::BufferUsageFlagBits::eTransferDst,
                       vk::MemoryPropertyFlagBits::eDeviceLocal,
                       {transferQueueFamilyIndex_, computeQueueFamilyIndex_});
    createStorageBuffer(projected_, sizeof(ProjectedGaussian) * safeGaussianCount);
    tileItemCapacity_ = initialTileCapacity;
    createStorageBuffer(tileItems_, sizeof(uint32_t) * tileItemCapacity_);
    createStorageBuffer(tileKeyLow_, sizeof(uint32_t) * tileItemCapacity_);
    createStorageBuffer(tileKeyHigh_, sizeof(uint32_t) * tileItemCapacity_);
    createStorageBuffer(tileSortIndices_, sizeof(uint32_t) * tileItemCapacity_);
    createStorageBuffer(tileSortScratch_, sizeof(uint32_t) * tileItemCapacity_);
    createStorageBuffer(tileItemsSorted_, sizeof(uint32_t) * tileItemCapacity_);
    createStorageBuffer(adaptiveChildMasks_, sizeof(uint32_t) * tileItemCapacity_);
    createStorageBuffer(tileRanges_, sizeof(glm::uvec4) * tileRangeCapacity);
    createStorageBuffer(gaussianTileRanges_, sizeof(glm::uvec2) * safeGaussianCount);
    createStorageBuffer(gaussianTilePrefixScratch_,
                        sizeof(uint32_t) * gaussianPrefixScratchElementCount(safeGaussianCount));
    createZeroedStorageBuffer(adaptiveChildCounts_, sizeof(glm::uvec4) * parentTileCount);
    createZeroedStorageBuffer(adaptiveTileSplitFlags_, sizeof(uint32_t) * parentTileCount);
    createZeroedStorageBuffer(adaptiveChildOffsets_, sizeof(glm::uvec4) * parentTileCount);
    createStorageBuffer(adaptiveChildPrefix_, sizeof(uint32_t) * parentTileCount);
    createStorageBuffer(adaptiveChildBlockSums_,
                        sizeof(uint32_t) * ((parentTileCount + 255u) / 256u));
    createStorageBuffer(adaptiveActiveParents_, sizeof(uint32_t) * parentTileCount);
    adaptiveChildDispatch_.create(
        device_, physicalDevice_, transferQueue_, transferQueueFamilyIndex_, nullptr,
        sizeof(glm::uvec4),
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eIndirectBuffer |
            vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        {transferQueueFamilyIndex_, computeQueueFamilyIndex_});
    createStorageBuffer(renderedColor_, sizeof(glm::vec4) * pixelCount);
    createStorageBuffer(targetColor_, sizeof(uint32_t) * pixelCount);
    createStorageBuffer(pixelGrads_, sizeof(PixelGrad) * pixelCount);
    createStorageBuffer(ssimBackwardStates_, sizeof(SsimBackwardState) * pixelCount);
    const uint64_t validationGroupCountX = (safeWidth + 15u) / 16u;
    const uint64_t validationGroupCountY = (safeHeight + 15u) / 16u;
    const uint64_t pixelValidationPartialCount = validationGroupCountX * validationGroupCountY;
    const uint32_t gaussianValidationPartialCount =
        (safeGaussianCount + kTrainingValidationWorkgroupSize - 1u) / kTrainingValidationWorkgroupSize;
    createStorageBuffer(pixelValidationPartials_,
                        sizeof(TrainingPixelValidationPartial) * pixelValidationPartialCount);
    createStorageBuffer(gaussianValidationPartials_,
                        sizeof(TrainingGaussianValidationPartial) * gaussianValidationPartialCount);
    createStorageBuffer(validationFinalResult_, sizeof(TrainingValidationGpuResult));
    createStorageBuffer(pixelBlendStates_, sizeof(PixelBlendState) * pixelCount);
    createZeroedStorageBuffer(densificationStates_, sizeof(GaussianDensificationState) * safeGaussianCount);
    createStorageBuffer(projectedGrads_, sizeof(ProjectedGaussianGrad) * safeGaussianCount);
    createStorageBuffer(loss_, sizeof(float) * pixelCount);
    const glm::uvec4 zeroCounters{0, 0, 0, 0};
    counters_.create(device_,
                     physicalDevice_,
                     transferQueue_,
                     transferQueueFamilyIndex_,
                     &zeroCounters,
                     sizeof(glm::uvec4),
                     vk::BufferUsageFlagBits::eStorageBuffer |
                         vk::BufferUsageFlagBits::eTransferSrc |
                     vk::BufferUsageFlagBits::eTransferDst,
                     vk::MemoryPropertyFlagBits::eDeviceLocal,
                     {transferQueueFamilyIndex_, computeQueueFamilyIndex_});
    createStorageBuffer(previewInstances_, sizeof(GaussianTrainParam) * safeGaussianCount);
    createUniformBuffer(camera_, sizeof(TrainingForwardCamera));
    cameraIntrinsics_.cleanup();
    createUniformBuffer(cameraIntrinsics_, sizeof(TrainingCameraIntrinsics));
    uploadedIntrinsics_ = {};
    exactIntrinsics_ = false;
    cameraIntrinsics_.upload(&uploadedIntrinsics_, sizeof(uploadedIntrinsics_));

    gaussianCapacity_ = safeGaussianCount;
    gaussianWorkspaceCapacity_ = safeGaussianCount;
    densificationCapacity_ = 0;
    extent_ = {safeWidth, safeHeight};
}

void TrainingBuffers::configureSsimScratch(TrainingSsimMode mode) {
    if (!validTrainingSsimMode(mode)) throw std::invalid_argument("Invalid SSIM mode");
    const uint32_t planes = ssimScratchPlaneCount(mode);
    if (planes == 0u) {
        ssimScratch_.cleanup();
        return;
    }
    if (!device_ || extent_.width == 0u || extent_.height == 0u) return;
    const vk::DeviceSize pixels = static_cast<vk::DeviceSize>(extent_.width) * extent_.height;
    const vk::DeviceSize bytes = pixels * planes * sizeof(glm::vec4);
    if (bytes > physicalDevice_.getProperties().limits.maxStorageBufferRange ||
        pixels > std::numeric_limits<uint32_t>::max() / planes) {
        throw std::runtime_error("Separable SSIM scratch exceeds maxStorageBufferRange or shader indexing limit");
    }
    if (ssimScratch_.getSize() != bytes) {
        Buffer scratch;
        createStorageBuffer(scratch, bytes);
        ssimScratch_ = std::move(scratch);
    }
}

void TrainingBuffers::resizeTileItems(uint32_t tileItemCapacity) {
    if (!device_) {
        throw std::runtime_error("TrainingBuffers must be initialized before resizing tile items");
    }

    const uint32_t safeCapacity = radixKeyValueCapacity(std::max(tileItemCapacity, 1u),
        physicalDevice_.getProperties().limits.minStorageBufferOffsetAlignment);
    if (safeCapacity <= tileItemCapacity_ && tileItems_.getBuffer()) {
        return;
    }

    tileItems_.cleanup();
    tileKeyLow_.cleanup();
    tileKeyHigh_.cleanup();
    tileSortIndices_.cleanup();
    tileSortScratch_.cleanup();
    tileItemsSorted_.cleanup();
    adaptiveChildMasks_.cleanup();
    tileSortStorage_.cleanup();
    createStorageBuffer(tileItems_, sizeof(uint32_t) * safeCapacity);
    createStorageBuffer(tileKeyLow_, sizeof(uint32_t) * safeCapacity);
    createStorageBuffer(tileKeyHigh_, sizeof(uint32_t) * safeCapacity);
    createStorageBuffer(tileSortIndices_, sizeof(uint32_t) * safeCapacity);
    createStorageBuffer(tileSortScratch_, sizeof(uint32_t) * safeCapacity);
    createStorageBuffer(tileItemsSorted_, sizeof(uint32_t) * safeCapacity);
    createStorageBuffer(adaptiveChildMasks_, sizeof(uint32_t) * safeCapacity);
    tileItemCapacity_ = safeCapacity;
}

void TrainingBuffers::ensureTileSortStorage(uint32_t tileItemCapacity,
                                            vk::DeviceSize sortStorageSize,
                                            vk::BufferUsageFlags sortStorageUsage) {
    if (!device_) {
        throw std::runtime_error("TrainingBuffers must be initialized before resizing tile sort storage");
    }

    const uint32_t safeCapacity = radixKeyValueCapacity(std::max(tileItemCapacity, 1u),
        physicalDevice_.getProperties().limits.minStorageBufferOffsetAlignment);
    if (safeCapacity > tileItemCapacity_ ||
        !tileItems_.getBuffer() ||
        !tileKeyLow_.getBuffer() ||
        !tileKeyHigh_.getBuffer() ||
        !tileSortIndices_.getBuffer() ||
        !tileSortScratch_.getBuffer() ||
        !tileItemsSorted_.getBuffer()) {
        resizeTileItems(safeCapacity);
    }

    if (sortStorageSize == 0) {
        return;
    }

    const auto currentInfo = tileSortStorage_.getDescriptorInfo();
    if (tileSortStorage_.getBuffer() && currentInfo.range >= sortStorageSize) {
        return;
    }

    tileSortStorage_.cleanup();
    tileSortStorage_.create(device_,
                            physicalDevice_,
                            transferQueue_,
                            transferQueueFamilyIndex_,
                            nullptr,
                            sortStorageSize,
                            sortStorageUsage,
                            vk::MemoryPropertyFlagBits::eDeviceLocal,
                            {transferQueueFamilyIndex_, computeQueueFamilyIndex_});
}

void TrainingBuffers::ensureDensificationCapacity(uint32_t gaussianCapacity) {
    if (!device_) {
        throw std::runtime_error("TrainingBuffers must be initialized before densification resize");
    }

    const uint32_t safeCapacity = roundedGaussianCapacity(gaussianCapacity);
    if (safeCapacity <= densificationCapacity_ &&
        densifiedParams_.getBuffer() &&
        densifiedAdamStates_.getBuffer() &&
        densifiedGaussianValidationPartials_.getBuffer() &&
        densificationCounters_.getBuffer()) {
        return;
    }

    densifiedParams_.cleanup();
    densifiedAdamStates_.cleanup();
    densifiedGaussianValidationPartials_.cleanup();

    createStorageBuffer(densifiedParams_, sizeof(GaussianTrainParam) * safeCapacity);
    createStorageBuffer(densifiedAdamStates_, sizeof(AdamState) * safeCapacity);
    const uint32_t validationPartialCount =
        (safeCapacity + kTrainingValidationWorkgroupSize - 1u) /
        kTrainingValidationWorkgroupSize;
    createStorageBuffer(densifiedGaussianValidationPartials_,
                        sizeof(TrainingGaussianValidationPartial) * validationPartialCount);
    if (!densificationCounters_.getBuffer()) {
        const std::array<uint32_t, 16> zeroCounters{};
        densificationCounters_.create(device_,
                                      physicalDevice_,
                                      transferQueue_,
                                      transferQueueFamilyIndex_,
                                      zeroCounters.data(),
                                      zeroCounters.size() * sizeof(uint32_t),
                                      vk::BufferUsageFlagBits::eStorageBuffer |
                                          vk::BufferUsageFlagBits::eIndirectBuffer |
                                          vk::BufferUsageFlagBits::eTransferSrc |
                                          vk::BufferUsageFlagBits::eTransferDst,
                                      vk::MemoryPropertyFlagBits::eDeviceLocal,
                                      {transferQueueFamilyIndex_, computeQueueFamilyIndex_});
    }
    densificationCapacity_ = safeCapacity;
}

void TrainingBuffers::adoptDensifiedGaussians(uint32_t gaussianCount) {
    if (gaussianCount == 0 || gaussianCount > densificationCapacity_) {
        throw std::runtime_error("Invalid densified gaussian count");
    }
    const uint32_t oldActiveCapacity = gaussianCapacity_;
    const uint32_t newActiveCapacity = densificationCapacity_;
    std::swap(gaussianParams_, densifiedParams_);
    std::swap(adamStates_, densifiedAdamStates_);
    std::swap(gaussianValidationPartials_, densifiedGaussianValidationPartials_);
    gaussianCapacity_ = newActiveCapacity;
    densificationCapacity_ = oldActiveCapacity;

    if (gaussianCount > gaussianWorkspaceCapacity_) {
        const uint64_t geometricCapacity = std::max<uint64_t>(
            gaussianCount,
            static_cast<uint64_t>(gaussianWorkspaceCapacity_) * 3ull / 2ull);
        const uint32_t workspaceCapacity = roundedGaussianCapacity(
            static_cast<uint32_t>(std::min<uint64_t>(geometricCapacity,
                                                     std::numeric_limits<uint32_t>::max())));
        gaussianGrads_.cleanup();
        projected_.cleanup();
        gaussianTileRanges_.cleanup();
        gaussianTilePrefixScratch_.cleanup();
        densificationStates_.cleanup();
        projectedGrads_.cleanup();
        previewInstances_.cleanup();
        createStorageBuffer(gaussianGrads_, sizeof(GaussianGrad) * workspaceCapacity);
        createStorageBuffer(projected_, sizeof(ProjectedGaussian) * workspaceCapacity);
        createStorageBuffer(gaussianTileRanges_, sizeof(glm::uvec2) * workspaceCapacity);
        createStorageBuffer(
            gaussianTilePrefixScratch_,
            sizeof(uint32_t) * gaussianPrefixScratchElementCount(workspaceCapacity));
        createZeroedStorageBuffer(
            densificationStates_, sizeof(GaussianDensificationState) * workspaceCapacity);
        createStorageBuffer(projectedGrads_, sizeof(ProjectedGaussianGrad) * workspaceCapacity);
        createStorageBuffer(previewInstances_, sizeof(GaussianTrainParam) * workspaceCapacity);
        gaussianWorkspaceCapacity_ = workspaceCapacity;
    }
}

std::array<Buffer, 3> TrainingBuffers::createGaussianReorderTargets() {
    std::array<Buffer, 3> targets;
    createStorageBuffer(targets[0], gaussianParams_.getSize());
    createStorageBuffer(targets[1], adamStates_.getSize());
    createStorageBuffer(targets[2], densificationStates_.getSize());
    return targets;
}

void TrainingBuffers::adoptGaussianReorder(std::array<Buffer, 3>& targets) {
    if (!targets[0].getBuffer() || !targets[1].getBuffer() || !targets[2].getBuffer() ||
        targets[0].getSize() != gaussianParams_.getSize() ||
        targets[1].getSize() != adamStates_.getSize() ||
        targets[2].getSize() != densificationStates_.getSize())
        throw std::invalid_argument("Gaussian reorder targets do not match live capacities");
    std::swap(gaussianParams_, targets[0]);
    std::swap(adamStates_, targets[1]);
    std::swap(densificationStates_, targets[2]);
}

void TrainingBuffers::uploadGaussianParams(const GaussianTrainParam* params, uint32_t gaussianCount) {
    if (!params) {
        throw std::runtime_error("Training gaussian params are null");
    }
    if (gaussianCount > gaussianCapacity_) {
        throw std::runtime_error("Training gaussian param upload exceeds buffer capacity");
    }

    gaussianParams_.upload(params, static_cast<vk::DeviceSize>(gaussianCount) * sizeof(GaussianTrainParam));
}

std::vector<GaussianTrainParam> TrainingBuffers::downloadGaussianParams(uint32_t gaussianCount) {
    if (gaussianCount > gaussianCapacity_) {
        throw std::runtime_error("Training gaussian param download exceeds buffer capacity");
    }

    std::vector<GaussianTrainParam> params(gaussianCount);
    if (gaussianCount == 0) {
        return params;
    }

    LOG_DEBUG("Reading back {} training Gaussian parameters", gaussianCount);
    gaussianParams_.download(params.data(),
                             static_cast<vk::DeviceSize>(gaussianCount) * sizeof(GaussianTrainParam));
    LOG_DEBUG("Completed training Gaussian parameter readback");
    return params;
}

std::vector<ProjectedGaussianGrad> TrainingBuffers::downloadProjectedGrads(uint32_t gaussianCount) {
    if (gaussianCount > gaussianCapacity_) {
        throw std::runtime_error("Projected gradient download exceeds Gaussian buffer capacity");
    }
    std::vector<ProjectedGaussianGrad> gradients(gaussianCount);
    if (gaussianCount != 0u) {
        projectedGrads_.download(gradients.data(),
                                 static_cast<vk::DeviceSize>(gaussianCount) *
                                     sizeof(ProjectedGaussianGrad));
    }
    return gradients;
}

TrainingScanlineCapture TrainingBuffers::downloadScanlineCapture(const TrainingScanlineHeader& header) {
    (void)trainingScanlineCaptureBytes(header);
    if (header.width != extent_.width || header.height != extent_.height ||
        header.gaussianCount > gaussianCapacity_ || header.itemCount > tileItemCapacity_)
        throw std::runtime_error("S3-2 capture exceeds current training buffer capacities");
    TrainingScanlineCapture data;
    data.header = header;
    data.projected.resize(header.gaussianCount);
    data.items.resize(header.itemCount);
    data.ranges.resize(header.tileCount);
    data.blends.resize(size_t(header.width) * header.height);
    if (!data.projected.empty()) projected_.download(data.projected.data(), data.projected.size() * sizeof(ProjectedGaussian));
    if (!data.items.empty()) tileItemsSorted_.download(data.items.data(), data.items.size() * sizeof(uint32_t));
    tileRanges_.download(data.ranges.data(), data.ranges.size() * sizeof(glm::uvec4));
    pixelBlendStates_.download(data.blends.data(), data.blends.size() * sizeof(PixelBlendState));
    data.pixelGrads = downloadPixelGrads(static_cast<uint32_t>(data.blends.size()));
    validateTrainingScanlineCapture(data);
    return data;
}

std::vector<TrainingAdaptiveTileDiagnostic> TrainingBuffers::downloadAdaptiveTileDiagnostics(
    uint32_t parentItemCount) {
    if (parentItemCount > tileItemCapacity_) {
        throw std::runtime_error("Adaptive diagnostics exceed tile item buffer capacity");
    }
    const auto layout = makeTrainingTileLayout(extent_, kTrainingTileSize16);
    std::vector<TrainingAdaptiveTileDiagnostic> result(layout.tileCount);
    std::vector<uint32_t> keys(parentItemCount);
    if (parentItemCount != 0u) {
        // The original high keys retain the parent ID for every emitted pair.
        // Unlike leaf ranges, they are not overwritten when children install.
        tileKeyHigh_.download(keys.data(), static_cast<vk::DeviceSize>(parentItemCount) * sizeof(uint32_t));
    }
    for (uint32_t parent : keys) {
        if (parent >= layout.tileCount) {
            throw std::runtime_error("Adaptive diagnostics found an invalid parent key");
        }
        ++result[parent].parentItems;
    }

    std::vector<glm::uvec4> counts(layout.tileCount);
    std::vector<uint32_t> flags(layout.tileCount);
    adaptiveChildCounts_.download(counts.data(), counts.size() * sizeof(glm::uvec4));
    adaptiveTileSplitFlags_.download(flags.data(), flags.size() * sizeof(uint32_t));
    for (uint32_t parent = 0u; parent < layout.tileCount; ++parent) {
        if (flags[parent] > 1u) {
            throw std::runtime_error("Adaptive diagnostics found an invalid split flag");
        }
        result[parent].selected = flags[parent] != 0u;
        for (uint32_t child = 0u; child < 4u; ++child) {
            result[parent].childItems[child] = counts[parent][static_cast<glm::length_t>(child)];
        }
    }

    const size_t pixelCount = static_cast<size_t>(extent_.width) * extent_.height;
    std::vector<PixelBlendState> blends(pixelCount);
    pixelBlendStates_.download(blends.data(), pixelCount * sizeof(PixelBlendState));
    for (uint32_t y = 0u; y < extent_.height; ++y) {
        for (uint32_t x = 0u; x < extent_.width; ++x) {
            auto& parent = result[(y / 16u) * layout.tileCountX + x / 16u];
            const uint32_t child = ((y % 16u) / 8u) * 2u + (x % 16u) / 8u;
            const uint32_t rangeCount = parent.selected ? parent.childItems[child] : parent.parentItems;
            const uint32_t processed = std::min(
                blends[static_cast<size_t>(y) * extent_.width + x].processedCount, rangeCount);
            parent.maxProcessed[child] = std::max(parent.maxProcessed[child], processed);
        }
    }
    return result;
}

std::vector<uint32_t> TrainingBuffers::downloadAdaptiveChildMasks(uint32_t parentItemCount) {
    if (parentItemCount > tileItemCapacity_) {
        throw std::runtime_error("Adaptive mask download exceeds tile item buffer capacity");
    }
    std::vector<uint32_t> masks(parentItemCount);
    if (parentItemCount != 0u) {
        adaptiveChildMasks_.download(masks.data(),
                                     static_cast<vk::DeviceSize>(parentItemCount) * sizeof(uint32_t));
    }
    return masks;
}

std::vector<uint8_t> TrainingBuffers::downloadAdaptiveChildMembership(
    uint32_t gaussianCount, uint32_t parentItemCount, uint32_t childItemCount) {
    if (gaussianCount > gaussianCapacity_) {
        throw std::runtime_error("Child membership download exceeds Gaussian buffer capacity");
    }
    const uint64_t totalItemCount = static_cast<uint64_t>(parentItemCount) + childItemCount;
    if (totalItemCount > tileItemCapacity_) {
        throw std::runtime_error("Child membership download exceeds tile item buffer capacity");
    }

    std::vector<uint8_t> membership(gaussianCount, 0u);
    if (childItemCount == 0u) return membership;

    std::vector<uint32_t> tileItems(static_cast<size_t>(totalItemCount));
    tileItemsSorted_.download(tileItems.data(), totalItemCount * sizeof(uint32_t));
    for (uint64_t item = parentItemCount; item < totalItemCount; ++item) {
        const uint32_t gaussianIndex = tileItems[static_cast<size_t>(item)];
        if (gaussianIndex >= gaussianCount) {
            throw std::runtime_error("Adaptive child item refers to an invalid Gaussian index");
        }
        membership[gaussianIndex] = 1u;
    }
    return membership;
}

std::vector<PixelGrad> TrainingBuffers::downloadPixelGrads(uint32_t pixelCount) {
    const uint64_t capacity = static_cast<uint64_t>(extent_.width) * extent_.height;
    if (pixelCount > capacity) {
        throw std::runtime_error("Pixel gradient download exceeds buffer capacity");
    }
    std::vector<PixelGrad> result(pixelCount);
    if (pixelCount != 0u) {
        pixelGrads_.download(result.data(), static_cast<vk::DeviceSize>(pixelCount) * sizeof(PixelGrad));
    }
    return result;
}

std::vector<float> TrainingBuffers::downloadLoss(uint32_t pixelCount) {
    const uint64_t capacity = static_cast<uint64_t>(extent_.width) * static_cast<uint64_t>(extent_.height);
    if (pixelCount > capacity) {
        throw std::runtime_error("Training loss download exceeds buffer capacity");
    }

    std::vector<float> loss(pixelCount);
    if (pixelCount == 0) {
        return loss;
    }

    LOG_DEBUG("Reading back {} training loss values", pixelCount);
    loss_.download(loss.data(), static_cast<vk::DeviceSize>(pixelCount) * sizeof(float));
    LOG_DEBUG("Completed training loss readback");
    return loss;
}

std::vector<glm::vec4> TrainingBuffers::downloadRenderedColor(uint32_t pixelCount) {
    const uint64_t capacity = static_cast<uint64_t>(extent_.width) * static_cast<uint64_t>(extent_.height);
    if (pixelCount > capacity) {
        throw std::runtime_error("Training rendered color download exceeds buffer capacity");
    }

    std::vector<glm::vec4> rendered(pixelCount);
    if (pixelCount == 0) {
        return rendered;
    }

    LOG_DEBUG("Reading back {} rendered training pixels", pixelCount);
    renderedColor_.download(rendered.data(), static_cast<vk::DeviceSize>(pixelCount) * sizeof(glm::vec4));
    LOG_DEBUG("Completed rendered training pixel readback");
    return rendered;
}

void TrainingBuffers::uploadTargetColor(const uint8_t* pixels, uint32_t width, uint32_t height) {
    if (!pixels) {
        throw std::runtime_error("Training target pixels are null");
    }
    if (width != extent_.width || height != extent_.height) {
        throw std::runtime_error("Training target image size does not match training buffer extent");
    }

    const vk::DeviceSize uploadSize = static_cast<vk::DeviceSize>(width) *
                                      static_cast<vk::DeviceSize>(height) *
                                      sizeof(uint32_t);
    targetColor_.upload(pixels, uploadSize);
}

void TrainingBuffers::uploadCamera(const TrainingForwardCamera& camera,
                                   std::optional<TrainingCameraIntrinsics> intrinsics) {
    exactIntrinsics_ = false;
    camera_.upload(&camera, sizeof(TrainingForwardCamera));
    uploadedCamera_ = camera;
    if (intrinsics) {
        const auto& v = intrinsics->focalPrincipal;
        if (std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z) && std::isfinite(v.w) &&
            v.x > 0 && v.y > 0) {
            cameraIntrinsics_.upload(&*intrinsics, sizeof(TrainingCameraIntrinsics));
            uploadedIntrinsics_ = *intrinsics;
            exactIntrinsics_ = true;
        }
    }
}

void TrainingBuffers::recordTileItemCountReadback(vk::CommandBuffer commandBuffer) {
    if (!commandBuffer || !counters_.getBuffer() || !tileItemCountReadback_.getBuffer()) {
        throw std::runtime_error("Tile item count readback is not initialized");
    }

    vk::BufferCopy copy{};
    copy.setSize(sizeof(glm::uvec4));
    commandBuffer.copyBuffer(counters_.getBuffer(), tileItemCountReadback_.getBuffer(), copy);

    vk::BufferMemoryBarrier hostReadBarrier{};
    hostReadBarrier.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
                   .setDstAccessMask(vk::AccessFlagBits::eHostRead)
                   .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                   .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                   .setBuffer(tileItemCountReadback_.getBuffer())
                   .setOffset(0)
                   .setSize(sizeof(glm::uvec4));
    commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                  vk::PipelineStageFlagBits::eHost,
                                  vk::DependencyFlagBits{},
                                  0,
                                  nullptr,
                                  1,
                                  &hostReadBarrier,
                                  0,
                                  nullptr);
}

uint32_t TrainingBuffers::requiredTileItemCount() {
    if (!tileItemCountReadbackMapped_) {
        throw std::runtime_error("Tile item count readback is not mapped");
    }
    glm::uvec4 counters{0, 0, 0, 0};
    std::memcpy(&counters, tileItemCountReadbackMapped_, sizeof(counters));
    return counters.x;
}

std::array<uint32_t, 2> TrainingBuffers::adaptiveChildStats() const {
    if (!tileItemCountReadbackMapped_) {
        throw std::runtime_error("Adaptive child statistics readback is not mapped");
    }
    glm::uvec4 counters{0, 0, 0, 0};
    std::memcpy(&counters, tileItemCountReadbackMapped_, sizeof(counters));
    return {counters.y, counters.z};
}

void TrainingBuffers::recordDensificationStatsReadback(vk::CommandBuffer commandBuffer) {
    if (!commandBuffer || !densificationCounters_.getBuffer() ||
        !densificationStatsReadback_.getBuffer()) {
        throw std::runtime_error("Densification stats readback is not initialized");
    }

    vk::BufferCopy copy{};
    copy.setSize(sizeof(uint32_t) * 16u);
    commandBuffer.copyBuffer(
        densificationCounters_.getBuffer(), densificationStatsReadback_.getBuffer(), copy);

    vk::BufferMemoryBarrier hostReadBarrier{};
    hostReadBarrier.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
                   .setDstAccessMask(vk::AccessFlagBits::eHostRead)
                   .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                   .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                   .setBuffer(densificationStatsReadback_.getBuffer())
                   .setOffset(0)
                   .setSize(sizeof(uint32_t) * 16u);
    commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                  vk::PipelineStageFlagBits::eHost,
                                  vk::DependencyFlagBits{},
                                  0,
                                  nullptr,
                                  1,
                                  &hostReadBarrier,
                                  0,
                                  nullptr);
}

uint32_t TrainingBuffers::densifiedGaussianCount() {
    if (!densificationStatsReadbackMapped_) {
        throw std::runtime_error("Densification stats readback is not mapped");
    }
    std::array<uint32_t, 16> counters{};
    std::memcpy(counters.data(),
                densificationStatsReadbackMapped_,
                counters.size() * sizeof(uint32_t));
    return counters[1];
}

TrainingDensificationStats TrainingBuffers::densificationStats() {
    if (!densificationStatsReadbackMapped_) {
        throw std::runtime_error("Densification stats readback is not mapped");
    }
    std::array<uint32_t, 16> counters{};
    std::memcpy(counters.data(),
                densificationStatsReadbackMapped_,
                counters.size() * sizeof(uint32_t));

    TrainingDensificationStats stats{};
    stats.outputCount = counters[1];
    stats.pruneOpacityHits = counters[2];
    stats.pruneScreenHits = counters[3];
    stats.pruneWorldHits = counters[4];
    stats.keptSources = counters[5];
    stats.cloneSources = counters[6];
    stats.splitSources = counters[7];
    stats.prunedSources = counters[8];
    return stats;
}

void TrainingBuffers::createStorageBuffer(Buffer& buffer, vk::DeviceSize size) {
    buffer.create(device_,
                  physicalDevice_,
                  transferQueue_,
                  transferQueueFamilyIndex_,
                  nullptr,
                  size,
                  vk::BufferUsageFlagBits::eStorageBuffer |
                      vk::BufferUsageFlagBits::eTransferSrc |
                      vk::BufferUsageFlagBits::eTransferDst,
                  vk::MemoryPropertyFlagBits::eDeviceLocal,
                  {transferQueueFamilyIndex_, computeQueueFamilyIndex_});
}

void TrainingBuffers::createZeroedStorageBuffer(Buffer& buffer, vk::DeviceSize size) {
    std::vector<std::byte> zeros(static_cast<size_t>(size));
    buffer.create(device_,
                  physicalDevice_,
                  transferQueue_,
                  transferQueueFamilyIndex_,
                  zeros.data(),
                  size,
                  vk::BufferUsageFlagBits::eStorageBuffer |
                      vk::BufferUsageFlagBits::eTransferSrc |
                      vk::BufferUsageFlagBits::eTransferDst,
                  vk::MemoryPropertyFlagBits::eDeviceLocal,
                  {transferQueueFamilyIndex_, computeQueueFamilyIndex_});
}

void TrainingBuffers::createUniformBuffer(Buffer& buffer, vk::DeviceSize size) {
    buffer.create(device_,
                  physicalDevice_,
                  transferQueue_,
                  transferQueueFamilyIndex_,
                  nullptr,
                  size,
                  vk::BufferUsageFlagBits::eUniformBuffer |
                      vk::BufferUsageFlagBits::eTransferDst,
                  vk::MemoryPropertyFlagBits::eHostVisible |
                      vk::MemoryPropertyFlagBits::eHostCoherent,
                  {transferQueueFamilyIndex_, computeQueueFamilyIndex_});
}

} // namespace vulkan3DGS
