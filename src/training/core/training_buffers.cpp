#include "training/core/training_buffers.hpp"

#include "utils/logger.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>
#include <glm/glm.hpp>

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

void TrainingBuffers::initialize(vk::Device device,
                                 vk::PhysicalDevice physicalDevice,
                                 vk::Queue transferQueue,
                                 uint32_t transferQueueFamilyIndex,
                                 uint32_t computeQueueFamilyIndex) {
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

void TrainingBuffers::cleanup() {
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
    targetColor_.cleanup();
    renderedColor_.cleanup();
    gaussianTilePrefixScratch_.cleanup();
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
    clearTargetColorDescriptor();
    if (!device_) {
        throw std::runtime_error("TrainingBuffers must be initialized before resize");
    }

    const uint32_t safeGaussianCount = std::max(gaussianCount, 1u);
    const uint32_t safeWidth = std::max(extent.width, 1u);
    const uint32_t safeHeight = std::max(extent.height, 1u);
    const uint64_t pixelCount = static_cast<uint64_t>(safeWidth) * static_cast<uint64_t>(safeHeight);
    const uint64_t tileCount = ((safeWidth + 15u) / 16u) * ((safeHeight + 15u) / 16u);

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
    renderedColor_.cleanup();
    targetColor_.cleanup();
    pixelGrads_.cleanup();
    ssimBackwardStates_.cleanup();
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

    createStorageBuffer(gaussianParams_, sizeof(GaussianTrainParam) * safeGaussianCount);
    createStorageBuffer(gaussianGrads_, sizeof(GaussianGrad) * safeGaussianCount);
    std::vector<AdamState> zeroAdamStates(safeGaussianCount);
    adamStates_.create(device_,
                       physicalDevice_,
                       transferQueue_,
                       transferQueueFamilyIndex_,
                       zeroAdamStates.data(),
                       sizeof(AdamState) * safeGaussianCount,
                       vk::BufferUsageFlagBits::eStorageBuffer |
                           vk::BufferUsageFlagBits::eTransferSrc |
                       vk::BufferUsageFlagBits::eTransferDst,
                       vk::MemoryPropertyFlagBits::eDeviceLocal,
                       {transferQueueFamilyIndex_, computeQueueFamilyIndex_});
    createStorageBuffer(projected_, sizeof(ProjectedGaussian) * safeGaussianCount);
    tileItemCapacity_ = static_cast<uint32_t>(
        std::min<uint64_t>(std::max<uint64_t>(static_cast<uint64_t>(safeGaussianCount) * 8ull, 1ull),
                           std::numeric_limits<uint32_t>::max()));
    createStorageBuffer(tileItems_, sizeof(uint32_t) * tileItemCapacity_);
    createStorageBuffer(tileKeyLow_, sizeof(uint32_t) * tileItemCapacity_);
    createStorageBuffer(tileKeyHigh_, sizeof(uint32_t) * tileItemCapacity_);
    createStorageBuffer(tileSortIndices_, sizeof(uint32_t) * tileItemCapacity_);
    createStorageBuffer(tileSortScratch_, sizeof(uint32_t) * tileItemCapacity_);
    createStorageBuffer(tileItemsSorted_, sizeof(uint32_t) * tileItemCapacity_);
    createStorageBuffer(tileRanges_, sizeof(glm::uvec4) * tileCount);
    createStorageBuffer(gaussianTileRanges_, sizeof(glm::uvec2) * safeGaussianCount);
    createStorageBuffer(gaussianTilePrefixScratch_,
                        sizeof(uint32_t) * gaussianPrefixScratchElementCount(safeGaussianCount));
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

    gaussianCapacity_ = safeGaussianCount;
    gaussianWorkspaceCapacity_ = safeGaussianCount;
    densificationCapacity_ = 0;
    extent_ = {safeWidth, safeHeight};
}

void TrainingBuffers::resizeTileItems(uint32_t tileItemCapacity) {
    if (!device_) {
        throw std::runtime_error("TrainingBuffers must be initialized before resizing tile items");
    }

    const uint32_t safeCapacity = std::max(tileItemCapacity, 1u);
    if (safeCapacity <= tileItemCapacity_ && tileItems_.getBuffer()) {
        return;
    }

    tileItems_.cleanup();
    tileKeyLow_.cleanup();
    tileKeyHigh_.cleanup();
    tileSortIndices_.cleanup();
    tileSortScratch_.cleanup();
    tileItemsSorted_.cleanup();
    tileSortStorage_.cleanup();
    createStorageBuffer(tileItems_, sizeof(uint32_t) * safeCapacity);
    createStorageBuffer(tileKeyLow_, sizeof(uint32_t) * safeCapacity);
    createStorageBuffer(tileKeyHigh_, sizeof(uint32_t) * safeCapacity);
    createStorageBuffer(tileSortIndices_, sizeof(uint32_t) * safeCapacity);
    createStorageBuffer(tileSortScratch_, sizeof(uint32_t) * safeCapacity);
    createStorageBuffer(tileItemsSorted_, sizeof(uint32_t) * safeCapacity);
    tileItemCapacity_ = safeCapacity;
}

void TrainingBuffers::ensureTileSortStorage(uint32_t tileItemCapacity,
                                            vk::DeviceSize sortStorageSize,
                                            vk::BufferUsageFlags sortStorageUsage) {
    if (!device_) {
        throw std::runtime_error("TrainingBuffers must be initialized before resizing tile sort storage");
    }

    const uint32_t safeCapacity = std::max(tileItemCapacity, 1u);
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

void TrainingBuffers::uploadCamera(const TrainingForwardCamera& camera) {
    camera_.upload(&camera, sizeof(TrainingForwardCamera));
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
