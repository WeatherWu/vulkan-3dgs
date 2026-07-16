#include "gaussian_training/training_buffers.hpp"

#include "utils/logger.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <vector>
#include <stdexcept>
#include <glm/glm.hpp>

namespace vulkan3DGS {

void TrainingBuffers::initialize(vk::Device device,
                                 vk::PhysicalDevice physicalDevice,
                                 vk::Queue transferQueue,
                                 uint32_t transferQueueFamilyIndex) {
    device_ = device;
    physicalDevice_ = physicalDevice;
    transferQueue_ = transferQueue;
    transferQueueFamilyIndex_ = transferQueueFamilyIndex;
}

void TrainingBuffers::cleanup() {
    clearTargetColorDescriptor();
    camera_.cleanup();
    previewInstances_.cleanup();
    counters_.cleanup();
    loss_.cleanup();
    validationFinalResult_.cleanup();
    gaussianValidationPartials_.cleanup();
    pixelValidationPartials_.cleanup();
    projectedGrads_.cleanup();
    densificationCandidateStates_.cleanup();
    densificationCandidateAdamStates_.cleanup();
    densificationCandidateParams_.cleanup();
    densificationCounters_.cleanup();
    densifiedAdamStates_.cleanup();
    densifiedParams_.cleanup();
    densificationStates_.cleanup();
    gaussianVisibility_.cleanup();
    pixelBlendStates_.cleanup();
    pixelGrads_.cleanup();
    ssimBackwardStates_.cleanup();
    targetColor_.cleanup();
    renderedColor_.cleanup();
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
    densificationCapacity_ = 0;
    tileItemCapacity_ = 0;
    extent_ = {};
    device_ = nullptr;
    physicalDevice_ = nullptr;
    transferQueue_ = nullptr;
    transferQueueFamilyIndex_ = 0;
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
    renderedColor_.cleanup();
    targetColor_.cleanup();
    pixelGrads_.cleanup();
    ssimBackwardStates_.cleanup();
    pixelValidationPartials_.cleanup();
    gaussianValidationPartials_.cleanup();
    validationFinalResult_.cleanup();
    pixelBlendStates_.cleanup();
    gaussianVisibility_.cleanup();
    projectedGrads_.cleanup();
    densificationCandidateStates_.cleanup();
    densificationCandidateAdamStates_.cleanup();
    densificationCandidateParams_.cleanup();
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
                       vk::MemoryPropertyFlagBits::eDeviceLocal);
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
    createStorageBuffer(renderedColor_, sizeof(glm::vec4) * pixelCount);
    createStorageBuffer(targetColor_, sizeof(uint32_t) * pixelCount);
    createStorageBuffer(pixelGrads_, sizeof(PixelGrad) * pixelCount);
    createStorageBuffer(ssimBackwardStates_, sizeof(SsimBackwardState) * pixelCount);
    const uint64_t pixelValidationPartialCount =
        (pixelCount + kTrainingValidationWorkgroupSize - 1u) / kTrainingValidationWorkgroupSize;
    const uint32_t gaussianValidationPartialCount =
        (safeGaussianCount + kTrainingValidationWorkgroupSize - 1u) / kTrainingValidationWorkgroupSize;
    createStorageBuffer(pixelValidationPartials_,
                        sizeof(TrainingPixelValidationPartial) * pixelValidationPartialCount);
    createStorageBuffer(gaussianValidationPartials_,
                        sizeof(TrainingGaussianValidationPartial) * gaussianValidationPartialCount);
    createStorageBuffer(validationFinalResult_, sizeof(TrainingValidationGpuResult));
    createStorageBuffer(pixelBlendStates_, sizeof(PixelBlendState) * pixelCount);
    createStorageBuffer(gaussianVisibility_, sizeof(GaussianVisibilityState) * safeGaussianCount);
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
                     vk::MemoryPropertyFlagBits::eDeviceLocal);
    createStorageBuffer(previewInstances_, sizeof(GaussianTrainParam) * safeGaussianCount);
    createUniformBuffer(camera_, sizeof(TrainingForwardCamera));

    gaussianCapacity_ = safeGaussianCount;
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
                            vk::MemoryPropertyFlagBits::eDeviceLocal);
}

void TrainingBuffers::ensureDensificationCapacity(uint32_t gaussianCapacity) {
    if (!device_) {
        throw std::runtime_error("TrainingBuffers must be initialized before densification resize");
    }

    const uint32_t safeCapacity = std::max(gaussianCapacity, 1u);
    if (safeCapacity <= densificationCapacity_ &&
        densifiedParams_.getBuffer() &&
        densifiedAdamStates_.getBuffer() &&
        densificationCandidateParams_.getBuffer() &&
        densificationCandidateAdamStates_.getBuffer() &&
        densificationCandidateStates_.getBuffer() &&
        densificationCounters_.getBuffer()) {
        return;
    }

    densifiedParams_.cleanup();
    densifiedAdamStates_.cleanup();
    densificationCandidateParams_.cleanup();
    densificationCandidateAdamStates_.cleanup();
    densificationCandidateStates_.cleanup();
    densificationCounters_.cleanup();

    createStorageBuffer(densifiedParams_, sizeof(GaussianTrainParam) * safeCapacity);
    createZeroedStorageBuffer(densifiedAdamStates_, sizeof(AdamState) * safeCapacity);
    createStorageBuffer(densificationCandidateParams_, sizeof(GaussianTrainParam) * safeCapacity);
    createZeroedStorageBuffer(densificationCandidateAdamStates_, sizeof(AdamState) * safeCapacity);
    createZeroedStorageBuffer(densificationCandidateStates_, sizeof(GaussianDensificationState) * safeCapacity);
    const std::array<uint32_t, 16> zeroCounters{};
    densificationCounters_.create(device_,
                                  physicalDevice_,
                                  transferQueue_,
                                  transferQueueFamilyIndex_,
                                  zeroCounters.data(),
                                  zeroCounters.size() * sizeof(uint32_t),
                                  vk::BufferUsageFlagBits::eStorageBuffer |
                                      vk::BufferUsageFlagBits::eTransferSrc |
                                      vk::BufferUsageFlagBits::eTransferDst,
                                  vk::MemoryPropertyFlagBits::eDeviceLocal);
    densificationCapacity_ = safeCapacity;
}

void TrainingBuffers::adoptDensifiedGaussians(uint32_t gaussianCount) {
    if (gaussianCount == 0 || gaussianCount > densificationCapacity_) {
        throw std::runtime_error("Invalid densified gaussian count");
    }
    const uint32_t newCapacity = densificationCapacity_;

    gaussianParams_.cleanup();
    adamStates_.cleanup();
    gaussianGrads_.cleanup();
    projected_.cleanup();
    gaussianVisibility_.cleanup();
    densificationStates_.cleanup();
    projectedGrads_.cleanup();
    previewInstances_.cleanup();
    gaussianValidationPartials_.cleanup();
    densificationCandidateStates_.cleanup();
    densificationCandidateAdamStates_.cleanup();
    densificationCandidateParams_.cleanup();

    gaussianParams_ = std::move(densifiedParams_);
    adamStates_ = std::move(densifiedAdamStates_);
    densificationCounters_.cleanup();
    createStorageBuffer(gaussianGrads_, sizeof(GaussianGrad) * newCapacity);
    createStorageBuffer(projected_, sizeof(ProjectedGaussian) * newCapacity);
    createStorageBuffer(gaussianVisibility_, sizeof(GaussianVisibilityState) * newCapacity);
    createZeroedStorageBuffer(densificationStates_, sizeof(GaussianDensificationState) * newCapacity);
    createStorageBuffer(projectedGrads_, sizeof(ProjectedGaussianGrad) * newCapacity);
    createStorageBuffer(previewInstances_, sizeof(GaussianTrainParam) * newCapacity);
    const uint32_t gaussianValidationPartialCount =
        (newCapacity + kTrainingValidationWorkgroupSize - 1u) / kTrainingValidationWorkgroupSize;
    createStorageBuffer(gaussianValidationPartials_,
                        sizeof(TrainingGaussianValidationPartial) * gaussianValidationPartialCount);

    gaussianCapacity_ = newCapacity;
    densificationCapacity_ = 0;
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

std::vector<GaussianVisibilityState> TrainingBuffers::downloadGaussianVisibility(uint32_t gaussianCount) {
    if (gaussianCount > gaussianCapacity_) {
        throw std::runtime_error("Training gaussian visibility download exceeds buffer capacity");
    }

    std::vector<GaussianVisibilityState> visibility(gaussianCount);
    if (gaussianCount == 0) {
        return visibility;
    }

    gaussianVisibility_.download(visibility.data(),
                                 static_cast<vk::DeviceSize>(gaussianCount) * sizeof(GaussianVisibilityState));
    return visibility;
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

uint32_t TrainingBuffers::requiredTileItemCount() {
    glm::uvec4 counters{0, 0, 0, 0};
    counters_.download(&counters, sizeof(counters));
    return counters.x;
}

uint32_t TrainingBuffers::densifiedGaussianCount() {
    std::array<uint32_t, 16> counters{};
    densificationCounters_.download(counters.data(), counters.size() * sizeof(uint32_t));
    return counters[1];
}

TrainingDensificationStats TrainingBuffers::densificationStats() {
    std::array<uint32_t, 16> counters{};
    densificationCounters_.download(counters.data(), counters.size() * sizeof(uint32_t));

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
                  vk::MemoryPropertyFlagBits::eDeviceLocal);
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
                  vk::MemoryPropertyFlagBits::eDeviceLocal);
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
                      vk::MemoryPropertyFlagBits::eHostCoherent);
}

} // namespace vulkan3DGS
