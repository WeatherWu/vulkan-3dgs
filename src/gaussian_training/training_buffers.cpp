#include "gaussian_training/training_buffers.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>
#include <stdexcept>
#include <glm/glm.hpp>

namespace vk_gs {

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
    camera_.cleanup();
    previewInstances_.cleanup();
    counters_.cleanup();
    loss_.cleanup();
    projectedGrads_.cleanup();
    densificationCounters_.cleanup();
    densifiedAdamStates_.cleanup();
    densifiedParams_.cleanup();
    densificationStates_.cleanup();
    gaussianVisibility_.cleanup();
    pixelBlendStates_.cleanup();
    pixelGrads_.cleanup();
    targetColor_.cleanup();
    renderedColor_.cleanup();
    tileRanges_.cleanup();
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
    tileRanges_.cleanup();
    renderedColor_.cleanup();
    targetColor_.cleanup();
    pixelGrads_.cleanup();
    pixelBlendStates_.cleanup();
    gaussianVisibility_.cleanup();
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
                       vk::MemoryPropertyFlagBits::eDeviceLocal);
    createStorageBuffer(projected_, sizeof(ProjectedGaussian) * safeGaussianCount);
    tileItemCapacity_ = static_cast<uint32_t>(
        std::min<uint64_t>(std::max<uint64_t>(static_cast<uint64_t>(safeGaussianCount) * 8ull, 1ull),
                           std::numeric_limits<uint32_t>::max()));
    createStorageBuffer(tileItems_, sizeof(uint32_t) * tileItemCapacity_);
    createStorageBuffer(tileRanges_, sizeof(glm::uvec4) * tileCount);
    createStorageBuffer(renderedColor_, sizeof(glm::vec4) * pixelCount);
    createStorageBuffer(targetColor_, sizeof(glm::vec4) * pixelCount);
    createStorageBuffer(pixelGrads_, sizeof(PixelGrad) * pixelCount);
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
    createStorageBuffer(tileItems_, sizeof(uint32_t) * safeCapacity);
    tileItemCapacity_ = safeCapacity;
}

void TrainingBuffers::ensureDensificationCapacity(uint32_t gaussianCapacity) {
    if (!device_) {
        throw std::runtime_error("TrainingBuffers must be initialized before densification resize");
    }

    const uint32_t safeCapacity = std::max(gaussianCapacity, 1u);
    if (safeCapacity <= densificationCapacity_ &&
        densifiedParams_.getBuffer() &&
        densifiedAdamStates_.getBuffer() &&
        densificationCounters_.getBuffer()) {
        return;
    }

    densifiedParams_.cleanup();
    densifiedAdamStates_.cleanup();
    densificationCounters_.cleanup();

    createStorageBuffer(densifiedParams_, sizeof(GaussianTrainParam) * safeCapacity);
    createZeroedStorageBuffer(densifiedAdamStates_, sizeof(AdamState) * safeCapacity);
    const glm::uvec4 zeroCounters{0, 0, 0, 0};
    densificationCounters_.create(device_,
                                  physicalDevice_,
                                  transferQueue_,
                                  transferQueueFamilyIndex_,
                                  &zeroCounters,
                                  sizeof(glm::uvec4),
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

    gaussianParams_ = std::move(densifiedParams_);
    adamStates_ = std::move(densifiedAdamStates_);
    densificationCounters_.cleanup();
    createStorageBuffer(gaussianGrads_, sizeof(GaussianGrad) * newCapacity);
    createStorageBuffer(projected_, sizeof(ProjectedGaussian) * newCapacity);
    createStorageBuffer(gaussianVisibility_, sizeof(GaussianVisibilityState) * newCapacity);
    createZeroedStorageBuffer(densificationStates_, sizeof(GaussianDensificationState) * newCapacity);
    createStorageBuffer(projectedGrads_, sizeof(ProjectedGaussianGrad) * newCapacity);
    createStorageBuffer(previewInstances_, sizeof(GaussianTrainParam) * newCapacity);

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

    gaussianParams_.download(params.data(),
                             static_cast<vk::DeviceSize>(gaussianCount) * sizeof(GaussianTrainParam));
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

void TrainingBuffers::uploadTargetColor(const glm::vec4* pixels, uint32_t width, uint32_t height) {
    if (!pixels) {
        throw std::runtime_error("Training target pixels are null");
    }
    if (width != extent_.width || height != extent_.height) {
        throw std::runtime_error("Training target image size does not match training buffer extent");
    }

    const vk::DeviceSize uploadSize = static_cast<vk::DeviceSize>(width) *
                                      static_cast<vk::DeviceSize>(height) *
                                      sizeof(glm::vec4);
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
    glm::uvec4 counters{0, 0, 0, 0};
    densificationCounters_.download(&counters, sizeof(counters));
    return counters.x;
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
                  vk::MemoryPropertyFlagBits::eDeviceLocal);
}

} // namespace vk_gs
