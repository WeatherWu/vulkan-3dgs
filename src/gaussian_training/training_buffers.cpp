#include "gaussian_training/training_buffers.hpp"

#include <algorithm>
#include <vector>
#include <stdexcept>

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
    projectedGrads_.cleanup();
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
    createStorageBuffer(tileItems_, sizeof(uint32_t) * safeGaussianCount * 8u);
    createStorageBuffer(tileRanges_, sizeof(glm::uvec4) * tileCount);
    createStorageBuffer(renderedColor_, sizeof(glm::vec4) * pixelCount);
    createStorageBuffer(targetColor_, sizeof(glm::vec4) * pixelCount);
    createStorageBuffer(pixelGrads_, sizeof(PixelGrad) * pixelCount);
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
    extent_ = {safeWidth, safeHeight};
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
