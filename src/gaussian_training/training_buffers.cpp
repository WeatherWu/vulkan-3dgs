#include "gaussian_training/training_buffers.hpp"

#include <algorithm>
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
    createStorageBuffer(adamStates_, sizeof(AdamState) * safeGaussianCount);
    createStorageBuffer(projected_, sizeof(ProjectedGaussian) * safeGaussianCount);
    createStorageBuffer(tileItems_, sizeof(uint32_t) * safeGaussianCount * 8u);
    createStorageBuffer(tileRanges_, sizeof(glm::uvec4) * tileCount);
    createStorageBuffer(renderedColor_, sizeof(glm::vec4) * pixelCount);
    createStorageBuffer(targetColor_, sizeof(glm::vec4) * pixelCount);
    createStorageBuffer(pixelGrads_, sizeof(PixelGrad) * pixelCount);
    createStorageBuffer(projectedGrads_, sizeof(ProjectedGaussianGrad) * safeGaussianCount);
    createStorageBuffer(loss_, sizeof(float) * pixelCount);
    createStorageBuffer(counters_, sizeof(glm::uvec4));
    createStorageBuffer(previewInstances_, sizeof(GaussianTrainParam) * safeGaussianCount);
    createUniformBuffer(camera_, sizeof(TrainingForwardCamera));

    gaussianCapacity_ = safeGaussianCount;
    extent_ = {safeWidth, safeHeight};
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
