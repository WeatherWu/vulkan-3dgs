#pragma once

#include <vulkan/vulkan.hpp>

#include "gaussian_training/training_types.hpp"
#include "vulkan/buffer.hpp"

namespace vk_gs {

class TrainingBuffers {
public:
    void initialize(vk::Device device,
                    vk::PhysicalDevice physicalDevice,
                    vk::Queue transferQueue,
                    uint32_t transferQueueFamilyIndex);
    void cleanup();

    void resize(uint32_t gaussianCount, TrainingExtent extent);
    void uploadGaussianParams(const GaussianTrainParam* params, uint32_t gaussianCount);
    void uploadTargetColor(const glm::vec4* pixels, uint32_t width, uint32_t height);
    void uploadCamera(const TrainingForwardCamera& camera);

    uint32_t gaussianCapacity() const { return gaussianCapacity_; }
    TrainingExtent extent() const { return extent_; }

    vk::DescriptorBufferInfo gaussianParamsInfo() const { return gaussianParams_.getDescriptorInfo(); }
    vk::DescriptorBufferInfo gaussianGradsInfo() const { return gaussianGrads_.getDescriptorInfo(); }
    vk::DescriptorBufferInfo adamStatesInfo() const { return adamStates_.getDescriptorInfo(); }
    vk::DescriptorBufferInfo projectedInfo() const { return projected_.getDescriptorInfo(); }
    vk::DescriptorBufferInfo tileItemsInfo() const { return tileItems_.getDescriptorInfo(); }
    vk::DescriptorBufferInfo tileRangesInfo() const { return tileRanges_.getDescriptorInfo(); }
    vk::DescriptorBufferInfo renderedColorInfo() const { return renderedColor_.getDescriptorInfo(); }
    vk::DescriptorBufferInfo targetColorInfo() const { return targetColor_.getDescriptorInfo(); }
    vk::DescriptorBufferInfo pixelGradsInfo() const { return pixelGrads_.getDescriptorInfo(); }
    vk::DescriptorBufferInfo projectedGradsInfo() const { return projectedGrads_.getDescriptorInfo(); }
    vk::DescriptorBufferInfo lossInfo() const { return loss_.getDescriptorInfo(); }
    vk::DescriptorBufferInfo countersInfo() const { return counters_.getDescriptorInfo(); }
    vk::DescriptorBufferInfo previewInstancesInfo() const { return previewInstances_.getDescriptorInfo(); }
    vk::DescriptorBufferInfo cameraInfo() const { return camera_.getDescriptorInfo(); }

private:
    void createStorageBuffer(Buffer& buffer, vk::DeviceSize size);
    void createUniformBuffer(Buffer& buffer, vk::DeviceSize size);

    vk::Device device_ = nullptr;
    vk::PhysicalDevice physicalDevice_ = nullptr;
    vk::Queue transferQueue_ = nullptr;
    uint32_t transferQueueFamilyIndex_ = 0;

    uint32_t gaussianCapacity_ = 0;
    TrainingExtent extent_{};

    Buffer gaussianParams_;
    Buffer gaussianGrads_;
    Buffer adamStates_;
    Buffer projected_;
    Buffer tileItems_;
    Buffer tileRanges_;
    Buffer renderedColor_;
    Buffer targetColor_;
    Buffer pixelGrads_;
    Buffer projectedGrads_;
    Buffer loss_;
    Buffer counters_;
    Buffer previewInstances_;
    Buffer camera_;
};

} // namespace vk_gs
