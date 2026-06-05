#pragma once

#include <vulkan/vulkan.hpp>

#include "gaussian_training/training_buffers.hpp"
#include "gaussian_training/training_pipeline.hpp"

namespace vk_gs {

class GaussianTraining {
public:
    void initialize(vk::Device device,
                    vk::PhysicalDevice physicalDevice,
                    vk::Queue transferQueue,
                    uint32_t transferQueueFamilyIndex);
    void cleanup();
    void resize(uint32_t gaussianCount, TrainingExtent extent);

    bool isInitialized() const { return initialized_; }
    TrainingBuffers& buffers() { return buffers_; }
    TrainingPipelines& pipelines() { return pipelines_; }

private:
    bool initialized_ = false;
    TrainingBuffers buffers_;
    TrainingPipelines pipelines_;

};

} // namespace vk_gs
