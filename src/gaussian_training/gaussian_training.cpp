#include "gaussian_training.hpp"

namespace vk_gs {

void GaussianTraining::initialize(vk::Device device,
                                  vk::PhysicalDevice physicalDevice,
                                  vk::Queue transferQueue,
                                  uint32_t transferQueueFamilyIndex) {
    if (initialized_) {
        return;
    }

    buffers_.initialize(device, physicalDevice, transferQueue, transferQueueFamilyIndex);
    pipelines_.initialize(device);
    initialized_ = true;
}

void GaussianTraining::cleanup() {
    pipelines_.cleanup();
    buffers_.cleanup();
    initialized_ = false;
}

void GaussianTraining::resize(uint32_t gaussianCount, TrainingExtent extent) {
    buffers_.resize(gaussianCount, extent);
}

} // namespace vk_gs
