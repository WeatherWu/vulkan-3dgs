#include "gaussian_training.hpp"

#include "gaussian_renderer/gaussian_renderer.hpp"

namespace vk_gs {

TrainingRenderer::TrainingRenderer() = default;

TrainingRenderer::~TrainingRenderer() {
    cleanup();
}

void TrainingRenderer::initialize(GLFWwindow* window,
                                  vk::Device device,
                                  vk::PhysicalDevice physicalDevice,
                                  vk::Queue computeQueue,
                                  uint32_t computeQueueFamilyIndex,
                                  uint32_t gaussianCount,
                                  TrainingExtent extent) {
    if (initialized_) {
        return;
    }

    forward_ = std::make_unique<GaussianRenderer>();
    forward_->initialize(window);

    backward_ = std::make_unique<GaussianBackwardRenderer>();
    backward_->initialize(device,
                          physicalDevice,
                          computeQueue,
                          computeQueueFamilyIndex,
                          gaussianCount,
                          extent);

    initialized_ = true;
}

void TrainingRenderer::cleanup() {
    if (backward_) {
        backward_->cleanup();
        backward_.reset();
    }

    if (forward_) {
        forward_->cleanup();
        forward_.reset();
    }

    initialized_ = false;
}

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
    renderer_.cleanup();
    pipelines_.cleanup();
    buffers_.cleanup();
    initialized_ = false;
}

void GaussianTraining::resize(uint32_t gaussianCount, TrainingExtent extent) {
    buffers_.resize(gaussianCount, extent);
}

void GaussianTraining::initializeRenderer(GLFWwindow* window,
                                          vk::Device device,
                                          vk::PhysicalDevice physicalDevice,
                                          vk::Queue computeQueue,
                                          uint32_t computeQueueFamilyIndex,
                                          uint32_t gaussianCount,
                                          TrainingExtent extent) {
    renderer_.initialize(window,
                         device,
                         physicalDevice,
                         computeQueue,
                         computeQueueFamilyIndex,
                         gaussianCount,
                         extent);
}

} // namespace vk_gs
