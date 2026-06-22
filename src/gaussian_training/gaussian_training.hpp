#pragma once

#include <memory>
#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>

#include "gaussian_training/training_buffers.hpp"
#include "gaussian_training/training_pipeline.hpp"
#include "gaussian_training/gaussian_backward_renderer.hpp"
#include "renderer.hpp"

namespace vk_gs {

class GaussianRenderer;

class TrainingRenderer {
public:
    TrainingRenderer();
    ~TrainingRenderer();

    void initialize(GLFWwindow* window,
                    vk::Device device,
                    vk::PhysicalDevice physicalDevice,
                    vk::Queue computeQueue,
                    uint32_t computeQueueFamilyIndex,
                    uint32_t gaussianCount,
                    TrainingExtent extent);
    void cleanup();

    Renderer& forward() { return *forward_; }
    BackwardRenderer& backward() { return *backward_; }
    const Renderer& forward() const { return *forward_; }
    const BackwardRenderer& backward() const { return *backward_; }

    bool isInitialized() const { return initialized_; }

private:
    std::unique_ptr<Renderer> forward_;
    std::unique_ptr<BackwardRenderer> backward_;
    bool initialized_ = false;
};

class GaussianTraining {
public:
    void initialize(vk::Device device,
                    vk::PhysicalDevice physicalDevice,
                    vk::Queue transferQueue,
                    uint32_t transferQueueFamilyIndex);
    void cleanup();
    void resize(uint32_t gaussianCount, TrainingExtent extent);
    void initializeRenderer(GLFWwindow* window,
                            vk::Device device,
                            vk::PhysicalDevice physicalDevice,
                            vk::Queue computeQueue,
                            uint32_t computeQueueFamilyIndex,
                            uint32_t gaussianCount,
                            TrainingExtent extent);
    bool isInitialized() const { return initialized_; }
    TrainingBuffers& buffers() { return buffers_; }
    TrainingPipelines& pipelines() { return pipelines_; }
    TrainingRenderer& renderer() { return renderer_; }

private:
    bool initialized_ = false;
    TrainingBuffers buffers_;
    TrainingPipelines pipelines_;
    TrainingRenderer renderer_;

};

} // namespace vk_gs
