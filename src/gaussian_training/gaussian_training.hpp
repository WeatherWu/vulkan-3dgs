#pragma once

#include <memory>
#include <filesystem>
#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>

#include "gaussian_training/training_buffers.hpp"
#include "gaussian_training/training_dataset.hpp"
#include "gaussian_training/training_pipeline.hpp"
#include "gaussian_training/gaussian_backward_renderer.hpp"
#include "renderer.hpp"
#include "vulkan/command_pool.hpp"

namespace vk_gs {

class GaussianRenderer;
class GaussianModel;

class GaussianTraining {
public:
    GaussianTraining();
    ~GaussianTraining();

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
    void initializeBackward(vk::Device device,
                            vk::PhysicalDevice physicalDevice,
                            vk::Queue computeQueue,
                            uint32_t computeQueueFamilyIndex,
                            uint32_t gaussianCount,
                            TrainingExtent extent);

    void initializeTraining(GLFWwindow* window,
                            vk::Device device,
                            vk::PhysicalDevice physicalDevice,
                            vk::Queue transferQueue,
                            uint32_t transferQueueFamilyIndex,
                            vk::Queue computeQueue,
                            uint32_t computeQueueFamilyIndex,
                            uint32_t gaussianCount,
                            TrainingExtent extent);
    void trainStep();
    void loadMipNeRF360Dataset(const std::filesystem::path& sceneRoot,
                               uint32_t preferredDownscale = 4);
    void setTrainingFrameIndex(size_t frameIndex);
    void setForwardModel(const GaussianModel* model);
    void setForwardRenderer(Renderer& renderer);

    bool isInitialized() const { return initialized_; }
    bool isRendererInitialized() const { return rendererInitialized_; }
    bool hasDataset() const { return !dataset_.empty(); }
    size_t datasetFrameCount() const { return dataset_.size(); }
    size_t currentFrameIndex() const { return currentDatasetFrameIndex_; }
    TrainingBuffers& buffers() { return buffers_; }
    TrainingPipelines& pipelines() { return pipelines_; }
    Renderer& forward() { return *forward_; }
    BackwardRenderer& backward() { return *backward_; }
    const Renderer& forward() const { return *forward_; }
    const BackwardRenderer& backward() const { return *backward_; }

private:
    TrainingPushConstants createPushConstants() const;
    void createTrainingCommandResources(vk::Device device,
                                        vk::Queue computeQueue,
                                        uint32_t computeQueueFamilyIndex);
    void destroyTrainingCommandResources();
    void copyForwardRenderToTrainingBuffer(vk::CommandBuffer commandBuffer,
                                           const vk::DescriptorBufferInfo& sourceInfo);
    void uploadCurrentTrainingFrame();
    TrainingForwardCamera createTrainingCamera(const TrainingCameraFrame& frame) const;
    void syncForwardRendererToCurrentFrame();
    glm::mat4 createProjectionMatrix(const TrainingCameraFrame& frame) const;

    bool initialized_ = false;
    bool rendererInitialized_ = false;
    vk::Queue computeQueue_ = nullptr;
    uint32_t computeQueueFamilyIndex_ = 0;
    CommandPool trainingCommandPool_;
    vk::CommandBuffer trainingCommandBuffer_ = nullptr;

    TrainingBuffers buffers_;
    TrainingPipelines pipelines_;
    TrainingDataset dataset_;
    size_t currentDatasetFrameIndex_ = 0;
    const GaussianModel* forwardModel_ = nullptr;
    std::unique_ptr<Renderer> ownedForward_;
    Renderer* forward_ = nullptr;
    std::unique_ptr<BackwardRenderer> backward_;

};

} // namespace vk_gs
