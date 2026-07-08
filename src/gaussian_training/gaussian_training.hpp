#pragma once

#include <memory>
#include <filesystem>
#include <random>
#include <vector>
#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>

#include "gaussian_training/training_buffers.hpp"
#include "gaussian_training/training_dataset.hpp"
#include "gaussian_training/gaussian_backward_renderer.hpp"
#include "gaussian_training/gaussian_densification_renderer.hpp"
#include "gaussian_training/gaussian_forward_renderer.hpp"
#include "renderer.hpp"
#include "vulkan/command_pool.hpp"

namespace vulkan3DGS {

class GaussianRenderer;

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
    void initializeTrainingRenderers(vk::Device device,
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
    void initializeModelFromDataset(const TrainingInitializationConfig& config = {});
    bool exportToPLY(const std::filesystem::path& path);
    void setTrainingFrameIndex(size_t frameIndex);
    void setDensificationConfig(const TrainingDensificationConfig& config) { densificationConfig_ = config; }
    void setOptimizerConfig(const TrainingOptimizerConfig& config) { optimizerConfig_ = config; }
    void setScheduleConfig(const TrainingScheduleConfig& config);
    void setValidationInterval(uint32_t interval) { validationInterval_ = interval; }

    bool isInitialized() const { return initialized_; }
    bool isRendererInitialized() const { return rendererInitialized_; }
    bool hasDataset() const { return !dataset_.empty(); }
    bool hasTrainableModel() const { return trainableGaussianCount_ > 0; }
    bool isTrainingComplete() const;
    bool usedRandomInitialization() const { return usedRandomInitialization_; }
    uint32_t gaussianCount() const { return trainableGaussianCount_; }
    uint32_t trainingIteration() const { return trainingIteration_; }
    uint32_t totalIterations() const { return scheduleConfig_.totalIterations; }
    const TrainingValidationStats& validationStats() const { return validationStats_; }
    size_t datasetFrameCount() const { return dataset_.size(); }
    size_t currentFrameIndex() const { return currentDatasetFrameIndex_; }
    TrainingBuffers& buffers() { return buffers_; }
    ForwardTrainingRenderer& forward() { return *forward_; }
    BackwardRenderer& backward() { return *backward_; }
    const ForwardTrainingRenderer& forward() const { return *forward_; }
    const BackwardRenderer& backward() const { return *backward_; }

private:
    TrainingPushConstants createPushConstants() const;
    TrainingDensificationPushConstants createDensificationPushConstants(bool pruneByScreenSize) const;
    bool shouldRunDensification() const;
    void createTrainingCommandResources(vk::Device device,
                                        vk::Queue computeQueue,
                                        uint32_t computeQueueFamilyIndex);
    void destroyTrainingCommandResources();
    void selectTrainingFrameForIteration();
    void uploadCurrentTrainingFrame();
    void validateTrainingStep(uint32_t tileItemCount);
    std::vector<GaussianTrainParam> createSparsePointInitialGaussians() const;
    std::vector<GaussianTrainParam> createRandomInitialGaussians(const TrainingInitializationConfig& config) const;
    TrainingForwardCamera createTrainingCamera(const TrainingCameraFrame& frame) const;
    glm::mat4 createProjectionMatrix(const TrainingCameraFrame& frame) const;
    float estimateSceneExtent() const;

    bool initialized_ = false;
    bool rendererInitialized_ = false;
    vk::Queue computeQueue_ = nullptr;
    uint32_t computeQueueFamilyIndex_ = 0;
    CommandPool trainingCommandPool_;
    vk::CommandBuffer trainingCommandBuffer_ = nullptr;

    TrainingBuffers buffers_;
    TrainingDataset dataset_;
    size_t currentDatasetFrameIndex_ = 0;
    uint32_t trainableGaussianCount_ = 0;
    bool usedRandomInitialization_ = false;
    std::unique_ptr<ForwardTrainingRenderer> forward_;
    std::unique_ptr<BackwardRenderer> backward_;
    std::unique_ptr<GaussianDensificationRenderer> densification_;
    TrainingDensificationConfig densificationConfig_{};
    TrainingOptimizerConfig optimizerConfig_{};
    TrainingScheduleConfig scheduleConfig_{};
    std::mt19937 frameRng_{1u};
    std::vector<size_t> randomFrameStack_;
    uint32_t trainingIteration_ = 0;
    uint32_t validationInterval_ = 10;
    TrainingValidationStats validationStats_{};
    float sceneExtent_ = 1.0f;

};

} // namespace vulkan3DGS
