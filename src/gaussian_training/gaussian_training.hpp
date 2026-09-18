#pragma once

#include <algorithm>
#include <array>
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
#include "image/image_streamer.hpp"
#include "renderer.hpp"
#include "vulkan/command_pool.hpp"
#include "vulkan/device_image_cache.hpp"

namespace vulkan3DGS {

class GaussianRenderer;

class GaussianTraining {
public:
    GaussianTraining();
    ~GaussianTraining();

    void initialize(vk::Device device,
                    vk::PhysicalDevice physicalDevice,
                    vk::Queue transferQueue,
                    uint32_t transferQueueFamilyIndex,
                    uint32_t computeQueueFamilyIndex);
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
    void startFixedWorkloadBenchmark(const TrainingFixedBenchmarkConfig& config);
    void stopFixedWorkloadBenchmark();
    void setDensificationConfig(const TrainingDensificationConfig& config) { densificationConfig_ = config; }
    void setOptimizerConfig(const TrainingOptimizerConfig& config) { optimizerConfig_ = config; }
    void setScheduleConfig(const TrainingScheduleConfig& config);
    void setPixelTo2DGSMode(TrainingPixelTo2DGSMode mode);
    void setPixelTo2DGSMinSubgroupUtilization(float utilization) {
        pixelTo2DGSMinSubgroupUtilization_ = std::clamp(utilization, 0.0f, 1.0f);
    }
    void setForwardCompositeMode(TrainingForwardCompositeMode mode);
    bool subgroupPixelTo2DGSSupported() const;
    bool tileGaussianPixelTo2DGSSupported() const;
    bool vkSplatPerSplatSupported() const;
    bool vkSplatTensorSupported() const;
    TrainingPixelTo2DGSMode activePixelTo2DGSMode() const;
    void setValidationInterval(uint32_t interval) { validationInterval_ = interval; }
    uint32_t validationInterval() const { return validationInterval_; }

    bool isInitialized() const { return initialized_; }
    bool isRendererInitialized() const { return rendererInitialized_; }
    bool hasDataset() const { return !dataset_.empty(); }
    bool hasTrainableModel() const { return trainableGaussianCount_ > 0; }
    bool isTrainingComplete() const;
    bool isFixedWorkloadBenchmarkActive() const { return fixedBenchmarkStats_.active; }
    const TrainingFixedBenchmarkStats& fixedWorkloadBenchmarkStats() const {
        return fixedBenchmarkStats_;
    }
    bool usedRandomInitialization() const { return usedRandomInitialization_; }
    uint32_t gaussianCount() const { return trainableGaussianCount_; }
    uint32_t trainingIteration() const { return trainingIteration_; }
    uint32_t totalIterations() const { return scheduleConfig_.totalIterations; }
    const TrainingValidationStats& validationStats() const { return validationStats_; }
    const TrainingCandidateProfileStats& candidateProfileStats() const { return candidateProfileStats_; }
    const TrainingDensificationStats& densificationStats() const { return lastDensificationStats_; }
    uint32_t densificationStatsIteration() const { return lastDensificationStatsIteration_; }
    const TrainingProfilingStats& profilingStats() const { return profilingStats_; }
    ImageStreamerStats imageCacheStats() const { return imageStreamer_ ? imageStreamer_->stats() : ImageStreamerStats{}; }
    DeviceImageCacheStats deviceImageCacheStats() const {
        return deviceImageCache_ ? deviceImageCache_->stats() : DeviceImageCacheStats{};
    }
    size_t datasetFrameCount() const { return dataset_.size(); }
    size_t currentFrameIndex() const { return currentDatasetFrameIndex_; }
    TrainingBuffers& buffers() { return buffers_; }
    ForwardTrainingRenderer& forward() { return *forward_; }
    BackwardRenderer& backward() { return *backward_; }
    const ForwardTrainingRenderer& forward() const { return *forward_; }
    const BackwardRenderer& backward() const { return *backward_; }

private:
    struct ValidationReadbackSlot {
        vk::Buffer buffer = nullptr;
        vk::DeviceMemory memory = nullptr;
        void* mapped = nullptr;
        vk::Fence fence = nullptr;
        uint32_t iteration = 0;
        uint32_t tileItemCount = 0;
        uint32_t gaussianCount = 0;
        bool pending = false;
    };

    TrainingPushConstants createPushConstants() const;
    TrainingDensificationPushConstants createDensificationPushConstants(bool pruneByScreenSize) const;
    bool shouldRunDensification() const;
    void createTrainingCommandResources(vk::Device device,
                                        vk::Queue computeQueue,
                                        uint32_t computeQueueFamilyIndex);
    void destroyTrainingCommandResources();
    void createValidationReadbackResources();
    void destroyValidationReadbackResources();
    ValidationReadbackSlot* acquireValidationReadbackSlot();
    void recordValidationReadbackCopy(ValidationReadbackSlot& slot);
    void collectCompletedValidationReadbacks();
    void createTrainingProfilingResources(vk::Device device,
                                          vk::PhysicalDevice physicalDevice,
                                          uint32_t computeQueueFamilyIndex);
    void destroyTrainingProfilingResources();
    void collectGpuProfilingStats();
    void resetProfilingStats();
    void resetCandidateProfileStats();
    void resetProfilingLastSamples();
    void recordCpuProfilingSample(TrainingCpuProfileStage stage, float milliseconds);
    void recordGpuProfilingSample(TrainingGpuProfileStage stage, float milliseconds);
    void selectTrainingFrameForIteration();
    void prefetchUpcomingTrainingFrames();
    void uploadCurrentTrainingFrame();
    void initializeDeviceImageCache();
    void refreshDeviceImageCacheBudget(bool reserveForDensification);
    uint64_t initialDeviceImageCacheBudget() const;
    void validateTrainingStep(const TrainingValidationGpuResult& result,
                              uint32_t tileItemCount,
                              uint32_t gaussianCount);
    std::vector<GaussianTrainParam> createSparsePointInitialGaussians() const;
    std::vector<GaussianTrainParam> createRandomInitialGaussians(const TrainingInitializationConfig& config) const;
    TrainingForwardCamera createTrainingCamera(const TrainingCameraFrame& frame) const;
    glm::mat4 createProjectionMatrix(const TrainingCameraFrame& frame) const;
    float estimateSceneExtent() const;

    bool initialized_ = false;
    bool rendererInitialized_ = false;
    vk::Device device_ = nullptr;
    vk::PhysicalDevice physicalDevice_ = nullptr;
    vk::Queue transferQueue_ = nullptr;
    uint32_t transferQueueFamilyIndex_ = 0;
    uint64_t pendingImageUploadValue_ = 0;
    uint32_t deviceCacheGrowthResumeIteration_ = 0;
    vk::Queue computeQueue_ = nullptr;
    uint32_t computeQueueFamilyIndex_ = 0;
    CommandPool trainingCommandPool_;
    vk::CommandBuffer prepareCommandBuffer_ = nullptr;
    vk::CommandBuffer mainCommandBuffer_ = nullptr;
    vk::Fence prepareFence_ = nullptr;
    vk::Fence mainFence_ = nullptr;
    std::array<ValidationReadbackSlot, 3> validationReadbackSlots_{};
    size_t nextValidationReadbackSlot_ = 0;
    vk::Device profilingDevice_ = nullptr;
    vk::QueryPool profilingQueryPool_ = nullptr;
    float timestampPeriodNanoseconds_ = 0.0f;
    bool gpuTimestampProfilingAvailable_ = false;

    TrainingBuffers buffers_;
    std::unique_ptr<ImageStreamer> imageStreamer_;
    std::unique_ptr<DeviceImageCache> deviceImageCache_;
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
    TrainingFixedBenchmarkStats fixedBenchmarkStats_{};
    uint32_t fixedBenchmarkCompletedSteps_ = 0;
    std::mt19937 frameRng_{1u};
    std::vector<size_t> randomFrameStack_;
    uint32_t trainingIteration_ = 0;
    uint32_t optimizerStep_ = 0;
    uint32_t validationInterval_ = 100;
    TrainingValidationStats validationStats_{};
    TrainingCandidateProfileStats candidateProfileStats_{};
    TrainingDensificationStats lastDensificationStats_{};
    uint32_t lastDensificationStatsIteration_ = 0;
    TrainingProfilingStats profilingStats_{};
    TrainingPixelTo2DGSMode pixelTo2DGSMode_ = TrainingPixelTo2DGSMode::Auto;
    float pixelTo2DGSMinSubgroupUtilization_ = 0.5f;
    TrainingForwardCompositeMode forwardCompositeMode_ = TrainingForwardCompositeMode::WorkgroupShared;
    float sceneExtent_ = 1.0f;

};

} // namespace vulkan3DGS
