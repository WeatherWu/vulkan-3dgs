#pragma once

#include <algorithm>
#include <memory>
#include <optional>
#include <filesystem>
#include <vector>
#include <vulkan/vulkan.hpp>

#include "training/core/training_buffers.hpp"
#include "training/core/training_dataset.hpp"
#include "training/core/gaussian_backward_renderer.hpp"
#include "training/core/gaussian_densification_renderer.hpp"
#include "training/core/gaussian_forward_renderer.hpp"
#include "training/core/training_frame_scheduler.hpp"
#include "training/core/training_model_io.hpp"
#include "training/core/training_image_runtime.hpp"
#include "training/core/training_step_executor.hpp"
#include "training/core/training_tile_layout.hpp"
#include "training/core/training_profiling_service.hpp"
#include "training/core/training_validation_service.hpp"
#include "training/core/training_pixel_culling_policy.hpp"
#include "render/renderer.hpp"

namespace vulkan3DGS {

class GaussianTraining {
public:
    GaussianTraining();
    ~GaussianTraining();

    void initialize(vk::Device device,
                    vk::PhysicalDevice physicalDevice,
                    vk::Queue transferQueue,
                    uint32_t transferQueueFamilyIndex,
                    uint32_t computeQueueFamilyIndex,
                    std::optional<bool> vulkanValidationLayersEnabled = std::nullopt);
    void cleanup();
    void resize(uint32_t gaussianCount, TrainingExtent extent);
    void initializeTrainingRenderers(vk::Device device,
                                     vk::PhysicalDevice physicalDevice,
                                     vk::Queue computeQueue,
                                     uint32_t computeQueueFamilyIndex,
                                     uint32_t gaussianCount,
                                     TrainingExtent extent);

    void trainStep();
    const TrainingStepObservation& lastStepObservation() const { return lastStepObservation_; }
    void setStepObservationEnabled(bool enabled) { stepObservationEnabled_ = enabled; }
    // Worker/stopped-owner only, after a completed step; inspection does not advance training.
    TrainingStateFiniteCheck inspectTrainingStateFinite();
    uint64_t diagnosticTrainingStateHash();
    void loadMipNeRF360Dataset(const std::filesystem::path& sceneRoot,
                               uint32_t preferredDownscale = 4);
    void initializeModelFromDataset(const TrainingInitializationConfig& config = {});
    bool exportToPLY(const std::filesystem::path& path);
    void exportFixedBenchmarkGradients(const std::filesystem::path& path);
    void exportScanlineDiagnosticInputs(const std::filesystem::path& path);
    // Explicit mutation: exactly one eligible ordinary step, then exclusive export, no next step.
    void captureOptimizerStepAndPause(const std::filesystem::path& path);
    void exportAdaptiveTileDiagnostics(const std::filesystem::path& path);
    void setTrainingFrameIndex(size_t frameIndex);
    void startFixedWorkloadBenchmark(const TrainingFixedBenchmarkConfig& config);
    void stopFixedWorkloadBenchmark();
    void setDensificationConfig(const TrainingDensificationConfig& config) { densificationConfig_ = config; }
    void setOptimizerConfig(const TrainingOptimizerConfig& config) { optimizerConfig_ = config; }
    void setScheduleConfig(const TrainingScheduleConfig& config);
    void setPixelTo2DGSMode(TrainingPixelTo2DGSMode mode);
    void setAdaptiveTensorChildBatch(uint32_t batch);
    void setTensorSubgroupReduction(bool enabled);
    void setTensorGlobalPixelGrad(bool enabled);
    void setTensorRgbLatencyHiding(bool enabled);
    void setTensorEarlyCull(bool enabled) { tensorEarlyCull_ = enabled; }
    TrainingPixelCullStatus tensorEarlyCullStatus() const;
    void resetTensorEarlyCullRunStats();
    const TrainingPixelCullRunStats& tensorEarlyCullRunStats() const { return pixelCullRunStats_; }
    void setSsimMode(TrainingSsimMode mode);
    TrainingSsimMode activeSsimMode() const;
    bool tensorRgbLatencyHidingActive() const;
    bool tensorGlobalPixelGradActive() const;
    bool tensorSubgroupReductionSupported() const;
    bool tensorSubgroupReductionActive() const;
    uint32_t adaptiveTensorChildBatch() const { return adaptiveTensorChildBatch_; }
    void setPixelTo2DGSMinSubgroupUtilization(float utilization) {
        pixelTo2DGSMinSubgroupUtilization_ = std::clamp(utilization, 0.0f, 1.0f);
    }
    void setForwardCompositeMode(TrainingForwardCompositeMode mode);
    void setTileMode(TrainingTileMode mode, uint32_t adaptiveSplitThreshold = 256u);
    bool subgroupPixelTo2DGSSupported() const;
    bool tileGaussianPixelTo2DGSSupported() const;
    bool vkSplatPerSplatSupported() const;
    bool vkSplatTensorSupported() const;
    bool adaptiveSpatialBackwardSupported() const;
    TrainingPixelTo2DGSMode activePixelTo2DGSMode() const;
    TrainingTileMode tileMode() const { return tileMode_; }
    uint32_t activeTileSize() const {
        return tileMode_ == TrainingTileMode::AdaptiveSpatial ? 0u : trainingTileSize(tileMode_);
    }
    uint32_t adaptiveTileSplitThreshold() const { return adaptiveTileSplitThreshold_; }
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
    uint32_t totalIterations() const { return frameScheduler_.totalIterations(); }
    const TrainingValidationStats& validationStats() const {
        return validationService_.stats();
    }
    const TrainingCandidateProfileStats& candidateProfileStats() const {
        return validationService_.candidateProfileStats();
    }
    const TrainingDensificationStats& densificationStats() const { return lastDensificationStats_; }
    uint32_t densificationStatsIteration() const { return lastDensificationStatsIteration_; }
    const TrainingProfilingStats& profilingStats() const {
        return profilingService_.stats();
    }
    ImageStreamerStats imageCacheStats() const {
        return imageRuntime_.imageCacheStats();
    }
    DeviceImageCacheStats deviceImageCacheStats() const {
        return imageRuntime_.deviceImageCacheStats();
    }
    size_t datasetFrameCount() const { return dataset_.size(); }
    size_t currentFrameIndex() const { return frameScheduler_.currentFrame(); }
    TrainingBuffers& buffers() { return buffers_; }
    ForwardTrainingRenderer& forward() { return *forward_; }
    BackwardRenderer& backward() { return *backward_; }
    const ForwardTrainingRenderer& forward() const { return *forward_; }
    const BackwardRenderer& backward() const { return *backward_; }

private:
    TrainingPushConstants createPushConstants() const;
    TrainingDensificationPushConstants createDensificationPushConstants(bool pruneByScreenSize) const;
    bool shouldRunDensification() const;
    void createTrainingProfilingResources(vk::Device device,
                                          vk::PhysicalDevice physicalDevice,
                                          uint32_t computeQueueFamilyIndex);
    void destroyTrainingProfilingResources();
    void detachProfilingQueryPool();
    void resetProfilingStats();
    void resetCandidateProfileStats();
    void resetProfilingLastSamples();
    void recordCpuProfilingSample(TrainingCpuProfileStage stage, float milliseconds);
    void prefetchUpcomingTrainingFrames();
    void uploadCurrentTrainingFrame();
    TrainingForwardCamera createTrainingCamera(const TrainingCameraFrame& frame) const;
    glm::mat4 createProjectionMatrix(const TrainingCameraFrame& frame) const;

    bool initialized_ = false;
    std::optional<bool> vulkanValidationLayersEnabled_;
    bool rendererInitialized_ = false;
    vk::Device device_ = nullptr;
    vk::PhysicalDevice physicalDevice_ = nullptr;
    vk::Queue transferQueue_ = nullptr;
    uint32_t transferQueueFamilyIndex_ = 0;
    uint32_t computeQueueFamilyIndex_ = 0;
    TrainingValidationService validationService_;
    TrainingProfilingService profilingService_;
    TrainingFrameScheduler frameScheduler_;
    TrainingImageRuntime imageRuntime_;
    TrainingStepExecutor stepExecutor_;

    TrainingBuffers buffers_;
    TrainingDataset dataset_;
    uint32_t trainableGaussianCount_ = 0;
    TrainingStepObservation lastStepObservation_{};
    bool stepObservationEnabled_ = false;
    bool usedRandomInitialization_ = false;
    std::unique_ptr<ForwardTrainingRenderer> forward_;
    std::unique_ptr<BackwardRenderer> backward_;
    std::unique_ptr<GaussianDensificationRenderer> densification_;
    TrainingDensificationConfig densificationConfig_{};
    TrainingOptimizerConfig optimizerConfig_{};
    TrainingFixedBenchmarkStats fixedBenchmarkStats_{};
    uint32_t fixedBenchmarkCompletedSteps_ = 0;
    bool checkpointDiagnosticsRequested_ = false;
    bool checkpointWarpRequested_ = false;
    bool benchmarkTensorEarlyCullRequested_ = false;
    TrainingPixelCullRunStats pixelCullRunStats_{};
    uint32_t trainingIteration_ = 0;
    uint32_t optimizerStep_ = 0;
    uint32_t validationInterval_ = 100;
    TrainingDensificationStats lastDensificationStats_{};
    uint32_t lastDensificationStatsIteration_ = 0;
    TrainingPixelTo2DGSMode pixelTo2DGSMode_ = TrainingPixelTo2DGSMode::Auto;
    float pixelTo2DGSMinSubgroupUtilization_ = 0.5f;
    TrainingForwardCompositeMode forwardCompositeMode_ = TrainingForwardCompositeMode::WorkgroupShared;
    TrainingTileMode tileMode_ = TrainingTileMode::Fixed16;
    uint32_t adaptiveTileSplitThreshold_ = 256u;
    uint32_t adaptiveTensorChildBatch_ = 8u;
    bool tensorSubgroupReduction_ = false;
    bool tensorGlobalPixelGrad_ = false;
    bool tensorRgbLatencyHiding_ = false;
    bool tensorEarlyCull_ = false;
    TrainingSsimMode ssimMode_ = TrainingSsimMode::TwoDimensional;
    float sceneExtent_ = 1.0f;
    TrainingOptimizerCaptureSession* optimizerCapture_ = nullptr;
    bool optimizerCaptureStateUncertain_ = false;

};

} // namespace vulkan3DGS
