#pragma once

#include "training/app/training_settings.hpp"
#include "training/core/gaussian_training.hpp"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <stop_token>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace vulkan3DGS {

class Device;

struct TrainingUiSnapshot {
    bool rendererInitialized = false;
    bool hasDataset = false;
    bool hasTrainableModel = false;
    bool trainingComplete = false;
    bool subgroupSupported = false;
    bool tileGaussianSupported = false;
    bool vkSplatPerSplatSupported = false;
    bool vkSplatTensorSupported = false;
    TrainingPixelTo2DGSMode activePixelMode = TrainingPixelTo2DGSMode::Direct;
    uint32_t gaussianCount = 0;
    uint32_t trainingIteration = 0;
    uint32_t totalIterations = 0;
    uint32_t densificationIteration = 0;
    size_t currentFrameIndex = 0;
    TrainingFixedBenchmarkStats fixedBenchmark{};
    TrainingValidationStats validation{};
    TrainingCandidateProfileStats candidateProfile{};
    TrainingDensificationStats densification{};
    TrainingProfilingStats profiling{};
    ImageStreamerStats imageCache{};
    DeviceImageCacheStats deviceImageCache{};
};

struct TrainingWorkerResult {
    bool completed = false;
    bool pure = false;
    std::string error;
    std::chrono::steady_clock::time_point finishedAt{};
};

struct TrainingLifecycleState {
    bool initialized = false;
    bool datasetValid = false;
    bool datasetLoaded = false;
    bool running = false;
    bool pureActive = false;
    uint32_t frameCount = 0;
    uint32_t imageWidth = 0;
    uint32_t imageHeight = 0;
    uint64_t stepsDone = 0;
    std::string status;
    std::string error;
    bool errorPopupPending = false;
};

struct TrainingGpuInfo {
    uint32_t vulkanIndex = 0;
    std::string name;
    std::string typeName;
    std::string apiVersion;
    std::string driverVersion;
    std::string uuid;
    uint64_t deviceLocalMemoryBytes = 0;
    bool selected = false;
};

// Sole owner of the live training device, GaussianTraining instance, worker,
// lifecycle policy, timer and persisted training settings.
class TrainingController {
public:
    TrainingController();
    ~TrainingController();

    TrainingController(const TrainingController&) = delete;
    TrainingController& operator=(const TrainingController&) = delete;

    void initializeDevice(std::optional<std::string> commandLineSelector);
    void tick();
    void shutdown();

    [[nodiscard]] TrainingSettings& editableSettings() noexcept {
        return settings_;
    }
    [[nodiscard]] const TrainingSettings& settings() const noexcept {
        return settings_;
    }
    [[nodiscard]] const TrainingLifecycleState& state() const noexcept {
        return state_;
    }
    [[nodiscard]] TrainingUiSnapshot snapshotForUi() const;
    [[nodiscard]] bool isActive() const noexcept {
        return active_;
    }
    [[nodiscard]] bool isStopRequested() const noexcept {
        return stopRequested_;
    }
    [[nodiscard]] bool isTimerRunning() const noexcept {
        return timerRunning_;
    }
    [[nodiscard]] double elapsedSeconds() const;
    [[nodiscard]] std::string elapsedText() const;

    [[nodiscard]] std::vector<TrainingGpuInfo> gpuOptions() const;
    void selectGpu(const std::string& selector);

    void invalidateDataset(std::string message);
    void setDatasetPath(const std::filesystem::path& path, bool validateNow);
    void validateDataset();
    void loadDataset();
    void applyLiveSettings();
    void toggleTraining();
    void startFixedBenchmark();
    void exportDefaultModel();
    void exportModelTo(const std::filesystem::path& path);
    void setOutputDirectory(const std::filesystem::path& path);
    void setOutputPath(const std::filesystem::path& path);
    void reportError(std::string message) {
        setError(std::move(message));
    }
    void clearErrorPopup() noexcept {
        state_.errorPopupPending = false;
    }
    void saveSettings();

private:
    void createTrainingDevice(std::optional<std::string> selector);
    void replaceTrainingDevice(std::optional<std::string> selector);
    void cleanupTrainingResources();
    void applyConfiguration(const TrainingRunConfig& config);
    void initializeTrainingIfNeeded();
    void startWorker();
    void requestStop();
    void stopAndJoin();
    void consumeWorkerResult();
    void runSynchronousStep();
    void finishTrainingRun();

    void startTimer();
    void stopTimer();
    void stopTimerAt(std::chrono::steady_clock::time_point endTime);
    void resetTimer();
    void setStatus(std::string message);
    void setError(std::string message);
    void syncDefaultOutputName();
    [[nodiscard]] std::filesystem::path outputPlyPath() const;

    [[nodiscard]] TrainingUiSnapshot captureSnapshot() const;
    void publishSnapshot();
    void workerMain(std::stop_token stopToken, bool pure) noexcept;

    TrainingSettings settings_{};
    std::unique_ptr<const TrainingRunConfig> activeRunConfig_;
    GaussianTraining training_;
    std::unique_ptr<Device> trainingDevice_;
    std::jthread worker_;
    std::atomic<bool> resultReady_{false};
    mutable std::mutex resultMutex_;
    TrainingWorkerResult result_{};
    mutable std::mutex snapshotMutex_;
    TrainingUiSnapshot snapshot_{};
    bool snapshotValid_ = false;
    bool active_ = false;
    bool stopRequested_ = false;
    std::chrono::steady_clock::time_point timerStartedAt_{};
    double elapsedSeconds_ = 0.0;
    bool timerRunning_ = false;
    TrainingLifecycleState state_{};
};

} // namespace vulkan3DGS
