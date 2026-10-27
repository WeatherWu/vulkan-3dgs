#include "training/app/training_controller.hpp"

#include "context/context.hpp"
#include "context/device.hpp"
#include "training/core/training_optimizer_capture.hpp"
#include "utils/file_utils.hpp"
#include "utils/logger.hpp"

#include <algorithm>
#include <exception>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace vulkan3DGS {

namespace {

constexpr uint32_t kTrainingSnapshotPublishInterval = 100u;

} // namespace

TrainingController::TrainingController() {
    setTextBuffer(settings_.paths.dataset, "data/mipnerf360/bicycle");
    setTextBuffer(settings_.paths.outputDirectory, "output");
    setTextBuffer(settings_.paths.outputName, "bicycle.ply");
    TrainingSettingsStore::load(TrainingSettingsStore::defaultPath(), settings_);
    state_.status = "Training dataset is not loaded";
}

TrainingController::~TrainingController() {
    cleanupTrainingResources();
}

void TrainingController::configureSmoke(std::optional<TrainingSmokeOptions> options) {
    if (active_) throw std::logic_error("Pause training before configuring smoke diagnostics");
    if (options) {
        options->validate();
        options->report = std::filesystem::absolute(options->report);
    }
    smokeOptions_ = std::move(options);
    if (smokeOptions_)
        LOG_INFO(
            "Training smoke configured: stop at {}, report {}. Start Training in the UI to begin.",
            smokeOptions_->stopIteration, smokeOptions_->report.string());
}

void TrainingController::initializeDevice(std::optional<std::string> commandLineSelector,
                                          std::optional<bool> vulkanValidationLayersEnabled) {
    vulkanValidationLayersEnabled_ = vulkanValidationLayersEnabled;
    const bool usePersisted = !commandLineSelector && !settings_.gpuSelector.empty();
    std::optional<std::string> selector = commandLineSelector;
    if (usePersisted) selector = settings_.gpuSelector;
    try {
        createTrainingDevice(std::move(selector));
    } catch (const std::exception& error) {
        if (!usePersisted) throw;
        LOG_WARN("Saved training GPU is unavailable ({}); selecting the default GPU", error.what());
        createTrainingDevice(std::nullopt);
    }
}

void TrainingController::tick() {
    consumeWorkerResult();
    if (!state_.running || active_) return;

    try {
        const TrainingRunSettings& run = activeRunConfig_ ? activeRunConfig_->run : settings_.run;
        const uint32_t budget = std::max(run.benchmarkStepsPerFrame, 1u);
        for (uint32_t step = 0; state_.running && step < budget; ++step) {
            runSynchronousStep();
        }
    } catch (const std::exception& error) {
        state_.running = false;
        state_.pureActive = false;
        stopTimer();
        setError(error.what());
    }
}

void TrainingController::shutdown() {
    stopAndJoin();
    stopTimer();
    try {
        saveSettings();
    } catch (const std::exception& error) {
        LOG_WARN("Could not save training settings: {}", error.what());
    }
    cleanupTrainingResources();
    state_.initialized = false;
}

void TrainingController::createTrainingDevice(std::optional<std::string> selector) {
    if (trainingDevice_) {
        throw std::runtime_error("Training device is already initialized");
    }
    auto device =
        std::make_unique<Device>(vk::SurfaceKHR{}, std::move(selector), DeviceRole::Training);
    device->createDevice();
    trainingDevice_ = std::move(device);
}

void TrainingController::replaceTrainingDevice(std::optional<std::string> selector) {
    auto replacement =
        std::make_unique<Device>(vk::SurfaceKHR{}, std::move(selector), DeviceRole::Training);
    replacement->createDevice();
    stopAndJoin();
    training_.cleanup();
    trainingDevice_ = std::move(replacement);
}

void TrainingController::cleanupTrainingResources() {
    stopAndJoin();
    training_.cleanup();
    trainingDevice_.reset();
}

std::vector<TrainingGpuInfo> TrainingController::gpuOptions() const {
    std::vector<TrainingGpuInfo> result;
    if (!trainingDevice_) return result;
    const auto& selected = trainingDevice_->getSelectedPhysicalDeviceInfo();
    for (const auto& info : trainingDevice_->getAvailablePhysicalDeviceInfos()) {
        result.push_back(TrainingGpuInfo{
            .vulkanIndex = info.vulkanIndex,
            .name = info.name,
            .typeName = info.typeName,
            .apiVersion = info.apiVersion,
            .driverVersion = info.driverVersion,
            .uuid = info.uuid,
            .deviceLocalMemoryBytes = info.deviceLocalMemoryBytes,
            .selected = info.vulkanIndex == selected.vulkanIndex,
        });
    }
    return result;
}

void TrainingController::selectGpu(const std::string& selector) {
    if (state_.running) return;
    resetTimer();
    try {
        replaceTrainingDevice(selector);
        state_.initialized = false;
        state_.datasetLoaded = false;
        state_.stepsDone = 0;
        const auto& gpu = trainingDevice_->getSelectedPhysicalDeviceInfo();
        settings_.gpuSelector = gpu.uuid.empty() ? std::to_string(gpu.vulkanIndex) : gpu.uuid;
        saveSettings();
        setStatus("Training GPU selected: [" + std::to_string(gpu.vulkanIndex) + "] " + gpu.name +
                  ". Dataset must be loaded again.");
    } catch (const std::exception& error) {
        setError(std::string("Failed to switch training GPU: ") + error.what());
    }
}

void TrainingController::invalidateDataset(std::string message) {
    if (active_) return;
    state_.datasetValid = false;
    state_.datasetLoaded = false;
    state_.initialized = false;
    state_.running = false;
    state_.pureActive = false;
    state_.frameCount = 0;
    state_.imageWidth = 0;
    state_.imageHeight = 0;
    resetTimer();
    setStatus(std::move(message));
}

void TrainingController::setDatasetPath(const std::filesystem::path& path, bool validateNow) {
    if (state_.running || active_) {
        setError("Pause training before changing the dataset.");
        return;
    }
    setTextBuffer(settings_.paths.dataset, path.string());
    syncDefaultOutputName();
    invalidateDataset("Dataset path changed. Validate or start training to load it.");
    if (validateNow) validateDataset();
}

void TrainingController::validateDataset() {
    if (state_.running || active_) {
        setError("Pause training before validating a dataset.");
        return;
    }
    try {
        const auto validation = TrainingDatasetLoader::validateMipNeRF360Scene(
            textBufferString(settings_.paths.dataset),
            static_cast<uint32_t>(settings_.initialization.downscale));
        state_.datasetValid = validation.valid;
        state_.frameCount = validation.frameCount;
        state_.imageWidth = validation.width;
        state_.imageHeight = validation.height;
        if (!validation.valid) {
            state_.datasetLoaded = false;
            state_.initialized = false;
            state_.running = false;
            setError(validation.message);
            return;
        }
        syncDefaultOutputName();
        setStatus("Dataset valid: " + std::to_string(validation.frameCount) + " frames, " +
                  std::to_string(validation.width) + "x" + std::to_string(validation.height));
    } catch (const std::exception& error) {
        invalidateDataset("Dataset validation failed.");
        setError(error.what());
    }
}

void TrainingController::loadDataset() {
    if (state_.running || active_) {
        setError("Pause training before loading a dataset.");
        return;
    }
    validateDataset();
    if (!state_.datasetValid) return;
    if (!trainingDevice_) {
        setError("Training GPU is not initialized");
        return;
    }
    try {
        stopAndJoin();
        resetTimer();
        state_.datasetLoaded = false;
        state_.initialized = false;
        state_.stepsDone = 0;
        training_.cleanup();
        auto& device = *trainingDevice_;
        const auto computeIndex = device.getQueueFamilyIndices().computeIndex;
        if (!computeIndex) throw std::runtime_error("Training compute queue family is unavailable");
        const uint32_t compute = *computeIndex;
        const uint32_t transfer = device.getQueueFamilyIndices().transferIndex.value_or(compute);
        training_.initialize(device.getDevice(), device.getPhysicalDevice(),
                             device.getTransferQueue(), transfer, compute,
                             vulkanValidationLayersEnabled_);
        training_.loadMipNeRF360Dataset(textBufferString(settings_.paths.dataset),
                                        static_cast<uint32_t>(settings_.initialization.downscale));
        state_.datasetLoaded = true;
        setStatus("Dataset loaded: " + std::to_string(state_.frameCount) + " frames, " +
                  std::to_string(state_.imageWidth) + "x" + std::to_string(state_.imageHeight) +
                  ". Press Start Training to initialize and train.");
    } catch (const std::exception& error) {
        setError(error.what());
    }
}

void TrainingController::applyLiveSettings() {
    if (active_ || state_.running) return;
    training_.setPixelTo2DGSMode(
        static_cast<TrainingPixelTo2DGSMode>(std::clamp(settings_.run.pixelTo2DGSMode, 0, 6)));
    training_.setPixelTo2DGSMinSubgroupUtilization(settings_.run.pixelTo2DGSMinSubgroupUtilization);
    training_.setForwardCompositeMode(static_cast<TrainingForwardCompositeMode>(
        std::clamp(settings_.run.forwardCompositeMode, 0, 1)));
    training_.setTileMode(static_cast<TrainingTileMode>(std::clamp(settings_.run.tileMode, 0, 2)),
                          settings_.run.adaptiveTileSplitThreshold);
    training_.setAdaptiveTensorChildBatch(settings_.run.adaptiveTensorChildBatch);
    training_.setTensorSubgroupReduction(settings_.run.tensorSubgroupReduction);
    training_.setTensorGlobalPixelGrad(settings_.run.tensorGlobalPixelGrad);
    training_.setTensorRgbLatencyHiding(settings_.run.tensorRgbLatencyHiding);
    training_.setTensorEarlyCull(settings_.run.tensorEarlyCull);
    try {
        training_.setSsimMode(static_cast<TrainingSsimMode>(settings_.run.ssimMode));
    } catch (const std::exception& error) {
        settings_.run.ssimMode = static_cast<int>(training_.activeSsimMode());
        setError(error.what());
    }
    training_.setValidationInterval(settings_.schedule.validationInterval);
}

void TrainingController::applyConfiguration(const TrainingRunConfig& config) {
    if (!state_.datasetLoaded) {
        throw std::runtime_error("Load a training dataset before starting training");
    }
    const auto& scheduleSettings = config.schedule;
    const auto& initSettings = config.initialization;
    const auto& optimizerSettings = config.optimizer;
    const auto& densifySettings = config.densification;

    TrainingScheduleConfig schedule{};
    schedule.imageSelectionMode = scheduleSettings.imageSelectionMode == 1
                                      ? TrainingImageSelectionMode::Random
                                      : TrainingImageSelectionMode::Sequential;
    schedule.totalIterations = scheduleSettings.imageSelectionMode == 1 ? 30000u : 0u;
    schedule.randomSeed = initSettings.randomSeed;
    training_.setScheduleConfig(schedule);
    training_.setPixelTo2DGSMode(
        static_cast<TrainingPixelTo2DGSMode>(std::clamp(config.run.pixelTo2DGSMode, 0, 6)));
    training_.setPixelTo2DGSMinSubgroupUtilization(config.run.pixelTo2DGSMinSubgroupUtilization);
    training_.setForwardCompositeMode(static_cast<TrainingForwardCompositeMode>(
        std::clamp(config.run.forwardCompositeMode, 0, 1)));
    training_.setTileMode(static_cast<TrainingTileMode>(std::clamp(config.run.tileMode, 0, 2)),
                          config.run.adaptiveTileSplitThreshold);
    training_.setAdaptiveTensorChildBatch(config.run.adaptiveTensorChildBatch);
    training_.setTensorSubgroupReduction(config.run.tensorSubgroupReduction);
    training_.setTensorGlobalPixelGrad(config.run.tensorGlobalPixelGrad);
    training_.setTensorRgbLatencyHiding(config.run.tensorRgbLatencyHiding);
    training_.setTensorEarlyCull(config.run.tensorEarlyCull);
    training_.setSsimMode(static_cast<TrainingSsimMode>(config.run.ssimMode));
    training_.setValidationInterval(config.schedule.validationInterval);

    bool initializedModel = false;
    bool usedRandomInitialization = false;
    if (!training_.hasTrainableModel()) {
        TrainingInitializationConfig init{};
        init.randomGaussianCount = initSettings.gaussianCount;
        init.randomSeed = initSettings.randomSeed;
        init.initialOpacity = initSettings.opacity;
        init.sceneRadiusScale = initSettings.sceneRadiusScale;
        training_.initializeModelFromDataset(init);
        initializedModel = true;
        usedRandomInitialization = training_.usedRandomInitialization();
    }

    TrainingOptimizerConfig optimizer{};
    optimizer.positionLearningRate = optimizerSettings.positionLearningRate;
    optimizer.positionLearningRateFinal = optimizerSettings.positionLearningRateFinal;
    optimizer.positionLearningRateDelayMult = optimizerSettings.positionLearningRateDelayMult;
    optimizer.positionLearningRateDelaySteps = optimizerSettings.positionLearningRateDelaySteps;
    optimizer.positionLearningRateMaxSteps = optimizerSettings.positionLearningRateMaxSteps;
    optimizer.featureLearningRate = optimizerSettings.featureLearningRate;
    optimizer.featureRestLearningRate = optimizerSettings.featureRestLearningRate;
    optimizer.opacityLearningRate = optimizerSettings.opacityLearningRate;
    optimizer.scaleLearningRate = optimizerSettings.scaleLearningRate;
    optimizer.rotationLearningRate = optimizerSettings.rotationLearningRate;
    optimizer.beta1 = optimizerSettings.adamBeta1;
    optimizer.beta2 = optimizerSettings.adamBeta2;
    optimizer.epsilon = optimizerSettings.adamEpsilon;
    optimizer.gradClip = optimizerSettings.gradientClip;
    optimizer.lossDssimWeight = optimizerSettings.lossDssimWeight;
    optimizer.maxSHDegree = std::min(optimizerSettings.maxSHDegree, 3u);
    optimizer.shDegreeInterval = std::max(optimizerSettings.shDegreeInterval, 1u);
    training_.setOptimizerConfig(optimizer);

    TrainingDensificationConfig densify{};
    densify.enabled = densifySettings.enabled;
    densify.densifyFromIteration = densifySettings.fromIteration;
    densify.densifyUntilIteration =
        std::max(densifySettings.untilIteration, densifySettings.fromIteration);
    densify.densificationInterval = std::max(densifySettings.interval, 1u);
    densify.opacityResetInterval = std::max(densifySettings.opacityResetInterval, 1u);
    densify.maxGaussianCount = std::max(densifySettings.maxGaussians, 1u);
    densify.splitChildren = std::clamp(densifySettings.splitChildren, 2u, 8u);
    densify.gradientMode = static_cast<TrainingDensificationGradientMode>(
        std::clamp(densifySettings.gradientMode, 0, 1));
    densify.densifyGradThreshold = std::max(densifySettings.gradientThreshold, 0.0f);
    densify.minOpacity = std::clamp(densifySettings.minOpacity, 0.0f, 0.99f);
    densify.percentDense = std::clamp(densifySettings.percentDense, 0.0f, 1.0f);
    densify.screenSizePruneThreshold = std::max(densifySettings.screenPruneSize, 0.0f);
    densify.worldSizePruneThreshold = std::max(densifySettings.worldPruneSize, 0.0f);
    training_.setDensificationConfig(densify);

    if (initializedModel) {
        state_.initialized = false;
        setStatus("Training initialized with " + std::to_string(training_.gaussianCount()) +
                  " gaussians from " +
                  (usedRandomInitialization ? "random fallback" : "COLMAP sparse points"));
    }
}

void TrainingController::initializeTrainingIfNeeded() {
    if (state_.initialized) return;
    if (!trainingDevice_) {
        throw std::runtime_error("Training GPU is not initialized");
    }
    auto& device = *trainingDevice_;
    const auto computeIndex = device.getQueueFamilyIndices().computeIndex;
    if (!computeIndex) throw std::runtime_error("Training compute queue family is unavailable");
    const uint32_t compute = *computeIndex;
    const uint32_t transfer = device.getQueueFamilyIndices().transferIndex.value_or(compute);
    training_.initialize(device.getDevice(), device.getPhysicalDevice(), device.getTransferQueue(),
                         transfer, compute, vulkanValidationLayersEnabled_);
    training_.initializeTrainingRenderers(
        device.getDevice(), device.getPhysicalDevice(), device.getComputeQueue(), compute,
        std::max(training_.gaussianCount(), 1u),
        TrainingExtent{std::max(state_.imageWidth, 1u), std::max(state_.imageHeight, 1u)});
    state_.initialized = true;
}

void TrainingController::toggleTraining() {
    if (state_.running) {
        const TrainingUiSnapshot snapshot = snapshotForUi();
        if (snapshot.fixedBenchmark.active) {
            training_.stopFixedWorkloadBenchmark();
            state_.running = false;
            activeRunConfig_.reset();
            setStatus("Fixed workload benchmark stopped.");
        } else if (active_) {
            requestStop();
            setStatus("Training pause requested; waiting for the current step.");
        } else {
            stopTimer();
            state_.running = false;
            state_.pureActive = false;
            activeRunConfig_.reset();
            setStatus("Training paused.");
        }
        return;
    }

    try {
        if (!state_.datasetLoaded) {
            loadDataset();
            if (!state_.datasetLoaded) return;
        }
        activeRunConfig_ =
            std::make_unique<const TrainingRunConfig>(makeTrainingRunConfig(settings_));
        applyConfiguration(*activeRunConfig_);
        initializeTrainingIfNeeded();
        if (snapshotForUi().trainingIteration == 0u) resetTimer();
        training_.resetTensorEarlyCullRunStats();
        startTimer();
        saveSettings();
        startWorker();
        if (smokeOptions_)
            setStatus("Training smoke started; normal schedule 30000, pause at " +
                      std::to_string(smokeOptions_->stopIteration) +
                      ". Report: " + smokeOptions_->report.string());
        else
            setStatus(activeRunConfig_->run.pureTraining
                          ? "Pure asynchronous training started; only elapsed time updates until "
                            "completion."
                          : "Asynchronous training started.");
    } catch (const std::exception& error) {
        stopAndJoin();
        stopTimer();
        activeRunConfig_.reset();
        setError(error.what());
    }
}

void TrainingController::startFixedBenchmark(bool checkpointDiagnostics, bool checkpointWarp,
                                             bool tensorEarlyCull) {
    if (state_.running) return;
    try {
        if (!state_.datasetLoaded) {
            loadDataset();
            if (!state_.datasetLoaded) return;
        }
        activeRunConfig_ =
            std::make_unique<const TrainingRunConfig>(makeTrainingRunConfig(settings_));
        applyConfiguration(*activeRunConfig_);
        initializeTrainingIfNeeded();
        TrainingFixedBenchmarkConfig config{};
        config.frameIndex = activeRunConfig_->run.benchmarkFrame;
        config.warmupSteps = activeRunConfig_->run.benchmarkWarmupSteps;
        config.measuredSteps = activeRunConfig_->run.benchmarkMeasuredSteps;
        config.checkpointDiagnostics = checkpointDiagnostics;
        config.checkpointWarp = checkpointWarp;
        config.tensorEarlyCull = tensorEarlyCull;
        training_.startFixedWorkloadBenchmark(config);
        state_.running = true;
        state_.pureActive = false;
        setStatus("Fixed workload benchmark started.");
    } catch (const std::exception& error) {
        state_.running = false;
        state_.pureActive = false;
        activeRunConfig_.reset();
        setError(error.what());
    }
}

void TrainingController::exportFixedBenchmarkGradients() {
    if (state_.running || active_) {
        setError("Finish or stop training before exporting benchmark gradients.");
        return;
    }
    try {
        const auto& benchmark = training_.fixedWorkloadBenchmarkStats();
        if (!benchmark.complete) {
            throw std::runtime_error("Complete a Fixed Benchmark before exporting gradients");
        }
        const char* tileName = benchmark.tileMode == TrainingTileMode::AdaptiveSpatial ? "adaptive"
                               : benchmark.tileMode == TrainingTileMode::Fixed8        ? "fixed8"
                                                                                       : "fixed16";
        const std::filesystem::path directory =
            std::filesystem::path(textBufferString(settings_.paths.outputDirectory)) /
            "benchmark_gradients";
        std::string stem = "frame_" + std::to_string(benchmark.frameIndex) + "_iter_" +
                           std::to_string(benchmark.trainingIteration) + "_" + tileName;
        if (benchmark.pixelMode == TrainingPixelTo2DGSMode::CheckpointedWarp)
            stem += "_checkpoint_warp32_layout2";
        if (benchmark.tileMode == TrainingTileMode::AdaptiveSpatial &&
            benchmark.pixelMode == TrainingPixelTo2DGSMode::VkSplatTensor) {
            stem += "_batch_" + std::to_string(benchmark.adaptiveTensorChildBatch);
        }
        if (benchmark.tileMode == TrainingTileMode::Fixed16 &&
            benchmark.pixelMode == TrainingPixelTo2DGSMode::VkSplatTensor) {
            stem += benchmark.tensorSubgroupReduction ? "_tensor_subgroup" : "_tensor_shared";
            stem += benchmark.tensorGlobalPixelGrad ? "_grad_global" : "_grad_shared";
            if (benchmark.tensorGlobalPixelGrad) {
                stem += benchmark.tensorRgbLatencyHiding ? "_rgb_latency" : "_rgb_baseline";
            }
        }
        if (benchmark.pixelCullMethod == 1u) stem += "_pixel_cull_v1";
        stem += std::string("_") + trainingSsimModeFileTag(benchmark.ssimMode);
        std::filesystem::path path = directory / (stem + ".bin");
        for (uint32_t suffix = 2u; std::filesystem::exists(path); ++suffix) {
            path = directory / (stem + "_" + std::to_string(suffix) + ".bin");
        }
        training_.exportFixedBenchmarkGradients(path);
        setStatus("Saved benchmark gradients: " + std::filesystem::absolute(path).string());
    } catch (const std::exception& error) {
        setError(error.what());
    }
}

void TrainingController::captureOptimizerStepAndPause() {
    if (state_.running || active_) {
        setError("Pause and join training before requesting an S4 optimizer capture.");
        return;
    }
    try {
        if (worker_.joinable()) worker_.join();
        // Keep the existing model/configuration. Do not initialize, reset schedule, or
        // apply newly edited UI controls as an undocumented part of this diagnostic.
        const auto directory =
            std::filesystem::path(textBufferString(settings_.paths.outputDirectory)) /
            "benchmark_optimizer";
        const auto stem = "step_" + std::to_string(training_.trainingIteration() + 1u);
        auto path = directory / (stem + ".optimizer.bin");
        for (uint32_t suffix = 2; std::filesystem::exists(path); ++suffix)
            path = directory / (stem + "_" + std::to_string(suffix) + ".optimizer.bin");
        resultReady_.store(false, std::memory_order_release);
        active_ = true;
        stopRequested_ = false;
        state_.running = true;
        state_.pureActive = false;
        publishSnapshot();
        worker_ = std::jthread(
            [this, path](std::stop_token stopToken) { workerMain(stopToken, false, path); });
        setStatus("S4 diagnostic: trains ONE eligible ordinary step, pauses, and writes a large "
                  "pre/post capture. No subsequent step is queued.");
    } catch (const std::exception& error) {
        stopAndJoin();
        setError(error.what());
    }
}

void TrainingController::exportScanlineDiagnosticInputs() {
    if (state_.running || active_) {
        setError("Finish training/benchmark before exporting S3-2 inputs.");
        return;
    }
    try {
        const auto& b = training_.fixedWorkloadBenchmarkStats();
        const auto directory =
            std::filesystem::path(textBufferString(settings_.paths.outputDirectory)) /
            "benchmark_scanline";
        const auto stem = "frame_" + std::to_string(b.frameIndex) + "_iter_" +
                          std::to_string(b.trainingIteration);
        auto path = directory / (stem + ".scanline.bin");
        for (uint32_t suffix = 2u; std::filesystem::exists(path); ++suffix)
            path = directory / (stem + "_" + std::to_string(suffix) + ".scanline.bin");
        training_.exportScanlineDiagnosticInputs(path);
        setStatus("Saved S3-2 frozen diagnostic inputs: " +
                  std::filesystem::absolute(path).string());
    } catch (const std::exception& error) {
        setError(error.what());
    }
}

void TrainingController::exportAdaptiveTileDiagnostics() {
    if (state_.running || active_) {
        setError("Finish the benchmark before exporting adaptive tile diagnostics.");
        return;
    }
    try {
        const auto& benchmark = training_.fixedWorkloadBenchmarkStats();
        const std::filesystem::path directory =
            std::filesystem::path(textBufferString(settings_.paths.outputDirectory)) /
            "benchmark_tiles";
        const std::string stem = "frame_" + std::to_string(benchmark.frameIndex) + "_iter_" +
                                 std::to_string(benchmark.trainingIteration) + "_threshold_" +
                                 std::to_string(benchmark.adaptiveSplitThreshold) + "_batch_" +
                                 std::to_string(benchmark.adaptiveTensorChildBatch);
        std::filesystem::path path = directory / (stem + ".csv");
        for (uint32_t suffix = 2u;
             std::filesystem::exists(path) ||
             std::filesystem::exists(std::filesystem::path(path).replace_extension(".masks.bin"));
             ++suffix) {
            path = directory / (stem + "_" + std::to_string(suffix) + ".csv");
        }
        training_.exportAdaptiveTileDiagnostics(path);
        setStatus("Saved adaptive tile CSV and .masks.bin: " +
                  std::filesystem::absolute(path).string());
    } catch (const std::exception& error) {
        setError(error.what());
    }
}

void TrainingController::runSynchronousStep() {
    if (!state_.datasetLoaded) {
        throw std::runtime_error("Load a training dataset before starting training");
    }
    const auto before = snapshotForUi();
    if (!before.fixedBenchmark.active && before.trainingComplete) {
        finishTrainingRun();
        return;
    }
    training_.trainStep();
    ++state_.stepsDone;
    const auto after = snapshotForUi();
    if (!after.fixedBenchmark.active && after.fixedBenchmark.complete) {
        state_.running = false;
        activeRunConfig_.reset();
        setStatus("Fixed workload benchmark completed: " +
                  std::to_string(after.fixedBenchmark.measuredSteps) + " measured steps.");
    } else if (after.trainingComplete) {
        finishTrainingRun();
    }
}

void TrainingController::startWorker() {
    if (!activeRunConfig_) {
        throw std::logic_error("Training run configuration is not initialized");
    }
    const bool pure = activeRunConfig_->run.pureTraining;
    if (worker_.joinable()) worker_.join();
    smokeRun_.reset();
    if (smokeOptions_) {
        if (training_.trainingIteration() != 0 ||
            activeRunConfig_->schedule.imageSelectionMode != 1 ||
            training_.totalIterations() != 30000)
            throw std::runtime_error(
                "Training smoke requires a fresh model and normal 30000-step Random schedule");
        auto& context = Context::Instance();
        if (!context.isValidationLayersEnabled() || !context.isSynchronizationValidationEnabled())
            throw std::runtime_error(
                "Training smoke requires actual core and synchronization validation");
        if (Context::validationErrorCount())
            throw std::runtime_error("Vulkan validation errors already occurred; inspect startup "
                                     "logs before smoke training");
        const auto& gpu = trainingDevice_->getSelectedPhysicalDeviceInfo();
        std::ostringstream identity;
        identity << "{\"device\":" << schedulingJsonQuote(gpu.name)
                 << ",\"device_uuid\":" << schedulingJsonQuote(gpu.uuid)
                 << ",\"driver\":" << schedulingJsonQuote(gpu.driverVersion)
                 << ",\"build\":" << trainingSmokeBuildJson()
                 << ",\"core_validation_enabled\":true,\"synchronization_validation_enabled\":true"
                 << ",\"validation_errors_scope\":\"Entire shared application instance, including "
                    "presentation\""
                 << ",\"run_configuration_text\":"
                 << schedulingJsonQuote(TrainingSettingsStore::describeRun(*activeRunConfig_))
                 << ",\"initial_gaussians\":" << training_.gaussianCount()
                 << ",\"dataset_frames\":" << training_.datasetFrameCount()
                 << ",\"width\":" << training_.buffers().extent().width
                 << ",\"height\":" << training_.buffers().extent().height
                 << ",\"tensor_subgroup_active\":"
                 << (training_.tensorSubgroupReductionActive() ? "true" : "false")
                 << ",\"tensor_global_pixel_grad_active\":"
                 << (training_.tensorGlobalPixelGradActive() ? "true" : "false")
                 << ",\"shader_fnv1a64\":{";
        bool first = true;
        for (const char* name :
             {"train_forward_project", "train_forward_tile_count", "train_forward_tile_emit",
              "train_forward_workgroup", "train_full_sort_prepare_test",
              "train_full_sort_emit_test", "train_range_boundaries_sorted_keys_test",
              "train_forward_tile_sort", "train_loss", "train_backward_loss_to_pixel",
              "train_ssim_coefficients_horizontal", "train_backward_loss_to_pixel_separable",
              "train_backward_vksplat_tensor_subgroup_global_grad",
              "train_backward_2dgs_to_3dgs_optimizer", "train_optimizer"}) {
            if (!first) identity << ',';
            first = false;
            identity << schedulingJsonQuote(name) << ":\""
                     << optimizerShaderHash(
                            FileUtils::resolvePath(std::string("shaders/") + name + ".comp.spv"))
                     << '"';
        }
        identity << "},\"initial_state_fnv1a64\":\"" << training_.diagnosticTrainingStateHash()
                 << "\"}";
        smokeRun_ = std::make_unique<TrainingSmokeRun>(*smokeOptions_, identity.str());
    }
    training_.setStepObservationEnabled(smokeRun_ != nullptr);
    resultReady_.store(false, std::memory_order_release);
    active_ = true;
    stopRequested_ = false;
    state_.running = true;
    state_.pureActive = pure;
    publishSnapshot();
    worker_ =
        std::jthread([this, pure](std::stop_token stopToken) { workerMain(stopToken, pure); });
}

void TrainingController::requestStop() {
    if (!active_ || !worker_.joinable()) return;
    stopRequested_ = true;
    worker_.request_stop();
}

void TrainingController::stopAndJoin() {
    if (worker_.joinable()) {
        worker_.request_stop();
        worker_.join();
    }
    resultReady_.store(false, std::memory_order_release);
    active_ = false;
    stopRequested_ = false;
    state_.running = false;
    state_.pureActive = false;
    activeRunConfig_.reset();
    smokeRun_.reset();
    training_.setStepObservationEnabled(false);
}

void TrainingController::consumeWorkerResult() {
    if (!resultReady_.load(std::memory_order_acquire)) return;
    if (worker_.joinable()) worker_.join();
    TrainingWorkerResult result;
    {
        std::lock_guard lock(resultMutex_);
        result = result_;
    }
    resultReady_.store(false, std::memory_order_release);
    active_ = false;
    stopRequested_ = false;
    state_.running = false;
    stopTimerAt(result.finishedAt);
    training_.setStepObservationEnabled(false);
    smokeRun_.reset();

    const auto snapshot = snapshotForUi();
    state_.stepsDone = snapshot.trainingIteration;
    LOG_INFO("Training EarlyCull run: completed={}, active={}, fallback={}, selection={}",
             snapshot.tensorEarlyCullRun.completedSteps, snapshot.tensorEarlyCullRun.activeSteps,
             snapshot.tensorEarlyCullRun.fallbackSteps,
             trainingPixelCullStatusName(snapshot.tensorEarlyCullStatus));
    if (!result.error.empty()) {
        state_.pureActive = false;
        activeRunConfig_.reset();
        setError(result.error);
    } else if (!result.smokeReportPath.empty()) {
        state_.pureActive = false;
        activeRunConfig_.reset();
        setStatus(std::string(result.smokeLimitReached ? "Training smoke limit reached; paused at "
                                                       : "Training smoke paused at ") +
                  std::to_string(snapshot.trainingIteration) +
                  ". Report: " + result.smokeReportPath.string());
    } else if (!result.optimizerCapturePath.empty()) {
        state_.pureActive = false;
        activeRunConfig_.reset();
        setStatus("S4 capture saved; training paused at iteration " +
                  std::to_string(snapshot.trainingIteration) + ": " +
                  std::filesystem::absolute(result.optimizerCapturePath).string());
    } else if (result.completed) {
        try {
            finishTrainingRun();
        } catch (const std::exception& error) {
            state_.pureActive = false;
            activeRunConfig_.reset();
            setError(error.what());
        }
    } else {
        state_.pureActive = false;
        activeRunConfig_.reset();
        setStatus("Training paused at " + std::to_string(snapshot.trainingIteration) +
                  " iterations after " + elapsedText() + ".");
    }
}

void TrainingController::finishTrainingRun() {
    const bool completedPure = state_.pureActive;
    const std::filesystem::path runOutputPath =
        activeRunConfig_ ? activeRunConfig_->outputPath : outputPlyPath();
    state_.running = false;
    state_.pureActive = false;
    stopTimer();
    const auto snapshot = snapshotForUi();
    std::string status = "Training completed at " + std::to_string(snapshot.trainingIteration) +
                         " iterations in " + elapsedText() + ".";
    if (completedPure) {
        const std::filesystem::path path = runOutputPath;
        if (!training_.exportToPLY(path)) {
            throw std::runtime_error("Failed to export PLY: " + path.string());
        }
        status += " Saved PLY: " + path.string();
    }
    activeRunConfig_.reset();
    setStatus(std::move(status));
}

void TrainingController::exportDefaultModel() {
    exportModelTo(outputPlyPath());
}

void TrainingController::exportModelTo(const std::filesystem::path& requestedPath) {
    try {
        if (!training_.hasTrainableModel()) {
            throw std::runtime_error("No trained Gaussian model is available to export");
        }
        std::filesystem::path path = requestedPath;
        if (path.extension().empty()) path += ".ply";
        if (!training_.exportToPLY(path)) {
            throw std::runtime_error("Failed to export PLY: " + path.string());
        }
        setStatus("Saved PLY: " + path.string());
    } catch (const std::exception& error) {
        setError(error.what());
    }
}

void TrainingController::setOutputDirectory(const std::filesystem::path& path) {
    setTextBuffer(settings_.paths.outputDirectory, path.string());
}

void TrainingController::setOutputPath(const std::filesystem::path& path) {
    setTextBuffer(settings_.paths.outputDirectory, path.parent_path().string());
    setTextBuffer(settings_.paths.outputName, path.filename().string());
}

void TrainingController::saveSettings() {
    if (trainingDevice_) {
        const auto& gpu = trainingDevice_->getSelectedPhysicalDeviceInfo();
        settings_.gpuSelector = gpu.uuid.empty() ? std::to_string(gpu.vulkanIndex) : gpu.uuid;
    }
    TrainingSettingsStore::save(TrainingSettingsStore::defaultPath(), settings_);
}

void TrainingController::startTimer() {
    if (timerRunning_) return;
    timerStartedAt_ = std::chrono::steady_clock::now();
    timerRunning_ = true;
}

void TrainingController::stopTimer() {
    stopTimerAt(std::chrono::steady_clock::now());
}

void TrainingController::stopTimerAt(std::chrono::steady_clock::time_point endTime) {
    if (!timerRunning_) return;
    if (endTime > timerStartedAt_) {
        elapsedSeconds_ += std::chrono::duration<double>(endTime - timerStartedAt_).count();
    }
    timerRunning_ = false;
}

void TrainingController::resetTimer() {
    elapsedSeconds_ = 0.0;
    timerStartedAt_ = {};
    timerRunning_ = false;
}

double TrainingController::elapsedSeconds() const {
    if (!timerRunning_) return elapsedSeconds_;
    return elapsedSeconds_ +
           std::chrono::duration<double>(std::chrono::steady_clock::now() - timerStartedAt_)
               .count();
}

std::string TrainingController::elapsedText() const {
    const uint64_t totalMilliseconds =
        static_cast<uint64_t>(std::max(elapsedSeconds(), 0.0) * 1000.0);
    const uint64_t milliseconds = totalMilliseconds % 1000u;
    const uint64_t totalSeconds = totalMilliseconds / 1000u;
    const uint64_t seconds = totalSeconds % 60u;
    const uint64_t totalMinutes = totalSeconds / 60u;
    const uint64_t minutes = totalMinutes % 60u;
    const uint64_t hours = totalMinutes / 60u;
    std::ostringstream stream;
    stream << std::setfill('0') << std::setw(2) << hours << ':' << std::setw(2) << minutes << ':'
           << std::setw(2) << seconds << '.' << std::setw(3) << milliseconds;
    return stream.str();
}

void TrainingController::setStatus(std::string message) {
    state_.status = message;
    LOG_INFO("{}", message);
}

void TrainingController::setError(std::string message) {
    state_.error = message;
    state_.status = message;
    state_.errorPopupPending = true;
    LOG_ERROR("{}", message);
}

void TrainingController::syncDefaultOutputName() {
    const std::filesystem::path datasetPath(textBufferString(settings_.paths.dataset));
    std::string name = datasetPath.filename().string();
    if (name.empty()) name = "trained";
    setTextBuffer(settings_.paths.outputName, name + ".ply");
}

std::filesystem::path TrainingController::outputPlyPath() const {
    std::filesystem::path outputDir(textBufferString(settings_.paths.outputDirectory));
    std::filesystem::path outputName(textBufferString(settings_.paths.outputName));
    if (outputName.extension().empty()) outputName += ".ply";
    return outputDir / outputName;
}

TrainingUiSnapshot TrainingController::snapshotForUi() const {
    {
        std::lock_guard lock(snapshotMutex_);
        if (active_ && snapshotValid_) return snapshot_;
    }
    return captureSnapshot();
}

TrainingUiSnapshot TrainingController::captureSnapshot() const {
    TrainingUiSnapshot snapshot{};
    snapshot.rendererInitialized = training_.isRendererInitialized();
    snapshot.hasDataset = training_.hasDataset();
    snapshot.hasTrainableModel = training_.hasTrainableModel();
    snapshot.trainingComplete = training_.isTrainingComplete();
    snapshot.subgroupSupported = training_.subgroupPixelTo2DGSSupported();
    snapshot.tileGaussianSupported = training_.tileGaussianPixelTo2DGSSupported();
    snapshot.vkSplatPerSplatSupported = training_.vkSplatPerSplatSupported();
    snapshot.vkSplatTensorSupported = training_.vkSplatTensorSupported();
    snapshot.tensorSubgroupReductionSupported = training_.tensorSubgroupReductionSupported();
    snapshot.tensorSubgroupReductionActive = training_.tensorSubgroupReductionActive();
    snapshot.tensorGlobalPixelGradActive = training_.tensorGlobalPixelGradActive();
    snapshot.tensorRgbLatencyHidingActive = training_.tensorRgbLatencyHidingActive();
    snapshot.tensorEarlyCullStatus = training_.tensorEarlyCullStatus();
    snapshot.tensorEarlyCullRun = training_.tensorEarlyCullRunStats();
    snapshot.activeSsimMode = training_.activeSsimMode();
    snapshot.activePixelMode = training_.activePixelTo2DGSMode();
    snapshot.tileMode = training_.tileMode();
    snapshot.activeTileSize = training_.activeTileSize();
    snapshot.adaptiveTensorChildBatch = training_.adaptiveTensorChildBatch();
    snapshot.gaussianCount = training_.gaussianCount();
    snapshot.trainingIteration = training_.trainingIteration();
    snapshot.totalIterations = training_.totalIterations();
    snapshot.densificationIteration = training_.densificationStatsIteration();
    snapshot.currentFrameIndex = snapshot.hasDataset ? training_.currentFrameIndex() : 0u;
    snapshot.fixedBenchmark = training_.fixedWorkloadBenchmarkStats();
    snapshot.validation = training_.validationStats();
    snapshot.candidateProfile = training_.candidateProfileStats();
    snapshot.densification = training_.densificationStats();
    snapshot.profiling = training_.profilingStats();
    snapshot.imageCache = training_.imageCacheStats();
    snapshot.deviceImageCache = training_.deviceImageCacheStats();
    return snapshot;
}

void TrainingController::publishSnapshot() {
    TrainingUiSnapshot snapshot = captureSnapshot();
    std::lock_guard lock(snapshotMutex_);
    snapshot_ = std::move(snapshot);
    snapshotValid_ = true;
}

void TrainingController::workerMain(std::stop_token stopToken, bool pure,
                                    std::filesystem::path optimizerCapturePath) noexcept {
    try {
        TrainingWorkerResult result{};
        const auto loopStartedAt = std::chrono::steady_clock::now();
        result.pure = pure;
        try {
            if (!optimizerCapturePath.empty()) {
                if (stopToken.stop_requested())
                    throw std::runtime_error("S4 capture cancelled before any step was submitted");
                training_.captureOptimizerStepAndPause(optimizerCapturePath);
                result.optimizerCapturePath = optimizerCapturePath;
            } else {
                uint32_t stepsSinceSnapshot = 0u;
                while (!stopToken.stop_requested() && !training_.isTrainingComplete()) {
                    training_.trainStep();
                    if (smokeRun_) {
                        smokeRun_->observe(training_.lastStepObservation());
                        if (Context::validationErrorCount())
                            throw std::runtime_error(
                                "Smoke detected Vulkan validation errors; inspect log and report");
                        if (smokeRun_->reached(training_.trainingIteration())) {
                            result.smokeLimitReached = true;
                            break;
                        }
                    }
                    if (!pure && ++stepsSinceSnapshot >= kTrainingSnapshotPublishInterval) {
                        publishSnapshot();
                        stepsSinceSnapshot = 0u;
                    }
                }
                result.completed = training_.isTrainingComplete();
            }
        } catch (const std::exception& error) {
            result.error = error.what();
        } catch (...) {
            result.error = "Unknown exception in asynchronous training worker";
        }
        result.finishedAt = std::chrono::steady_clock::now();
        if (smokeRun_ && optimizerCapturePath.empty()) {
            std::optional<TrainingStateFiniteCheck> finalState;
            try {
                if (result.error.empty() && training_.trainingIteration()) {
                    finalState = training_.inspectTrainingStateFinite();
                    if (!finalState->passed())
                        result.error = "Smoke final loss/model finite inspection failed";
                }
                if (Context::validationErrorCount() && result.error.empty())
                    result.error =
                        "Vulkan validation errors occurred during smoke/final inspection";
                smokeRun_->finish(
                    result.smokeLimitReached ? "diagnostic_step_limit"
                                             : (!result.error.empty() ? "error" : "user_pause"),
                    result.error, Context::validationErrorCount(),
                    std::chrono::duration<double, std::milli>(result.finishedAt - loopStartedAt)
                        .count(),
                    finalState);
                result.smokeReportPath = smokeRun_->path();
                LOG_INFO("Training smoke report saved: {} (iteration {}, limit reached={})",
                         result.smokeReportPath.string(), training_.trainingIteration(),
                         result.smokeLimitReached);
            } catch (const std::exception& error) {
                if (result.error.empty()) result.error = error.what();
                if (!smokeRun_->written()) {
                    try {
                        smokeRun_->finish("inspection_or_report_error", result.error,
                                          Context::validationErrorCount(),
                                          std::chrono::duration<double, std::milli>(
                                              result.finishedAt - loopStartedAt)
                                              .count(),
                                          finalState);
                        result.smokeReportPath = smokeRun_->path();
                    } catch (...) {
                        LOG_ERROR("Training smoke error report could not be written.");
                    }
                }
            }
            if (!result.error.empty() && !result.smokeReportPath.empty())
                result.error += "; report: " + result.smokeReportPath.string();
        }
        try {
            publishSnapshot();
        } catch (const std::exception& error) {
            if (result.error.empty()) {
                result.error =
                    std::string("Failed to publish final training snapshot: ") + error.what();
            }
        }
        {
            std::lock_guard lock(resultMutex_);
            result_ = std::move(result);
        }
        resultReady_.store(true, std::memory_order_release);
    } catch (...) {
        // A failure while allocating the string-based error result cannot be
        // reported safely across the worker boundary.
        std::terminate();
    }
}

} // namespace vulkan3DGS
