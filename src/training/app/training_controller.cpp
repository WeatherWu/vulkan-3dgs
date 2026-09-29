#include "training/app/training_controller.hpp"

#include "context/device.hpp"
#include "utils/logger.hpp"

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace vulkan3DGS {

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

void TrainingController::initializeDevice(
    std::optional<std::string> commandLineSelector) {
    const bool usePersisted = !commandLineSelector && !settings_.gpuSelector.empty();
    std::optional<std::string> selector = commandLineSelector;
    if (usePersisted) selector = settings_.gpuSelector;
    try {
        createTrainingDevice(std::move(selector));
    } catch (const std::exception& error) {
        if (!usePersisted) throw;
        LOG_WARN("Saved training GPU is unavailable ({}); selecting the default GPU",
                 error.what());
        createTrainingDevice(std::nullopt);
    }
}

void TrainingController::tick() {
    consumeWorkerResult();
    if (!state_.running || active_) return;

    try {
        const TrainingRunSettings& run = activeRunConfig_
            ? activeRunConfig_->run
            : settings_.run;
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

void TrainingController::createTrainingDevice(
    std::optional<std::string> selector) {
    if (trainingDevice_) {
        throw std::runtime_error("Training device is already initialized");
    }
    auto device = std::make_unique<Device>(
        vk::SurfaceKHR{}, std::move(selector), DeviceRole::Training);
    device->createDevice();
    trainingDevice_ = std::move(device);
}

void TrainingController::replaceTrainingDevice(
    std::optional<std::string> selector) {
    auto replacement = std::make_unique<Device>(
        vk::SurfaceKHR{}, std::move(selector), DeviceRole::Training);
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
        settings_.gpuSelector = gpu.uuid.empty()
            ? std::to_string(gpu.vulkanIndex)
            : gpu.uuid;
        saveSettings();
        setStatus("Training GPU selected: [" + std::to_string(gpu.vulkanIndex) +
                  "] " + gpu.name + ". Dataset must be loaded again.");
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

void TrainingController::setDatasetPath(const std::filesystem::path& path,
                                        bool validateNow) {
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
        setStatus("Dataset valid: " + std::to_string(validation.frameCount) +
                  " frames, " + std::to_string(validation.width) + "x" +
                  std::to_string(validation.height));
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
        const uint32_t compute = device.getQueueFamilyIndices().computeIndex.value();
        const uint32_t transfer =
            device.getQueueFamilyIndices().transferIndex.value_or(compute);
        training_.initialize(device.getDevice(), device.getPhysicalDevice(),
                             device.getTransferQueue(), transfer, compute);
        training_.loadMipNeRF360Dataset(
            textBufferString(settings_.paths.dataset),
            static_cast<uint32_t>(settings_.initialization.downscale));
        state_.datasetLoaded = true;
        setStatus("Dataset loaded: " + std::to_string(state_.frameCount) +
                  " frames, " + std::to_string(state_.imageWidth) + "x" +
                  std::to_string(state_.imageHeight) +
                  ". Press Start Training to initialize and train.");
    } catch (const std::exception& error) {
        setError(error.what());
    }
}

void TrainingController::applyLiveSettings() {
    if (active_ || state_.running) return;
    training_.setPixelTo2DGSMode(static_cast<TrainingPixelTo2DGSMode>(
        std::clamp(settings_.run.pixelTo2DGSMode, 0, 6)));
    training_.setPixelTo2DGSMinSubgroupUtilization(
        settings_.run.pixelTo2DGSMinSubgroupUtilization);
    training_.setForwardCompositeMode(static_cast<TrainingForwardCompositeMode>(
        std::clamp(settings_.run.forwardCompositeMode, 0, 1)));
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
    schedule.totalIterations =
        scheduleSettings.imageSelectionMode == 1 ? 30000u : 0u;
    schedule.randomSeed = initSettings.randomSeed;
    training_.setScheduleConfig(schedule);
    training_.setPixelTo2DGSMode(static_cast<TrainingPixelTo2DGSMode>(
        std::clamp(config.run.pixelTo2DGSMode, 0, 6)));
    training_.setPixelTo2DGSMinSubgroupUtilization(
        config.run.pixelTo2DGSMinSubgroupUtilization);
    training_.setForwardCompositeMode(static_cast<TrainingForwardCompositeMode>(
        std::clamp(config.run.forwardCompositeMode, 0, 1)));
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
    optimizer.positionLearningRateDelayMult =
        optimizerSettings.positionLearningRateDelayMult;
    optimizer.positionLearningRateDelaySteps =
        optimizerSettings.positionLearningRateDelaySteps;
    optimizer.positionLearningRateMaxSteps =
        optimizerSettings.positionLearningRateMaxSteps;
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
    densify.opacityResetInterval =
        std::max(densifySettings.opacityResetInterval, 1u);
    densify.maxGaussianCount = std::max(densifySettings.maxGaussians, 1u);
    densify.splitChildren = std::clamp(densifySettings.splitChildren, 2u, 8u);
    densify.densifyGradThreshold = std::max(densifySettings.gradientThreshold, 0.0f);
    densify.minOpacity = std::clamp(densifySettings.minOpacity, 0.0f, 0.99f);
    densify.percentDense = std::clamp(densifySettings.percentDense, 0.0f, 1.0f);
    densify.screenSizePruneThreshold =
        std::max(densifySettings.screenPruneSize, 0.0f);
    densify.worldSizePruneThreshold =
        std::max(densifySettings.worldPruneSize, 0.0f);
    training_.setDensificationConfig(densify);

    if (initializedModel) {
        state_.initialized = false;
        setStatus("Training initialized with " +
                  std::to_string(training_.gaussianCount()) + " gaussians from " +
                  (usedRandomInitialization ? "random fallback" : "COLMAP sparse points"));
    }
}

void TrainingController::initializeTrainingIfNeeded() {
    if (state_.initialized) return;
    if (!trainingDevice_) {
        throw std::runtime_error("Training GPU is not initialized");
    }
    auto& device = *trainingDevice_;
    const uint32_t compute = device.getQueueFamilyIndices().computeIndex.value();
    const uint32_t transfer =
        device.getQueueFamilyIndices().transferIndex.value_or(compute);
    training_.initialize(device.getDevice(), device.getPhysicalDevice(),
                         device.getTransferQueue(), transfer, compute);
    training_.initializeTrainingRenderers(
        device.getDevice(), device.getPhysicalDevice(),
        device.getComputeQueue(), compute,
        std::max(training_.gaussianCount(), 1u),
        TrainingExtent{std::max(state_.imageWidth, 1u),
                       std::max(state_.imageHeight, 1u)});
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
        activeRunConfig_ = std::make_unique<const TrainingRunConfig>(
            makeTrainingRunConfig(settings_));
        applyConfiguration(*activeRunConfig_);
        initializeTrainingIfNeeded();
        if (snapshotForUi().trainingIteration == 0u) resetTimer();
        startTimer();
        saveSettings();
        startWorker();
        setStatus(activeRunConfig_->run.pureTraining
            ? "Pure asynchronous training started; only elapsed time updates until completion."
            : "Asynchronous training started.");
    } catch (const std::exception& error) {
        stopAndJoin();
        stopTimer();
        activeRunConfig_.reset();
        setError(error.what());
    }
}

void TrainingController::startFixedBenchmark() {
    if (state_.running) return;
    try {
        if (!state_.datasetLoaded) {
            loadDataset();
            if (!state_.datasetLoaded) return;
        }
        activeRunConfig_ = std::make_unique<const TrainingRunConfig>(
            makeTrainingRunConfig(settings_));
        applyConfiguration(*activeRunConfig_);
        initializeTrainingIfNeeded();
        TrainingFixedBenchmarkConfig config{};
        config.frameIndex = activeRunConfig_->run.benchmarkFrame;
        config.warmupSteps = activeRunConfig_->run.benchmarkWarmupSteps;
        config.measuredSteps = activeRunConfig_->run.benchmarkMeasuredSteps;
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
                  std::to_string(after.fixedBenchmark.measuredSteps) +
                  " measured steps.");
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
    resultReady_.store(false, std::memory_order_release);
    active_ = true;
    stopRequested_ = false;
    state_.running = true;
    state_.pureActive = pure;
    publishSnapshot();
    worker_ = std::jthread([this, pure](std::stop_token stopToken) {
        workerMain(stopToken, pure);
    });
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

    const auto snapshot = snapshotForUi();
    state_.stepsDone = snapshot.trainingIteration;
    if (!result.error.empty()) {
        state_.pureActive = false;
        activeRunConfig_.reset();
        setError(result.error);
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
        setStatus("Training paused at " +
                  std::to_string(snapshot.trainingIteration) +
                  " iterations after " + elapsedText() + ".");
    }
}

void TrainingController::finishTrainingRun() {
    const bool completedPure = state_.pureActive;
    const std::filesystem::path runOutputPath = activeRunConfig_
        ? activeRunConfig_->outputPath
        : outputPlyPath();
    state_.running = false;
    state_.pureActive = false;
    stopTimer();
    const auto snapshot = snapshotForUi();
    std::string status = "Training completed at " +
        std::to_string(snapshot.trainingIteration) +
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
        settings_.gpuSelector = gpu.uuid.empty()
            ? std::to_string(gpu.vulkanIndex)
            : gpu.uuid;
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

void TrainingController::stopTimerAt(
    std::chrono::steady_clock::time_point endTime) {
    if (!timerRunning_) return;
    if (endTime > timerStartedAt_) {
        elapsedSeconds_ +=
            std::chrono::duration<double>(endTime - timerStartedAt_).count();
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
    return elapsedSeconds_ + std::chrono::duration<double>(
        std::chrono::steady_clock::now() - timerStartedAt_).count();
}

std::string TrainingController::elapsedText() const {
    const uint64_t totalMilliseconds = static_cast<uint64_t>(
        std::max(elapsedSeconds(), 0.0) * 1000.0);
    const uint64_t milliseconds = totalMilliseconds % 1000u;
    const uint64_t totalSeconds = totalMilliseconds / 1000u;
    const uint64_t seconds = totalSeconds % 60u;
    const uint64_t totalMinutes = totalSeconds / 60u;
    const uint64_t minutes = totalMinutes % 60u;
    const uint64_t hours = totalMinutes / 60u;
    std::ostringstream stream;
    stream << std::setfill('0') << std::setw(2) << hours << ':'
           << std::setw(2) << minutes << ':' << std::setw(2) << seconds << '.'
           << std::setw(3) << milliseconds;
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
    const std::filesystem::path datasetPath(
        textBufferString(settings_.paths.dataset));
    std::string name = datasetPath.filename().string();
    if (name.empty()) name = "trained";
    setTextBuffer(settings_.paths.outputName, name + ".ply");
}

std::filesystem::path TrainingController::outputPlyPath() const {
    std::filesystem::path outputDir(
        textBufferString(settings_.paths.outputDirectory));
    std::filesystem::path outputName(
        textBufferString(settings_.paths.outputName));
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
    snapshot.activePixelMode = training_.activePixelTo2DGSMode();
    snapshot.gaussianCount = training_.gaussianCount();
    snapshot.trainingIteration = training_.trainingIteration();
    snapshot.totalIterations = training_.totalIterations();
    snapshot.densificationIteration = training_.densificationStatsIteration();
    snapshot.currentFrameIndex =
        snapshot.hasDataset ? training_.currentFrameIndex() : 0u;
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

void TrainingController::workerMain(std::stop_token stopToken, bool pure) {
    TrainingWorkerResult result{};
    result.pure = pure;
    try {
        uint32_t stepsSinceSnapshot = 0u;
        while (!stopToken.stop_requested() && !training_.isTrainingComplete()) {
            training_.trainStep();
            if (!pure && ++stepsSinceSnapshot >= 32u) {
                publishSnapshot();
                stepsSinceSnapshot = 0u;
            }
        }
        result.completed = training_.isTrainingComplete();
    } catch (const std::exception& error) {
        result.error = error.what();
    } catch (...) {
        result.error = "Unknown exception in asynchronous training worker";
    }
    result.finishedAt = std::chrono::steady_clock::now();
    try {
        publishSnapshot();
    } catch (const std::exception& error) {
        if (result.error.empty()) {
            result.error = std::string("Failed to publish final training snapshot: ") +
                           error.what();
        }
    }
    {
        std::lock_guard lock(resultMutex_);
        result_ = std::move(result);
    }
    resultReady_.store(true, std::memory_order_release);
}

} // namespace vulkan3DGS
