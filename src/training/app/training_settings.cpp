#include "training/app/training_settings.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace vulkan3DGS {
namespace {

std::optional<std::filesystem::path> environmentDirectory(const char* name) {
#ifdef _WIN32
    char* value = nullptr;
    size_t length = 0;
    if (_dupenv_s(&value, &length, name) != 0 || !value || length <= 1u) {
        std::free(value);
        return std::nullopt;
    }
    std::filesystem::path path(value);
    std::free(value);
    return path;
#else
    const char* value = std::getenv(name);
    return value && value[0] != '\0' ? std::optional<std::filesystem::path>(value) : std::nullopt;
#endif
}

const std::unordered_set<std::string>& trainingKeys() {
    static const std::unordered_set<std::string> keys{"dataset",
                                                      "output_dir",
                                                      "output_name",
                                                      "gpu_selector",
                                                      "downscale",
                                                      "initial_gaussians",
                                                      "random_seed",
                                                      "initial_opacity",
                                                      "scene_radius_scale",
                                                      "training_mode",
                                                      "pixel_to_2dgs_mode",
                                                      "pixel_to_2dgs_min_subgroup_utilization",
                                                      "forward_composite_mode",
                                                      "total_iterations",
                                                      "steps_per_frame",
                                                      "validation_interval",
                                                      "benchmark_frame",
                                                      "benchmark_warmup",
                                                      "benchmark_measured",
                                                      "pure_training",
                                                      "position_lr",
                                                      "position_lr_final",
                                                      "position_lr_delay_mult",
                                                      "position_lr_delay_steps",
                                                      "position_lr_max_steps",
                                                      "feature_lr",
                                                      "feature_rest_lr",
                                                      "opacity_lr",
                                                      "scale_lr",
                                                      "rotation_lr",
                                                      "adam_beta1",
                                                      "adam_beta2",
                                                      "adam_epsilon",
                                                      "grad_clip",
                                                      "loss_dssim_weight",
                                                      "max_sh_degree",
                                                      "sh_degree_interval",
                                                      "densification_enabled",
                                                      "densify_from_iteration",
                                                      "densify_until_iteration",
                                                      "densification_interval",
                                                      "opacity_reset_interval",
                                                      "max_gaussians",
                                                      "split_children",
                                                      "densify_grad_threshold",
                                                      "min_opacity",
                                                      "percent_dense",
                                                      "screen_prune_size",
                                                      "world_prune_size"};
    return keys;
}

std::vector<std::string> unownedLines(const std::filesystem::path& path) {
    std::ifstream input(path);
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(input, line)) {
        std::istringstream row(line);
        std::string key;
        row >> key;
        if (!key.empty() && key != "version" && !trainingKeys().contains(key)) {
            lines.push_back(line);
        }
    }
    return lines;
}

} // namespace

std::filesystem::path TrainingSettingsStore::defaultPath() {
#ifdef _WIN32
    if (auto appData = environmentDirectory("APPDATA")) {
        return *appData / "vulkan-3dgs" / "training-settings.cfg";
    }
#else
    if (auto configHome = environmentDirectory("XDG_CONFIG_HOME")) {
        return *configHome / "vulkan-3dgs" / "training-settings.cfg";
    }
    if (auto home = environmentDirectory("HOME")) {
        return *home / ".config" / "vulkan-3dgs" / "training-settings.cfg";
    }
#endif
    return std::filesystem::current_path() / ".vulkan-3dgs-training-settings.cfg";
}

void TrainingSettingsStore::load(const std::filesystem::path& path, TrainingSettings& settings) {
    std::ifstream input(path);
    if (!input.is_open()) return;

    std::string line;
    while (std::getline(input, line)) {
        std::istringstream row(line);
        std::string key;
        row >> key;
        if (key.empty() || key[0] == '#') continue;

        if (key == "dataset") {
            std::string value;
            if (row >> std::quoted(value)) setTextBuffer(settings.paths.dataset, value);
        } else if (key == "output_dir") {
            std::string value;
            if (row >> std::quoted(value)) setTextBuffer(settings.paths.outputDirectory, value);
        } else if (key == "output_name") {
            std::string value;
            if (row >> std::quoted(value)) setTextBuffer(settings.paths.outputName, value);
        } else if (key == "gpu_selector")
            row >> std::quoted(settings.gpuSelector);
        else if (key == "downscale")
            row >> settings.initialization.downscale;
        else if (key == "initial_gaussians")
            row >> settings.initialization.gaussianCount;
        else if (key == "random_seed")
            row >> settings.initialization.randomSeed;
        else if (key == "initial_opacity")
            row >> settings.initialization.opacity;
        else if (key == "scene_radius_scale")
            row >> settings.initialization.sceneRadiusScale;
        else if (key == "training_mode")
            row >> settings.schedule.imageSelectionMode;
        else if (key == "total_iterations")
            row >> settings.schedule.totalIterations;
        else if (key == "validation_interval")
            row >> settings.schedule.validationInterval;
        else if (key == "pixel_to_2dgs_mode")
            row >> settings.run.pixelTo2DGSMode;
        else if (key == "pixel_to_2dgs_min_subgroup_utilization")
            row >> settings.run.pixelTo2DGSMinSubgroupUtilization;
        else if (key == "forward_composite_mode")
            row >> settings.run.forwardCompositeMode;
        else if (key == "steps_per_frame")
            row >> settings.run.benchmarkStepsPerFrame;
        else if (key == "benchmark_frame")
            row >> settings.run.benchmarkFrame;
        else if (key == "benchmark_warmup")
            row >> settings.run.benchmarkWarmupSteps;
        else if (key == "benchmark_measured")
            row >> settings.run.benchmarkMeasuredSteps;
        else if (key == "pure_training") {
            int value = 0;
            if (row >> value) settings.run.pureTraining = value != 0;
        } else if (key == "position_lr")
            row >> settings.optimizer.positionLearningRate;
        else if (key == "position_lr_final")
            row >> settings.optimizer.positionLearningRateFinal;
        else if (key == "position_lr_delay_mult")
            row >> settings.optimizer.positionLearningRateDelayMult;
        else if (key == "position_lr_delay_steps")
            row >> settings.optimizer.positionLearningRateDelaySteps;
        else if (key == "position_lr_max_steps")
            row >> settings.optimizer.positionLearningRateMaxSteps;
        else if (key == "feature_lr")
            row >> settings.optimizer.featureLearningRate;
        else if (key == "feature_rest_lr")
            row >> settings.optimizer.featureRestLearningRate;
        else if (key == "opacity_lr")
            row >> settings.optimizer.opacityLearningRate;
        else if (key == "scale_lr")
            row >> settings.optimizer.scaleLearningRate;
        else if (key == "rotation_lr")
            row >> settings.optimizer.rotationLearningRate;
        else if (key == "adam_beta1")
            row >> settings.optimizer.adamBeta1;
        else if (key == "adam_beta2")
            row >> settings.optimizer.adamBeta2;
        else if (key == "adam_epsilon")
            row >> settings.optimizer.adamEpsilon;
        else if (key == "grad_clip")
            row >> settings.optimizer.gradientClip;
        else if (key == "loss_dssim_weight")
            row >> settings.optimizer.lossDssimWeight;
        else if (key == "max_sh_degree")
            row >> settings.optimizer.maxSHDegree;
        else if (key == "sh_degree_interval")
            row >> settings.optimizer.shDegreeInterval;
        else if (key == "densification_enabled") {
            int value = 0;
            if (row >> value) settings.densification.enabled = value != 0;
        } else if (key == "densify_from_iteration")
            row >> settings.densification.fromIteration;
        else if (key == "densify_until_iteration")
            row >> settings.densification.untilIteration;
        else if (key == "densification_interval")
            row >> settings.densification.interval;
        else if (key == "opacity_reset_interval")
            row >> settings.densification.opacityResetInterval;
        else if (key == "max_gaussians")
            row >> settings.densification.maxGaussians;
        else if (key == "split_children")
            row >> settings.densification.splitChildren;
        else if (key == "densify_grad_threshold")
            row >> settings.densification.gradientThreshold;
        else if (key == "min_opacity")
            row >> settings.densification.minOpacity;
        else if (key == "percent_dense")
            row >> settings.densification.percentDense;
        else if (key == "screen_prune_size")
            row >> settings.densification.screenPruneSize;
        else if (key == "world_prune_size")
            row >> settings.densification.worldPruneSize;
    }
    clamp(settings);
}

void TrainingSettingsStore::save(const std::filesystem::path& path,
                                 const TrainingSettings& settings) {
    const auto preserved = unownedLines(path);
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::trunc);
    if (!output.is_open()) {
        throw std::runtime_error("Could not save training settings to " + path.string());
    }
    output << "version 1\n";
    for (const auto& line : preserved)
        output << line << '\n';
    write(output, settings);
}

void TrainingSettingsStore::write(std::ostream& output, const TrainingSettings& settings) {
    const auto& paths = settings.paths;
    const auto& init = settings.initialization;
    const auto& schedule = settings.schedule;
    const auto& run = settings.run;
    const auto& optimizer = settings.optimizer;
    const auto& densify = settings.densification;
    output << std::setprecision(std::numeric_limits<float>::max_digits10);
    output << "dataset " << std::quoted(textBufferString(paths.dataset)) << '\n';
    output << "output_dir " << std::quoted(textBufferString(paths.outputDirectory)) << '\n';
    output << "output_name " << std::quoted(textBufferString(paths.outputName)) << '\n';
    output << "gpu_selector " << std::quoted(settings.gpuSelector) << '\n';
    output << "downscale " << init.downscale << '\n';
    output << "initial_gaussians " << init.gaussianCount << '\n';
    output << "random_seed " << init.randomSeed << '\n';
    output << "initial_opacity " << init.opacity << '\n';
    output << "scene_radius_scale " << init.sceneRadiusScale << '\n';
    output << "training_mode " << schedule.imageSelectionMode << '\n';
    output << "total_iterations " << schedule.totalIterations << '\n';
    output << "validation_interval " << schedule.validationInterval << '\n';
    output << "pixel_to_2dgs_mode " << run.pixelTo2DGSMode << '\n';
    output << "pixel_to_2dgs_min_subgroup_utilization " << run.pixelTo2DGSMinSubgroupUtilization
           << '\n';
    output << "forward_composite_mode " << run.forwardCompositeMode << '\n';
    output << "steps_per_frame " << run.benchmarkStepsPerFrame << '\n';
    output << "benchmark_frame " << run.benchmarkFrame << '\n';
    output << "benchmark_warmup " << run.benchmarkWarmupSteps << '\n';
    output << "benchmark_measured " << run.benchmarkMeasuredSteps << '\n';
    output << "pure_training " << (run.pureTraining ? 1 : 0) << '\n';
    output << "position_lr " << optimizer.positionLearningRate << '\n';
    output << "position_lr_final " << optimizer.positionLearningRateFinal << '\n';
    output << "position_lr_delay_mult " << optimizer.positionLearningRateDelayMult << '\n';
    output << "position_lr_delay_steps " << optimizer.positionLearningRateDelaySteps << '\n';
    output << "position_lr_max_steps " << optimizer.positionLearningRateMaxSteps << '\n';
    output << "feature_lr " << optimizer.featureLearningRate << '\n';
    output << "feature_rest_lr " << optimizer.featureRestLearningRate << '\n';
    output << "opacity_lr " << optimizer.opacityLearningRate << '\n';
    output << "scale_lr " << optimizer.scaleLearningRate << '\n';
    output << "rotation_lr " << optimizer.rotationLearningRate << '\n';
    output << "adam_beta1 " << optimizer.adamBeta1 << '\n';
    output << "adam_beta2 " << optimizer.adamBeta2 << '\n';
    output << "adam_epsilon " << optimizer.adamEpsilon << '\n';
    output << "grad_clip " << optimizer.gradientClip << '\n';
    output << "loss_dssim_weight " << optimizer.lossDssimWeight << '\n';
    output << "max_sh_degree " << optimizer.maxSHDegree << '\n';
    output << "sh_degree_interval " << optimizer.shDegreeInterval << '\n';
    output << "densification_enabled " << (densify.enabled ? 1 : 0) << '\n';
    output << "densify_from_iteration " << densify.fromIteration << '\n';
    output << "densify_until_iteration " << densify.untilIteration << '\n';
    output << "densification_interval " << densify.interval << '\n';
    output << "opacity_reset_interval " << densify.opacityResetInterval << '\n';
    output << "max_gaussians " << densify.maxGaussians << '\n';
    output << "split_children " << densify.splitChildren << '\n';
    output << "densify_grad_threshold " << densify.gradientThreshold << '\n';
    output << "min_opacity " << densify.minOpacity << '\n';
    output << "percent_dense " << densify.percentDense << '\n';
    output << "screen_prune_size " << densify.screenPruneSize << '\n';
    output << "world_prune_size " << densify.worldPruneSize << '\n';
}

TrainingSettingsValidation TrainingSettingsStore::validate(const TrainingSettings& settings) {
    const auto invalid = [](std::string message) {
        return TrainingSettingsValidation{false, std::move(message)};
    };
    const auto finite = [](float value) { return std::isfinite(value); };
    const auto nonNegative = [&](float value) { return finite(value) && value >= 0.0f; };

    if (textBufferString(settings.paths.dataset).empty()) {
        return invalid("Dataset path must not be empty");
    }
    if (textBufferString(settings.paths.outputName).empty()) {
        return invalid("Output PLY name must not be empty");
    }

    const auto& init = settings.initialization;
    if (init.downscale < 1 || init.downscale > 16) {
        return invalid("Dataset downscale must be in [1, 16]");
    }
    if (init.gaussianCount == 0u) {
        return invalid("Initial Gaussian count must be greater than zero");
    }
    if (!finite(init.opacity) || init.opacity < 0.001f || init.opacity > 0.99f) {
        return invalid("Initial opacity must be in [0.001, 0.99]");
    }
    if (!finite(init.sceneRadiusScale) || init.sceneRadiusScale <= 0.0f) {
        return invalid("Scene radius scale must be positive");
    }

    const auto& schedule = settings.schedule;
    if (schedule.imageSelectionMode < 0 || schedule.imageSelectionMode > 1) {
        return invalid("Training image selection mode is invalid");
    }

    const auto& run = settings.run;
    if (run.pixelTo2DGSMode < 0 || run.pixelTo2DGSMode > 6) {
        return invalid("Pixel-to-2DGS mode is invalid");
    }
    if (!finite(run.pixelTo2DGSMinSubgroupUtilization) ||
        run.pixelTo2DGSMinSubgroupUtilization < 0.0f ||
        run.pixelTo2DGSMinSubgroupUtilization > 1.0f) {
        return invalid("Subgroup utilization must be in [0, 1]");
    }
    if (run.forwardCompositeMode < 0 || run.forwardCompositeMode > 1) {
        return invalid("Forward composite mode is invalid");
    }
    if (run.benchmarkStepsPerFrame == 0u || run.benchmarkMeasuredSteps == 0u) {
        return invalid("Benchmark step counts must be greater than zero");
    }

    const auto& optimizer = settings.optimizer;
    if (!nonNegative(optimizer.positionLearningRate) ||
        !nonNegative(optimizer.positionLearningRateFinal) ||
        !finite(optimizer.positionLearningRateDelayMult) ||
        optimizer.positionLearningRateDelayMult < 0.0f ||
        optimizer.positionLearningRateDelayMult > 1.0f ||
        !nonNegative(optimizer.positionLearningRateDelaySteps) ||
        !finite(optimizer.positionLearningRateMaxSteps) ||
        optimizer.positionLearningRateMaxSteps <= 0.0f ||
        !nonNegative(optimizer.featureLearningRate) ||
        !nonNegative(optimizer.featureRestLearningRate) ||
        !nonNegative(optimizer.opacityLearningRate) || !nonNegative(optimizer.scaleLearningRate) ||
        !nonNegative(optimizer.rotationLearningRate)) {
        return invalid("Optimizer learning-rate configuration is invalid");
    }
    if (!finite(optimizer.adamBeta1) || optimizer.adamBeta1 < 0.0f || optimizer.adamBeta1 >= 1.0f ||
        !finite(optimizer.adamBeta2) || optimizer.adamBeta2 < 0.0f || optimizer.adamBeta2 >= 1.0f ||
        !finite(optimizer.adamEpsilon) || optimizer.adamEpsilon <= 0.0f) {
        return invalid("Adam configuration is invalid");
    }
    if (!nonNegative(optimizer.gradientClip) || !finite(optimizer.lossDssimWeight) ||
        optimizer.lossDssimWeight < 0.0f || optimizer.lossDssimWeight > 1.0f ||
        optimizer.maxSHDegree > 3u || optimizer.shDegreeInterval == 0u) {
        return invalid("Optimizer limits are invalid");
    }

    const auto& densify = settings.densification;
    if (densify.untilIteration < densify.fromIteration || densify.interval == 0u ||
        densify.opacityResetInterval == 0u || densify.maxGaussians == 0u ||
        densify.splitChildren < 2u || densify.splitChildren > 8u ||
        !nonNegative(densify.gradientThreshold) || !finite(densify.minOpacity) ||
        densify.minOpacity < 0.0f || densify.minOpacity > 0.99f || !finite(densify.percentDense) ||
        densify.percentDense < 0.0f || densify.percentDense > 1.0f ||
        !nonNegative(densify.screenPruneSize) || !nonNegative(densify.worldPruneSize)) {
        return invalid("Densification configuration is invalid");
    }
    return {};
}

TrainingRunConfig makeTrainingRunConfig(const TrainingSettings& settings) {
    const TrainingSettingsValidation validation = TrainingSettingsStore::validate(settings);
    if (!validation) {
        throw std::invalid_argument(validation.message);
    }

    std::filesystem::path outputName(textBufferString(settings.paths.outputName));
    if (outputName.extension().empty()) outputName += ".ply";
    return TrainingRunConfig{
        .datasetPath = textBufferString(settings.paths.dataset),
        .outputPath =
            std::filesystem::path(textBufferString(settings.paths.outputDirectory)) / outputName,
        .gpuSelector = settings.gpuSelector,
        .initialization = settings.initialization,
        .schedule = settings.schedule,
        .optimizer = settings.optimizer,
        .densification = settings.densification,
        .run = settings.run,
    };
}

void TrainingSettingsStore::clamp(TrainingSettings& settings) {
    settings.initialization.downscale = std::clamp(settings.initialization.downscale, 1, 16);
    settings.schedule.imageSelectionMode = std::clamp(settings.schedule.imageSelectionMode, 0, 1);
    settings.run.pixelTo2DGSMode = std::clamp(settings.run.pixelTo2DGSMode, 0, 6);
    settings.run.pixelTo2DGSMinSubgroupUtilization =
        std::clamp(settings.run.pixelTo2DGSMinSubgroupUtilization, 0.0f, 1.0f);
    settings.run.forwardCompositeMode = std::clamp(settings.run.forwardCompositeMode, 0, 1);
    settings.run.benchmarkStepsPerFrame = std::max(settings.run.benchmarkStepsPerFrame, 1u);
    settings.run.benchmarkMeasuredSteps = std::max(settings.run.benchmarkMeasuredSteps, 1u);
    settings.optimizer.maxSHDegree = std::min(settings.optimizer.maxSHDegree, 3u);
    settings.optimizer.shDegreeInterval = std::max(settings.optimizer.shDegreeInterval, 1u);
    settings.densification.splitChildren = std::clamp(settings.densification.splitChildren, 2u, 8u);
}

} // namespace vulkan3DGS
