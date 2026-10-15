#include "training/app/training_settings.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>

using namespace vulkan3DGS;

namespace {

TrainingSettings populatedSettings() {
    TrainingSettings settings{};
    setTextBuffer(settings.paths.dataset, "dataset with spaces");
    setTextBuffer(settings.paths.outputDirectory, "output dir");
    setTextBuffer(settings.paths.outputName, "model.ply");
    settings.gpuSelector = "GPU-uuid";
    settings.initialization.downscale = 8;
    settings.run.pixelTo2DGSMode = 6;
    settings.run.tileMode = 0;
    settings.run.adaptiveTileSplitThreshold = 192;
    settings.run.adaptiveTensorChildBatch = 32;
    settings.run.tensorSubgroupReduction = true;
    settings.run.tensorGlobalPixelGrad = true;
    settings.run.tensorRgbLatencyHiding = true;
    settings.run.pureTraining = true;
    settings.optimizer.positionLearningRate = 0.012345f;
    settings.densification.maxGaussians = 7654321;
    settings.densification.splitChildren = 7;
    settings.densification.gradientMode = 1;
    return settings;
}

bool testRoundTrip(const std::filesystem::path& path) {
    const TrainingSettings source = populatedSettings();
    TrainingSettingsStore::save(path, source);
    TrainingSettings loaded{};
    TrainingSettingsStore::load(path, loaded);
    return textBufferString(loaded.paths.dataset) == "dataset with spaces" &&
        textBufferString(loaded.paths.outputDirectory) == "output dir" &&
        textBufferString(loaded.paths.outputName) == "model.ply" &&
        loaded.gpuSelector == "GPU-uuid" &&
        loaded.initialization.downscale == 8 &&
        loaded.run.pixelTo2DGSMode == 6 && loaded.run.tileMode == 0 &&
        loaded.run.adaptiveTileSplitThreshold == 192 &&
        loaded.run.adaptiveTensorChildBatch == 32 &&
        loaded.run.tensorSubgroupReduction &&
        loaded.run.tensorGlobalPixelGrad &&
        loaded.run.tensorRgbLatencyHiding &&
        loaded.run.pureTraining &&
        loaded.optimizer.positionLearningRate ==
            source.optimizer.positionLearningRate &&
        loaded.densification.maxGaussians == 7654321 &&
        loaded.densification.splitChildren == 7 &&
        loaded.densification.gradientMode == 1;
}

bool testValidationAndImmutableRunConfig() {
    if (TrainingSettings{}.run.tensorSubgroupReduction) return false;
    if (TrainingSettings{}.run.tensorGlobalPixelGrad) return false;
    if (TrainingSettings{}.run.tensorRgbLatencyHiding) return false;
    TrainingSettings settings = populatedSettings();
    if (!TrainingSettingsStore::validate(settings)) return false;

    const TrainingRunConfig config =
        makeTrainingRunConfig(settings);
    setTextBuffer(settings.paths.dataset, "changed");
    setTextBuffer(settings.paths.outputName, "changed.ply");
    settings.run.pureTraining = false;
    settings.run.tensorSubgroupReduction = false;
    settings.run.tensorGlobalPixelGrad = false;
    settings.run.tensorRgbLatencyHiding = false;
    settings.optimizer.positionLearningRate = 99.0f;
    if (config.datasetPath != "dataset with spaces" ||
        config.outputPath != std::filesystem::path("output dir") / "model.ply" ||
        !config.run.pureTraining || config.run.tileMode != 0 ||
        config.run.adaptiveTileSplitThreshold != 192 ||
        config.run.adaptiveTensorChildBatch != 32 ||
        !config.run.tensorSubgroupReduction ||
        !config.run.tensorGlobalPixelGrad ||
        !config.run.tensorRgbLatencyHiding ||
        config.optimizer.positionLearningRate != 0.012345f ||
        config.densification.gradientMode != 1) {
        return false;
    }

    settings = populatedSettings();
    settings.optimizer.adamBeta1 = 1.0f;
    const TrainingSettingsValidation invalidOptimizer =
        TrainingSettingsStore::validate(settings);
    if (invalidOptimizer.valid || invalidOptimizer.message.empty()) return false;

    settings = populatedSettings();
    settings.run.adaptiveTensorChildBatch = 12u;
    if (TrainingSettingsStore::validate(settings).valid) return false;

    settings = populatedSettings();
    settings.densification.gradientMode = 2;
    const TrainingSettingsValidation invalidGradientMode =
        TrainingSettingsStore::validate(settings);
    return !invalidGradientMode.valid && !invalidGradientMode.message.empty();
}

bool testLegacyKeys(const std::filesystem::path& path) {
    {
        std::ofstream output(path, std::ios::trunc);
        output << "version 1\n"
               << "dataset \"legacy scene\"\n"
               << "output_dir \"legacy output\"\n"
               << "output_name \"legacy.ply\"\n"
               << "training_mode 1\n"
               << "pixel_to_2dgs_mode 3\n"
               << "position_lr 0.00042\n"
               << "densify_from_iteration 321\n"
               << "pure_training 1\n";
    }
    TrainingSettings loaded{};
    TrainingSettingsStore::load(path, loaded);
    return textBufferString(loaded.paths.dataset) == "legacy scene" &&
        textBufferString(loaded.paths.outputDirectory) == "legacy output" &&
        textBufferString(loaded.paths.outputName) == "legacy.ply" &&
        loaded.schedule.imageSelectionMode == 1 &&
        loaded.run.pixelTo2DGSMode == 3 &&
        loaded.run.tileMode == 1 &&
        loaded.run.adaptiveTensorChildBatch == 8u &&
        !loaded.run.tensorSubgroupReduction &&
        !loaded.run.tensorGlobalPixelGrad &&
        !loaded.run.tensorRgbLatencyHiding &&
        loaded.optimizer.positionLearningRate == 0.00042f &&
        loaded.densification.fromIteration == 321 &&
        loaded.densification.gradientMode == 0 &&
        loaded.run.pureTraining;
}

bool testLegacyGlobalConfigUsesBaselineRgb(const std::filesystem::path& path) {
    {
        std::ofstream output(path, std::ios::trunc);
        output << "version 1\n"
               << "pixel_to_2dgs_mode 6\n"
               << "training_tile_mode 1\n"
               << "tensor_subgroup_reduction 1\n"
               << "tensor_global_pixel_grad 1\n";
    }
    TrainingSettings loaded{};
    TrainingSettingsStore::load(path, loaded);
    return loaded.run.tensorSubgroupReduction && loaded.run.tensorGlobalPixelGrad &&
           !loaded.run.tensorRgbLatencyHiding;
}

bool testClamp() {
    const auto path = std::filesystem::temp_directory_path() /
        "vulkan-3dgs-training-settings-clamp-test.cfg";
    {
        std::ofstream output(path, std::ios::trunc);
        output << "downscale 0\n"
               << "pixel_to_2dgs_mode 99\n"
               << "training_tile_mode 99\n"
               << "adaptive_tile_split_threshold 0\n"
               << "adaptive_tensor_child_batch 99\n"
               << "split_children 99\n"
               << "densification_gradient_mode 99\n";
    }
    TrainingSettings settings{};
    TrainingSettingsStore::load(path, settings);
    std::filesystem::remove(path);
    return settings.initialization.downscale == 1 &&
        settings.run.pixelTo2DGSMode == 6 &&
        settings.run.tileMode == 1 &&
        settings.run.adaptiveTileSplitThreshold == 1u &&
        settings.run.adaptiveTensorChildBatch == 8u &&
        settings.densification.splitChildren == 8 &&
        settings.densification.gradientMode == 1;
}

} // namespace

int main() {
    const auto path = std::filesystem::temp_directory_path() /
        "vulkan-3dgs-training-settings-test.cfg";
    const bool roundTrip = testRoundTrip(path);
    const bool validation = testValidationAndImmutableRunConfig();
    const bool legacyKeys = testLegacyKeys(path);
    const bool legacyGlobal = testLegacyGlobalConfigUsesBaselineRgb(path);
    const bool clamp = testClamp();
    std::filesystem::remove(path);
    if (!roundTrip || !validation || !legacyKeys || !legacyGlobal || !clamp) {
        std::cerr << "Training settings tests failed: roundTrip=" << roundTrip
                  << ", validation=" << validation << ", legacyKeys=" << legacyKeys
                  << ", legacyGlobal=" << legacyGlobal
                  << ", clamp=" << clamp << '\n';
        return 1;
    }
    return 0;
}
