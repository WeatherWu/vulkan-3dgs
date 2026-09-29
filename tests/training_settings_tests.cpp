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
    settings.run.pureTraining = true;
    settings.optimizer.positionLearningRate = 0.012345f;
    settings.densification.maxGaussians = 7654321;
    settings.densification.splitChildren = 7;
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
        loaded.run.pixelTo2DGSMode == 6 && loaded.run.pureTraining &&
        loaded.optimizer.positionLearningRate ==
            source.optimizer.positionLearningRate &&
        loaded.densification.maxGaussians == 7654321 &&
        loaded.densification.splitChildren == 7;
}

bool testValidationAndImmutableRunConfig() {
    TrainingSettings settings = populatedSettings();
    if (!TrainingSettingsStore::validate(settings)) return false;

    const TrainingRunConfig config =
        makeTrainingRunConfig(settings);
    setTextBuffer(settings.paths.dataset, "changed");
    setTextBuffer(settings.paths.outputName, "changed.ply");
    settings.run.pureTraining = false;
    settings.optimizer.positionLearningRate = 99.0f;
    if (config.datasetPath != "dataset with spaces" ||
        config.outputPath != std::filesystem::path("output dir") / "model.ply" ||
        !config.run.pureTraining ||
        config.optimizer.positionLearningRate != 0.012345f) {
        return false;
    }

    settings = populatedSettings();
    settings.optimizer.adamBeta1 = 1.0f;
    const TrainingSettingsValidation invalid =
        TrainingSettingsStore::validate(settings);
    return !invalid.valid && !invalid.message.empty();
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
        loaded.optimizer.positionLearningRate == 0.00042f &&
        loaded.densification.fromIteration == 321 &&
        loaded.run.pureTraining;
}

bool testClamp() {
    const auto path = std::filesystem::temp_directory_path() /
        "vulkan-3dgs-training-settings-clamp-test.cfg";
    {
        std::ofstream output(path, std::ios::trunc);
        output << "downscale 0\n"
               << "pixel_to_2dgs_mode 99\n"
               << "split_children 99\n";
    }
    TrainingSettings settings{};
    TrainingSettingsStore::load(path, settings);
    std::filesystem::remove(path);
    return settings.initialization.downscale == 1 &&
        settings.run.pixelTo2DGSMode == 6 &&
        settings.densification.splitChildren == 8;
}

} // namespace

int main() {
    const auto path = std::filesystem::temp_directory_path() /
        "vulkan-3dgs-training-settings-test.cfg";
    const bool passed = testRoundTrip(path) &&
        testValidationAndImmutableRunConfig() &&
        testLegacyKeys(path) &&
        testClamp();
    std::filesystem::remove(path);
    if (!passed) {
        std::cerr << "Training settings validation failed\n";
        return 1;
    }
    return 0;
}
