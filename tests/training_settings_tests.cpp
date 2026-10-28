#include "training/app/training_settings.hpp"
#include "training/core/training_morton_policy.hpp"
#include "training/core/training_ssim_mode.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <utility>

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
    settings.run.tensorEarlyCull = true;
    settings.run.fasterSort = true;
    settings.run.mortonOrdering = true;
    settings.run.ssimMode = 2;
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
           loaded.gpuSelector == "GPU-uuid" && loaded.initialization.downscale == 8 &&
           loaded.run.pixelTo2DGSMode == 6 && loaded.run.tileMode == 0 &&
           loaded.run.adaptiveTileSplitThreshold == 192 &&
           loaded.run.adaptiveTensorChildBatch == 32 && loaded.run.tensorSubgroupReduction &&
           loaded.run.tensorGlobalPixelGrad && loaded.run.tensorRgbLatencyHiding &&
           loaded.run.tensorEarlyCull && loaded.run.fasterSort && loaded.run.mortonOrdering &&
           loaded.run.ssimMode == 2 && loaded.run.pureTraining &&
           loaded.optimizer.positionLearningRate == source.optimizer.positionLearningRate &&
           loaded.densification.maxGaussians == 7654321 &&
           loaded.densification.splitChildren == 7 && loaded.densification.gradientMode == 1;
}

bool testValidationAndImmutableRunConfig() {
    if (TrainingSettings{}.run.tensorSubgroupReduction) return false;
    if (TrainingSettings{}.run.tensorGlobalPixelGrad) return false;
    if (TrainingSettings{}.run.tensorRgbLatencyHiding) return false;
    if (TrainingSettings{}.run.tensorEarlyCull) return false;
    if (TrainingSettings{}.run.mortonOrdering) return false;
    if (TrainingSettings{}.run.ssimMode != 0) return false;
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
    settings.run.tensorEarlyCull = false;
    settings.run.mortonOrdering = false;
    settings.run.ssimMode = 0;
    settings.optimizer.positionLearningRate = 99.0f;
    if (config.datasetPath != "dataset with spaces" ||
        config.outputPath != std::filesystem::path("output dir") / "model.ply" ||
        !config.run.pureTraining || config.run.tileMode != 0 ||
        config.run.adaptiveTileSplitThreshold != 192 || config.run.adaptiveTensorChildBatch != 32 ||
        !config.run.tensorSubgroupReduction || !config.run.tensorGlobalPixelGrad ||
        !config.run.tensorRgbLatencyHiding || !config.run.tensorEarlyCull ||
        !config.run.mortonOrdering || config.run.ssimMode != 2 ||
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
    settings.run.ssimMode = 3;
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
        !loaded.run.tensorEarlyCull &&
        loaded.run.ssimMode == 0 &&
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
           !loaded.run.tensorRgbLatencyHiding && !loaded.run.tensorEarlyCull;
}

bool testEarlyCullConfig(const std::filesystem::path& path) {
    for (const auto& entry : {std::pair{"tensor_early_cull 0\n", false},
             std::pair{"tensor_early_cull 1\n", true},
             std::pair{"tensor_early_cull 99\n", false},
             std::pair{"tensor_early_cull -1\n", false},
             std::pair{"tensor_early_cull invalid\n", false},
             std::pair{"tensor_early_cull 1\ntensor_early_cull invalid\n", false}}) {
        { std::ofstream output(path, std::ios::trunc); output << entry.first; }
        TrainingSettings loaded{};
        TrainingSettingsStore::load(path, loaded);
        if (loaded.run.tensorEarlyCull != entry.second) return false;
    }
    for (bool enabled : {false, true}) {
        TrainingSettings source{};
        source.run.tensorEarlyCull = enabled;
        TrainingSettingsStore::save(path, source);
        TrainingSettings loaded{};
        TrainingSettingsStore::load(path, loaded);
        if (loaded.run.tensorEarlyCull != enabled) return false;
        std::ifstream saved(path);
        const std::string contents((std::istreambuf_iterator<char>(saved)), {});
        if (contents.find(std::string("tensor_early_cull ") + (enabled ? "1" : "0")) == std::string::npos)
            return false;
    }
    return true;
}

bool testSsimModesAndMigration(const std::filesystem::path& path) {
    for (int mode : {0, 1, 2}) {
        TrainingSettings source = populatedSettings();
        source.run.ssimMode = mode;
        TrainingSettingsStore::save(path, source);
        TrainingSettings loaded{};
        TrainingSettingsStore::load(path, loaded);
        if (loaded.run.ssimMode != mode) return false;
        std::ifstream saved(path);
        const std::string contents((std::istreambuf_iterator<char>(saved)), {});
        if (contents.find("separable_ssim ") != std::string::npos ||
            contents.find("ssim_mode " + std::to_string(mode)) == std::string::npos) return false;
    }
    for (const auto& entry : {
             std::pair{"separable_ssim 0\n", 0},
             std::pair{"separable_ssim 1\n", 1},
             std::pair{"separable_ssim 1\nssim_mode 2\n", 2},
             std::pair{"ssim_mode 2\nseparable_ssim 0\n", 2},
             std::pair{"ssim_mode 99\nseparable_ssim 1\n", 0},
             std::pair{"ssim_mode invalid\nseparable_ssim 1\n", 0},
             std::pair{"separable_ssim 1\nssim_mode invalid\n", 0},
             std::pair{"ssim_mode -1\n", 0}}) {
        { std::ofstream output(path, std::ios::trunc); output << entry.first; }
        TrainingSettings loaded{};
        TrainingSettingsStore::load(path, loaded);
        if (loaded.run.ssimMode != entry.second) return false;
    }
    using Mode = TrainingSsimMode;
    return ssimScratchPlaneCount(Mode::TwoDimensional) == 0u &&
           ssimScratchPlaneCount(Mode::Separable) == 5u &&
           ssimScratchPlaneCount(Mode::BackwardSeparable) == 3u &&
           !ssimUsesSeparableLoss(Mode::BackwardSeparable) &&
           ssimUsesSeparableBackward(Mode::BackwardSeparable) &&
           std::string(trainingSsimModeFileTag(Mode::BackwardSeparable)) == "ssim_backward_separable";
}

bool testMortonConfigAndSchedule(const std::filesystem::path& path) {
    for (const auto& entry :
         {std::pair{"morton_ordering 0\n", false}, std::pair{"morton_ordering 1\n", true},
          std::pair{"morton_ordering invalid\n", false}, std::pair{"morton_ordering 2\n", false}}) {
        {
            std::ofstream output(path);
            output << entry.first;
        }
        TrainingSettings loaded;
        TrainingSettingsStore::load(path, loaded);
        if (loaded.run.mortonOrdering != entry.second) return false;
    }
    TrainingMortonStats stats;
    uint32_t events = 0;
    for (uint32_t completed = 0; completed <= 30000; ++completed) {
        if (trainingMortonDue(false, false, completed, stats) ||
            trainingMortonDue(true, true, completed, stats))
            return false;
        if (trainingMortonDue(true, false, completed, stats)) {
            if (completed != events * 5000 || events == 4) return false;
            ++stats.reorderCount;
            stats.lastCompletedSteps = completed;
            ++events;
            if (trainingMortonDue(true, false, completed, stats)) return false;
        }
    }
    return events == 4;
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
    const bool ssimModes = testSsimModesAndMigration(path);
    const bool earlyCull = testEarlyCullConfig(path);
    const bool morton = testMortonConfigAndSchedule(path);
    std::filesystem::remove(path);
    if (!roundTrip || !validation || !legacyKeys || !legacyGlobal || !clamp || !ssimModes ||
        !earlyCull || !morton) {
        std::cerr << "Training settings tests failed: morton=" << morton
                  << " roundTrip=" << roundTrip << ", validation=" << validation
                  << ", legacyKeys=" << legacyKeys << ", legacyGlobal=" << legacyGlobal
                  << ", clamp=" << clamp << ", ssimModes=" << ssimModes
                  << ", earlyCull=" << earlyCull << '\n';
        return 1;
    }
    return 0;
}
