#include "training/app/training_settings.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

using namespace vulkan3DGS;
namespace {
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void writeConfig(const std::filesystem::path& path, const char* text) {
    std::ofstream output(path);
    output << text;
    if (!output) throw std::runtime_error("Cannot write settings fixture");
}
void testSettings(const std::filesystem::path& path) {
    TrainingSettings settings{};
    check(!settings.run.tensorSubgroupReduction && !settings.run.tensorGlobalPixelGrad &&
              settings.run.ssimMode == 0,
          "Default controls changed");
    setTextBuffer(settings.paths.dataset, "dataset with spaces");
    setTextBuffer(settings.paths.outputDirectory, "output dir");
    setTextBuffer(settings.paths.outputName, "model.ply");
    settings.gpuSelector = "GPU-uuid";
    settings.initialization.downscale = 8;
    settings.run.pixelTo2DGSMode = 6;
    settings.run.tileMode = 0;
    settings.run.tensorSubgroupReduction = true;
    settings.run.tensorGlobalPixelGrad = true;
    settings.run.ssimMode = 2;
    settings.run.pureTraining = true;
    settings.optimizer.positionLearningRate = 0.012345f;
    settings.densification.maxGaussians = 7654321;
    settings.densification.splitChildren = 7;
    settings.densification.gradientMode = 1;
    check(TrainingSettingsStore::validate(settings).valid, "Valid settings rejected");
    TrainingSettingsStore::save(path, settings);
    TrainingSettings loaded{};
    TrainingSettingsStore::load(path, loaded);
    check(textBufferString(loaded.paths.dataset) == "dataset with spaces" &&
              loaded.gpuSelector == "GPU-uuid" && loaded.initialization.downscale == 8 &&
              loaded.run.pixelTo2DGSMode == 6 && loaded.run.tileMode == 0 &&
              loaded.run.tensorSubgroupReduction && loaded.run.tensorGlobalPixelGrad &&
              loaded.run.ssimMode == 2 && loaded.run.pureTraining &&
              loaded.optimizer.positionLearningRate == settings.optimizer.positionLearningRate &&
              loaded.densification.maxGaussians == 7654321 &&
              loaded.densification.splitChildren == 7 && loaded.densification.gradientMode == 1,
          "Settings round trip");
    const auto run = makeTrainingRunConfig(settings);
    settings.run.ssimMode = 0;
    setTextBuffer(settings.paths.dataset, "changed");
    check(run.run.ssimMode == 2 && run.datasetPath == "dataset with spaces" &&
              run.outputPath == std::filesystem::path("output dir") / "model.ply",
          "Immutable run configuration changed");
    settings.run.ssimMode = 1;
    check(!TrainingSettingsStore::validate(settings).valid, "Unsupported SSIM mode accepted");
    settings.run.ssimMode = 0;
    settings.run.tileMode = 2;
    check(!TrainingSettingsStore::validate(settings).valid, "Unsupported tile mode accepted");
    settings.run.tileMode = 1;
    settings.optimizer.adamBeta1 = 1.0f;
    check(!TrainingSettingsStore::validate(settings).valid, "Invalid optimizer accepted");

    for (const auto text : {"ssim_mode 2\nseparable_ssim 1\n", "separable_ssim 1\nssim_mode 2\n"}) {
        writeConfig(path, text);
        loaded = {};
        TrainingSettingsStore::load(path, loaded);
        check(loaded.run.ssimMode == 2, "Explicit SSIM mode precedence");
    }
    writeConfig(path, "ssim_mode 1\ntraining_tile_mode 2\n"
                      "tensor_rgb_latency_hiding 1\nadaptive_tensor_child_batch 32\n");
    loaded = {};
    TrainingSettingsStore::load(path, loaded);
    check(loaded.run.ssimMode == 0 && loaded.run.tileMode == 1,
          "Unsupported selections must not enable a different method");
    TrainingSettingsStore::save(path, loaded);
    std::ifstream saved(path);
    const std::string serialized((std::istreambuf_iterator<char>(saved)), {});
    check(serialized.find("adaptive_") == std::string::npos &&
              serialized.find("tensor_rgb_latency_hiding") == std::string::npos &&
              serialized.find("separable_ssim") == std::string::npos,
          "Unsupported settings persisted");

    writeConfig(path, "downscale 0\npixel_to_2dgs_mode 99\ntraining_tile_mode 99\n"
                      "split_children 99\ndensification_gradient_mode 99\n");
    loaded = {};
    TrainingSettingsStore::load(path, loaded);
    check(loaded.initialization.downscale == 1 && loaded.run.pixelTo2DGSMode == 6 &&
              loaded.run.tileMode == 1 && loaded.densification.splitChildren == 8 &&
              loaded.densification.gradientMode == 1,
          "Settings normalization");
}
} // namespace
int main() {
    const auto path =
        std::filesystem::temp_directory_path() / "vulkan-3dgs-training-settings-test.cfg";
    try {
        testSettings(path);
    } catch (const std::exception& error) {
        std::filesystem::remove(path);
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::filesystem::remove(path);
    std::cout << "Training settings contracts passed\n";
}
