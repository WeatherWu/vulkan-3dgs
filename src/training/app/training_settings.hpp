#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <string>

namespace vulkan3DGS {

struct TrainingPathSettings {
    std::array<char, 512> dataset{};
    std::array<char, 512> outputDirectory{};
    std::array<char, 256> outputName{};
};

struct TrainingInitializationSettings {
    int downscale = 4;
    uint32_t gaussianCount = 10000;
    uint32_t randomSeed = 0;
    float opacity = 0.1f;
    float sceneRadiusScale = 1.0f;
};

struct TrainingScheduleSettings {
    int imageSelectionMode = 1;
    uint32_t totalIterations = 30000;
    uint32_t validationInterval = 100;
};

struct TrainingOptimizerSettings {
    float positionLearningRate = 0.00016f;
    float positionLearningRateFinal = 0.0000016f;
    float positionLearningRateDelayMult = 0.01f;
    float positionLearningRateDelaySteps = 0.0f;
    float positionLearningRateMaxSteps = 30000.0f;
    float featureLearningRate = 0.0025f;
    float featureRestLearningRate = 0.000125f;
    float opacityLearningRate = 0.025f;
    float scaleLearningRate = 0.005f;
    float rotationLearningRate = 0.001f;
    float adamBeta1 = 0.9f;
    float adamBeta2 = 0.999f;
    float adamEpsilon = 1e-15f;
    float gradientClip = 0.0f;
    float lossDssimWeight = 0.2f;
    uint32_t maxSHDegree = 3;
    uint32_t shDegreeInterval = 1000;
};

struct TrainingDensificationSettings {
    bool enabled = true;
    uint32_t fromIteration = 500;
    uint32_t untilIteration = 15000;
    uint32_t interval = 100;
    uint32_t opacityResetInterval = 3000;
    uint32_t maxGaussians = 10000000;
    uint32_t splitChildren = 2;
    float gradientThreshold = 0.0002f;
    float minOpacity = 0.005f;
    float percentDense = 0.01f;
    float screenPruneSize = 20.0f;
    float worldPruneSize = 0.1f;
};

struct TrainingRunSettings {
    int pixelTo2DGSMode = 0;
    float pixelTo2DGSMinSubgroupUtilization = 0.5f;
    int forwardCompositeMode = 1;
    uint32_t benchmarkStepsPerFrame = 1;
    uint32_t benchmarkFrame = 0;
    uint32_t benchmarkWarmupSteps = 200;
    uint32_t benchmarkMeasuredSteps = 1000;
    bool pureTraining = false;
};

// Passive, copyable and persistable configuration. UI buffers remain inside
// the path group; Vulkan, worker and live GaussianTraining state do not.
struct TrainingSettings {
    TrainingPathSettings paths{};
    std::string gpuSelector;
    TrainingInitializationSettings initialization{};
    TrainingScheduleSettings schedule{};
    TrainingOptimizerSettings optimizer{};
    TrainingDensificationSettings densification{};
    TrainingRunSettings run{};
};

struct TrainingSettingsValidation {
    bool valid = true;
    std::string message;

    explicit operator bool() const noexcept { return valid; }
};

// Immutable-by-ownership snapshot created when a run starts. TrainingController
// stores it through a pointer-to-const so worker policy and Pure export cannot
// observe later UI edits.
struct TrainingRunConfig {
    std::filesystem::path datasetPath;
    std::filesystem::path outputPath;
    std::string gpuSelector;
    TrainingInitializationSettings initialization{};
    TrainingScheduleSettings schedule{};
    TrainingOptimizerSettings optimizer{};
    TrainingDensificationSettings densification{};
    TrainingRunSettings run{};
};

[[nodiscard]] TrainingRunConfig makeTrainingRunConfig(
    const TrainingSettings& settings);

template <size_t N>
void setTextBuffer(std::array<char, N>& buffer, const std::string& value) {
    buffer.fill('\0');
    value.copy(buffer.data(), N - 1u);
}

template <size_t N>
std::string textBufferString(const std::array<char, N>& buffer) {
    return std::string(buffer.data());
}

class TrainingSettingsStore {
public:
    static std::filesystem::path defaultPath();
    static void load(const std::filesystem::path& path, TrainingSettings& settings);
    static void save(const std::filesystem::path& path, const TrainingSettings& settings);
    [[nodiscard]] static TrainingSettingsValidation validate(
        const TrainingSettings& settings);

private:
    static void write(std::ostream& output, const TrainingSettings& settings);
    static void clamp(TrainingSettings& settings);
};

} // namespace vulkan3DGS
