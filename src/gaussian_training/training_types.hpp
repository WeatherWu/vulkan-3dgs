#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <glm/glm.hpp>

namespace vulkan3DGS {

struct TrainingExtent {
    uint32_t width = 0;
    uint32_t height = 0;
};

struct TrainingPushConstants {
    uint32_t gaussianCount = 0;
    uint32_t pixelCount = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t trainingIteration = 0;
    uint32_t activeSHDegree = 0;
    uint32_t maxSHDegree = 3;
    uint32_t tileItemCount = 0;
    float positionLearningRate = 0.00016f;
    float positionLearningRateFinal = 0.0000016f;
    float positionLearningRateDelayMult = 0.01f;
    float positionLearningRateMaxSteps = 30000.0f;
    float featureLearningRate = 0.0025f;
    float featureRestLearningRate = 0.000125f;
    float opacityLearningRate = 0.025f;
    float scaleLearningRate = 0.005f;
    float rotationLearningRate = 0.001f;
    float optimizerBeta1 = 0.9f;
    float optimizerBeta2 = 0.999f;
    float optimizerEpsilon = 1e-8f;
    float optimizerGradClip = 1e3f;
    float lossDssimWeight = 0.2f;
    float padding1 = 0.0f;
    float padding2 = 0.0f;
};

struct TrainingInitializationConfig {
    bool allowRandomFallback = true;
    uint32_t randomGaussianCount = 10000;
    uint32_t randomSeed = 1;
    float initialOpacity = 0.1f;
    float sceneRadiusScale = 1.0f;
};

enum class TrainingImageSelectionMode : uint32_t {
    Sequential = 0,
    Random = 1,
};

struct TrainingScheduleConfig {
    TrainingImageSelectionMode imageSelectionMode = TrainingImageSelectionMode::Sequential;
    uint32_t totalIterations = 0;
    uint32_t randomSeed = 1;
};

struct TrainingOptimizerConfig {
    float positionLearningRate = 0.00016f;
    float positionLearningRateFinal = 0.0000016f;
    float positionLearningRateDelayMult = 0.01f;
    float positionLearningRateMaxSteps = 30000.0f;
    float featureLearningRate = 0.0025f;
    float featureRestLearningRate = 0.000125f;
    float opacityLearningRate = 0.025f;
    float scaleLearningRate = 0.005f;
    float rotationLearningRate = 0.001f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;
    float gradClip = 1e3f;
    float lossDssimWeight = 0.2f;
    uint32_t maxSHDegree = 3;
    uint32_t shDegreeInterval = 1000;
};

struct TrainingValidationStats {
    float meanLoss = 0.0f;
    float maxLoss = 0.0f;
    float meanRenderedAlpha = 0.0f;
    float meanRenderedLuminance = 0.0f;
    uint32_t invalidLossCount = 0;
    uint32_t invalidRenderedPixelCount = 0;
    uint32_t nonFiniteGaussianCount = 0;
    uint32_t tileItemCount = 0;
    bool renderedNonEmpty = false;
    bool valid = true;
};

enum class TrainingCpuProfileStage : uint32_t {
    FrameUpload = 0,
    ImageRequest,
    TargetUpload,
    PrepareSubmit,
    TileCountReadback,
    TileBufferResize,
    MainSubmit,
    Validation,
    DensificationAdopt,
    Total,
    Count,
};

enum class TrainingGpuProfileStage : uint32_t {
    PrepareTileItems = 0,
    TileEmit,
    TileSortAndRanges,
    Composite,
    Loss,
    LossToPixel,
    PixelTo2DGS,
    TwoDGSTo3DGS,
    Optimizer,
    Densification,
    Count,
};

constexpr size_t kTrainingCpuProfileStageCount = static_cast<size_t>(TrainingCpuProfileStage::Count);
constexpr size_t kTrainingGpuProfileStageCount = static_cast<size_t>(TrainingGpuProfileStage::Count);
constexpr uint32_t kTrainingGpuTimestampQueryCount =
    static_cast<uint32_t>(kTrainingGpuProfileStageCount * 2u);

constexpr uint32_t trainingGpuTimestampQuery(TrainingGpuProfileStage stage, bool end) {
    return static_cast<uint32_t>(stage) * 2u + (end ? 1u : 0u);
}

struct TrainingTiming {
    float lastMs = 0.0f;
    float averageMs = 0.0f;
    uint32_t sampleCount = 0;
};

struct TrainingProfilingStats {
    bool gpuTimestampsAvailable = false;
    std::array<TrainingTiming, kTrainingCpuProfileStageCount> cpu{};
    std::array<TrainingTiming, kTrainingGpuProfileStageCount> gpu{};
};

struct TrainingDensificationStats {
    uint32_t outputCount = 0;
    uint32_t pruneOpacityHits = 0;
    uint32_t pruneScreenHits = 0;
    uint32_t pruneWorldHits = 0;
    uint32_t keptSources = 0;
    uint32_t cloneSources = 0;
    uint32_t splitSources = 0;
    uint32_t prunedSources = 0;
};

struct TrainingDensificationConfig {
    bool enabled = true;
    uint32_t densifyFromIteration = 500;
    uint32_t densifyUntilIteration = 15000;
    uint32_t densificationInterval = 100;
    uint32_t opacityResetInterval = 3000;
    uint32_t maxGaussianCount = 10000000;
    uint32_t splitChildren = 2;
    float densifyGradThreshold = 0.0002f;
    float minOpacity = 0.005f;
    float percentDense = 0.01f;
    float screenSizePruneThreshold = 20.0f;
    float worldSizePruneThreshold = 0.1f;
};

struct alignas(16) TrainingDensificationPushConstants {
    uint32_t gaussianCount = 0;
    uint32_t maxGaussianCount = 0;
    uint32_t splitChildren = 2;
    uint32_t pruneByScreenSize = 0;
    float sceneExtent = 1.0f;
    float gradThreshold = 0.0002f;
    float minOpacity = 0.005f;
    float percentDense = 0.01f;
    float screenSizePruneThreshold = 20.0f;
    uint32_t randomSeed = 1;
    uint32_t resetOpacity = 0;
    float worldSizePruneThreshold = 0.1f;
    uint32_t padding1 = 0;
};

struct alignas(16) GaussianTrainParam {
    glm::vec4 positionOpacity{};
    glm::vec4 scale{};
    glm::vec4 rotation{};
    glm::vec4 sh[16]{};
};

struct alignas(16) ProjectedGaussian {
    glm::vec4 centerRadius{};
    glm::vec4 conicOpacity{};
    glm::vec4 color{};
};

struct alignas(16) PixelGrad {
    glm::vec4 color{};
};

struct alignas(16) PixelBlendState {
    float finalTransmittance = 1.0f;
    float accumulatedAlpha = 0.0f;
    uint32_t processedCount = 0;
    uint32_t lastContributor = UINT32_MAX;
};

struct alignas(16) GaussianVisibilityState {
    uint32_t contributionCount = 0;
    float alphaSum = 0.0f;
    float maxScreenRadius = 0.0f;
    uint32_t padding0 = 0;
};

struct alignas(16) GaussianDensificationState {
    float screenGradSum = 0.0f;
    uint32_t screenGradCount = 0;
    float maxScreenRadius = 0.0f;
    uint32_t padding0 = 0;
};

struct alignas(16) ProjectedGaussianGrad {
    glm::vec4 centerRadius{};
    glm::vec4 conicOpacity{};
    glm::vec4 color{};
};

struct alignas(16) GaussianGrad {
    glm::vec4 positionOpacity{};
    glm::vec4 scale{};
    glm::vec4 rotation{};
    glm::vec4 sh[16]{};
};

struct alignas(16) TrainingForwardCamera {
    glm::mat4 view{1.0f};
    glm::mat4 projection{1.0f};
    glm::mat4 model{1.0f};
    glm::vec4 viewport{};
    glm::vec4 focalTan{};
    glm::vec4 cameraPosition{};
};

struct alignas(16) AdamState {
    GaussianGrad firstMoment{};
    GaussianGrad secondMoment{};
};

static_assert(sizeof(GaussianTrainParam) == sizeof(glm::vec4) * 19);
static_assert(sizeof(ProjectedGaussian) == sizeof(glm::vec4) * 3);
static_assert(sizeof(PixelGrad) == sizeof(glm::vec4));
static_assert(sizeof(PixelBlendState) == sizeof(glm::vec4));
static_assert(sizeof(GaussianVisibilityState) == sizeof(glm::vec4));
static_assert(sizeof(GaussianDensificationState) == sizeof(glm::vec4));
static_assert(sizeof(ProjectedGaussianGrad) == sizeof(glm::vec4) * 3);
static_assert(sizeof(GaussianGrad) == sizeof(glm::vec4) * 19);
static_assert(sizeof(TrainingForwardCamera) == sizeof(glm::vec4) * 15);
static_assert(sizeof(AdamState) == sizeof(GaussianGrad) * 2);
static_assert(sizeof(TrainingDensificationPushConstants) == sizeof(glm::vec4) * 4);
static_assert(sizeof(TrainingPushConstants) == sizeof(glm::vec4) * 6);

} // namespace vulkan3DGS
