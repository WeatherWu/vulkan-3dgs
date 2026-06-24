#pragma once

#include <cstdint>
#include <glm/glm.hpp>

namespace vk_gs {

struct TrainingExtent {
    uint32_t width = 0;
    uint32_t height = 0;
};

struct TrainingPushConstants {
    uint32_t gaussianCount = 0;
    uint32_t pixelCount = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    float optimizerLearningRate = 1e-3f;
    float optimizerBeta1 = 0.9f;
    float optimizerBeta2 = 0.999f;
    float optimizerEpsilon = 1e-8f;
    float optimizerGradClip = 1e3f;
    float padding0 = 0.0f;
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

struct TrainingOptimizerConfig {
    float learningRate = 1e-3f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;
    float gradClip = 1e3f;
};

struct TrainingDensificationConfig {
    bool enabled = true;
    uint32_t densifyFromIteration = 500;
    uint32_t densifyUntilIteration = 15000;
    uint32_t densificationInterval = 100;
    uint32_t opacityResetInterval = 3000;
    uint32_t maxGaussianCount = 1000000;
    uint32_t splitChildren = 2;
    float densifyGradThreshold = 0.0002f;
    float minOpacity = 0.005f;
    float percentDense = 0.01f;
    float screenSizePruneThreshold = 20.0f;
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
static_assert(sizeof(TrainingDensificationPushConstants) == sizeof(glm::vec4) * 3);

} // namespace vk_gs
