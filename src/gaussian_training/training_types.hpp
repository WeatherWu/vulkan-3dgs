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

struct alignas(16) GaussianGrad {
    glm::vec4 positionOpacity{};
    glm::vec4 scale{};
    glm::vec4 rotation{};
    glm::vec4 sh[16]{};
};

struct alignas(16) AdamState {
    GaussianGrad firstMoment{};
    GaussianGrad secondMoment{};
};

static_assert(sizeof(GaussianTrainParam) == sizeof(glm::vec4) * 19);
static_assert(sizeof(ProjectedGaussian) == sizeof(glm::vec4) * 3);
static_assert(sizeof(GaussianGrad) == sizeof(glm::vec4) * 19);
static_assert(sizeof(AdamState) == sizeof(GaussianGrad) * 2);

} // namespace vk_gs
