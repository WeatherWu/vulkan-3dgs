#pragma once
#include "training/core/training_types.hpp"
#include <algorithm>
#include <cmath>

namespace optimizer_reference {
// Independent scalar CPU Adam oracle: deliberately no GPU layout/loader helpers.
inline void component(float& parameter, float& m, float& v, float gradient, float rate, bool active,
                      const vulkan3DGS::TrainingPushConstants& p) {
    if (!active) return;
    if (p.optimizerGradClip > 0) {
        const double bound = std::max(double(p.optimizerGradClip), 1e-15);
        gradient = float(std::clamp(double(gradient), -bound, bound));
    }
    const double beta1 = std::clamp(double(p.optimizerBeta1), 0.0, 0.999999);
    const double beta2 = std::clamp(double(p.optimizerBeta2), 0.0, 0.999999);
    const double first = beta1 * m + double(p.optimizerOneMinusBeta1) * gradient;
    const double second = beta2 * v + double(p.optimizerOneMinusBeta2) * gradient * gradient;
    m = float(first);
    v = float(second);
    const double delta = std::max(double(rate), 0.0) * first * p.optimizerInvFirstMomentCorrection /
                         (std::sqrt(second * p.optimizerInvSecondMomentCorrection) +
                          std::max(double(p.optimizerEpsilon), 1e-15));
    parameter = float(double(parameter) - delta);
}
inline void adam(vulkan3DGS::GaussianTrainParam& g, vulkan3DGS::AdamState& state,
                 const vulkan3DGS::GaussianGrad& gradient,
                 const vulkan3DGS::TrainingPushConstants& p) {
    for (int lane = 0; lane < 4; ++lane) {
        component(g.positionOpacity[lane], state.firstMoment.positionOpacity[lane],
                  state.secondMoment.positionOpacity[lane], gradient.positionOpacity[lane],
                  lane == 3 ? p.opacityLearningRate : p.positionLearningRate, true, p);
        component(g.scale[lane], state.firstMoment.scale[lane], state.secondMoment.scale[lane],
                  gradient.scale[lane], p.scaleLearningRate, true, p);
        component(g.rotation[lane], state.firstMoment.rotation[lane],
                  state.secondMoment.rotation[lane], gradient.rotation[lane],
                  p.rotationLearningRate, true, p);
    }
    const uint32_t active = (p.activeSHDegree + 1) * (p.activeSHDegree + 1);
    for (uint32_t channel = 0; channel < 3; ++channel) {
        for (uint32_t coefficient = 0; coefficient < 16; ++coefficient) {
            const uint32_t pack = channel * 4 + coefficient / 4;
            const uint32_t lane = coefficient % 4;
            component(g.sh[pack][lane], state.firstMoment.sh[pack][lane],
                      state.secondMoment.sh[pack][lane], gradient.sh[pack][lane],
                      coefficient == 0 ? p.featureLearningRate : p.featureRestLearningRate,
                      coefficient < active, p);
        }
    }
}
} // namespace optimizer_reference
