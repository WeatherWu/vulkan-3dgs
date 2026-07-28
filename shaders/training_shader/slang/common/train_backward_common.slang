#include "train_forward_tile_common.slang"
#include "train_backward_types.slang"

float l1Derivative(float rendered, float target) {
    if (rendered > target) {
        return 1.0;
    }
    if (rendered < target) {
        return -1.0;
    }
    return 0.0;
}

void atomicAddFloat(inout float destination, float value) {
    float ignored = 0.0;
    InterlockedAdd(destination, value, ignored);
}

void atomicAddProjectedGaussianGrad(RWStructuredBuffer<ProjectedGaussianGrad> projectedGrads,
                                    uint gaussianIndex,
                                    ProjectedGaussianGrad grad) {
    atomicAddFloat(projectedGrads[gaussianIndex].centerRadius.x, grad.centerRadius.x);
    atomicAddFloat(projectedGrads[gaussianIndex].centerRadius.y, grad.centerRadius.y);
    atomicAddFloat(projectedGrads[gaussianIndex].centerRadius.z, grad.centerRadius.z);
    atomicAddFloat(projectedGrads[gaussianIndex].centerRadius.w, grad.centerRadius.w);

    atomicAddFloat(projectedGrads[gaussianIndex].conicOpacity.x, grad.conicOpacity.x);
    atomicAddFloat(projectedGrads[gaussianIndex].conicOpacity.y, grad.conicOpacity.y);
    atomicAddFloat(projectedGrads[gaussianIndex].conicOpacity.z, grad.conicOpacity.z);
    atomicAddFloat(projectedGrads[gaussianIndex].conicOpacity.w, grad.conicOpacity.w);

    atomicAddFloat(projectedGrads[gaussianIndex].color.x, grad.color.x);
    atomicAddFloat(projectedGrads[gaussianIndex].color.y, grad.color.y);
    atomicAddFloat(projectedGrads[gaussianIndex].color.z, grad.color.z);
    atomicAddFloat(projectedGrads[gaussianIndex].color.w, grad.color.w);
}

bool alphaIsDifferentiable(GaussianSample sample) {
    return sample.alpha >= TRAIN_ALPHA_MIN;
}

ProjectedGaussianGrad backwardGaussianSample(ProjectedGaussian gaussian,
                                             GaussianSample sample,
                                             float dLoss_dAlpha,
                                             float3 dLoss_dColor) {
    ProjectedGaussianGrad grad = makeZeroProjectedGaussianGrad();
    grad.color.xyz = dLoss_dColor;

    if (!alphaIsDifferentiable(sample)) {
        return grad;
    }

    float opacity = gaussian.conicOpacity.w;
    float dLoss_dCoverage = dLoss_dAlpha * opacity;
    float dLoss_dOpacity = dLoss_dAlpha * sample.coverage;
    float dLoss_dPower = dLoss_dCoverage * sample.coverage;

    float dx = sample.delta.x;
    float dy = sample.delta.y;
    float a = gaussian.conicOpacity.x;
    float b = gaussian.conicOpacity.y;
    float c = gaussian.conicOpacity.z;

    grad.conicOpacity.x += dLoss_dPower * (-0.5 * dx * dx);
    grad.conicOpacity.y += dLoss_dPower * (-0.5 * dx * dy);
    grad.conicOpacity.z += dLoss_dPower * (-0.5 * dy * dy);
    grad.conicOpacity.w += dLoss_dOpacity;

    grad.centerRadius.x += dLoss_dPower * (a * dx + b * dy);
    grad.centerRadius.y += dLoss_dPower * (b * dx + c * dy);
    return grad;
}
