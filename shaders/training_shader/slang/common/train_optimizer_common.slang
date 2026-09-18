GaussianGrad makeZeroGaussianGrad() {
    GaussianGrad grad;
    grad.positionOpacity = float4(0.0);
    grad.scale = float4(0.0);
    grad.rotation = float4(0.0);
    for (uint index = 0u; index < 12u; ++index) {
        grad.sh[index] = float4(0.0);
    }
    return grad;
}

float4 clipGrad(float4 grad) {
    if (params.optimizerGradClip <= 0.0) {
        return grad;
    }

    float gradClip = max(params.optimizerGradClip, 1e-15);
    return clamp(grad, float4(-gradClip), float4(gradClip));
}

float4 adamUpdateMasked(inout float4 firstMoment,
                        inout float4 secondMoment,
                        float4 grad,
                        float4 learningRate,
                        float4 activeMask) {
    grad = clipGrad(grad);
    float beta1 = clamp(params.optimizerBeta1, 0.0, 0.999999);
    float beta2 = clamp(params.optimizerBeta2, 0.0, 0.999999);
    float4 updatedFirstMoment =
        beta1 * firstMoment + params.optimizerOneMinusBeta1 * grad;
    float4 updatedSecondMoment =
        beta2 * secondMoment + params.optimizerOneMinusBeta2 * grad * grad;
    firstMoment = lerp(firstMoment, updatedFirstMoment, activeMask);
    secondMoment = lerp(secondMoment, updatedSecondMoment, activeMask);

    float4 correctedFirstMoment =
        firstMoment * params.optimizerInvFirstMomentCorrection;
    float4 correctedSecondMoment =
        secondMoment * params.optimizerInvSecondMomentCorrection;
    return activeMask * max(learningRate, float4(0.0)) * correctedFirstMoment /
           (sqrt(correctedSecondMoment) +
            float4(max(params.optimizerEpsilon, 1e-15)));
}

float4 adamUpdate(inout float4 firstMoment,
                  inout float4 secondMoment,
                  float4 grad,
                  float learningRate) {
    return adamUpdateMasked(
        firstMoment,
        secondMoment,
        grad,
        float4(learningRate),
        float4(1.0));
}

void updateGaussianParam(inout GaussianTrainParam gaussian,
                         inout AdamState adamState,
                         GaussianGrad grad) {
    float4 positionOpacityLearningRate = float4(
        max(params.positionLearningRate, 0.0),
        max(params.positionLearningRate, 0.0),
        max(params.positionLearningRate, 0.0),
        max(params.opacityLearningRate, 0.0));
    gaussian.positionOpacity -= adamUpdate(
        adamState.firstMoment.positionOpacity,
        adamState.secondMoment.positionOpacity,
        grad.positionOpacity,
        1.0) * positionOpacityLearningRate;
    gaussian.scale -= adamUpdate(
        adamState.firstMoment.scale,
        adamState.secondMoment.scale,
        grad.scale,
        max(params.scaleLearningRate, 0.0));
    gaussian.rotation -= adamUpdate(
        adamState.firstMoment.rotation,
        adamState.secondMoment.rotation,
        grad.rotation,
        max(params.rotationLearningRate, 0.0));

    uint activeCoefficientCount = params.activeSHDegree < 1u
        ? 1u
        : (params.activeSHDegree < 2u
            ? 4u
            : (params.activeSHDegree < 3u ? 9u : 16u));
    for (uint packedIndex = 0u; packedIndex < 12u; ++packedIndex) {
        uint coefficientBase = (packedIndex % 4u) * 4u;
        float4 activeMask = float4(
            coefficientBase + 0u < activeCoefficientCount ? 1.0 : 0.0,
            coefficientBase + 1u < activeCoefficientCount ? 1.0 : 0.0,
            coefficientBase + 2u < activeCoefficientCount ? 1.0 : 0.0,
            coefficientBase + 3u < activeCoefficientCount ? 1.0 : 0.0);
        float4 learningRate = float4(
            coefficientBase == 0u
                ? max(params.featureLearningRate, 0.0)
                : max(params.featureRestLearningRate, 0.0),
            max(params.featureRestLearningRate, 0.0),
            max(params.featureRestLearningRate, 0.0),
            max(params.featureRestLearningRate, 0.0));
        gaussian.sh[packedIndex] -= adamUpdateMasked(
            adamState.firstMoment.sh[packedIndex],
            adamState.secondMoment.sh[packedIndex],
            grad.sh[packedIndex],
            learningRate,
            activeMask);
    }
}
