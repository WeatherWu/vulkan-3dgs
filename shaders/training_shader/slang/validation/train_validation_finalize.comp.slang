#include "../common/train_common.slang"

[[vk::binding(29, 0)]]
StructuredBuffer<TrainingPixelValidationPartial> pixelValidationPartials;

[[vk::binding(30, 0)]]
StructuredBuffer<TrainingGaussianValidationPartial> gaussianValidationPartials;

[[vk::binding(31, 0)]]
RWStructuredBuffer<TrainingValidationGpuResult> validationFinalResult;

groupshared float validationLossSum[256];
groupshared float validationMaxLoss[256];
groupshared float validationAlphaSum[256];
groupshared float validationLuminanceSum[256];
groupshared uint validationInvalidLossCount[256];
groupshared uint validationInvalidRenderedCount[256];
groupshared uint validationValidRenderedCount[256];
groupshared uint validationNonFiniteGaussianCount[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 groupThreadID : SV_GroupThreadID) {
    uint localIndex = groupThreadID.x;
    uint pixelPartialCount = (params.pixelCount + 255u) / 256u;
    uint gaussianPartialCount = (params.gaussianCount + 255u) / 256u;

    float lossSum = 0.0;
    float maxLoss = 0.0;
    float alphaSum = 0.0;
    float luminanceSum = 0.0;
    uint invalidLossCount = 0u;
    uint invalidRenderedCount = 0u;
    uint validRenderedCount = 0u;
    uint nonFiniteGaussianCount = 0u;

    for (uint index = localIndex; index < pixelPartialCount; index += 256u) {
        TrainingPixelValidationPartial partial = pixelValidationPartials[index];
        lossSum += partial.lossSum;
        maxLoss = max(maxLoss, partial.maxLoss);
        alphaSum += partial.alphaSum;
        luminanceSum += partial.luminanceSum;
        invalidLossCount += partial.invalidLossCount;
        invalidRenderedCount += partial.invalidRenderedPixelCount;
        validRenderedCount += partial.validRenderedPixelCount;
    }
    for (uint index = localIndex; index < gaussianPartialCount; index += 256u) {
        nonFiniteGaussianCount += gaussianValidationPartials[index].nonFiniteGaussianCount;
    }

    validationLossSum[localIndex] = lossSum;
    validationMaxLoss[localIndex] = maxLoss;
    validationAlphaSum[localIndex] = alphaSum;
    validationLuminanceSum[localIndex] = luminanceSum;
    validationInvalidLossCount[localIndex] = invalidLossCount;
    validationInvalidRenderedCount[localIndex] = invalidRenderedCount;
    validationValidRenderedCount[localIndex] = validRenderedCount;
    validationNonFiniteGaussianCount[localIndex] = nonFiniteGaussianCount;
    GroupMemoryBarrierWithGroupSync();

    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (localIndex < stride) {
            validationLossSum[localIndex] += validationLossSum[localIndex + stride];
            validationMaxLoss[localIndex] = max(validationMaxLoss[localIndex],
                                                validationMaxLoss[localIndex + stride]);
            validationAlphaSum[localIndex] += validationAlphaSum[localIndex + stride];
            validationLuminanceSum[localIndex] += validationLuminanceSum[localIndex + stride];
            validationInvalidLossCount[localIndex] += validationInvalidLossCount[localIndex + stride];
            validationInvalidRenderedCount[localIndex] += validationInvalidRenderedCount[localIndex + stride];
            validationValidRenderedCount[localIndex] += validationValidRenderedCount[localIndex + stride];
            validationNonFiniteGaussianCount[localIndex] +=
                validationNonFiniteGaussianCount[localIndex + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (localIndex == 0u) {
        TrainingValidationGpuResult result;
        result.lossSum = validationLossSum[0];
        result.maxLoss = validationMaxLoss[0];
        result.alphaSum = validationAlphaSum[0];
        result.luminanceSum = validationLuminanceSum[0];
        result.invalidLossCount = validationInvalidLossCount[0];
        result.invalidRenderedPixelCount = validationInvalidRenderedCount[0];
        result.validRenderedPixelCount = validationValidRenderedCount[0];
        result.nonFiniteGaussianCount = validationNonFiniteGaussianCount[0];
        result.validationIteration = params.validationIteration;
        result.padding0 = 0u;
        result.padding1 = 0u;
        result.padding2 = 0u;
        result.padding3 = 0u;
        result.padding4 = 0u;
        result.padding5 = 0u;
        result.padding6 = 0u;
        validationFinalResult[0] = result;
    }
}
