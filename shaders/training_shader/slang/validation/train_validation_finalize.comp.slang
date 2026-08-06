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
groupshared float validationProcessedCandidateSum[256];
groupshared float validationContributorSum[256];
groupshared uint validationMaxProcessedCandidates[256];
groupshared uint validationNonFiniteGaussianCount[256];
groupshared uint validationNonFinitePositionCount[256];
groupshared uint validationNonFiniteOpacityCount[256];
groupshared uint validationNonFiniteRawScaleCount[256];
groupshared uint validationNonFiniteActivatedScaleCount[256];
groupshared uint validationNonFiniteRotationCount[256];
groupshared uint validationNonFiniteSHCount[256];
groupshared uint validationFirstNonFiniteGaussianIndex[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 groupThreadID : SV_GroupThreadID) {
    uint localIndex = groupThreadID.x;
    uint pixelPartialCount = ((params.width + 15u) / 16u) * ((params.height + 15u) / 16u);
    uint gaussianPartialCount = (params.gaussianCount + 255u) / 256u;

    float lossSum = 0.0;
    float maxLoss = 0.0;
    float alphaSum = 0.0;
    float luminanceSum = 0.0;
    uint invalidLossCount = 0u;
    uint invalidRenderedCount = 0u;
    uint validRenderedCount = 0u;
    float processedCandidateSum = 0.0;
    float contributorSum = 0.0;
    uint maxProcessedCandidates = 0u;
    uint nonFiniteGaussianCount = 0u;
    uint nonFinitePositionCount = 0u;
    uint nonFiniteOpacityCount = 0u;
    uint nonFiniteRawScaleCount = 0u;
    uint nonFiniteActivatedScaleCount = 0u;
    uint nonFiniteRotationCount = 0u;
    uint nonFiniteSHCount = 0u;
    uint firstNonFiniteGaussianIndex = 0xffffffffu;

    for (uint index = localIndex; index < pixelPartialCount; index += 256u) {
        TrainingPixelValidationPartial partial = pixelValidationPartials[index];
        lossSum += partial.lossSum;
        maxLoss = max(maxLoss, partial.maxLoss);
        alphaSum += partial.alphaSum;
        luminanceSum += partial.luminanceSum;
        invalidLossCount += partial.invalidLossCount;
        invalidRenderedCount += partial.invalidRenderedPixelCount;
        validRenderedCount += partial.validRenderedPixelCount;
        processedCandidateSum += partial.processedCandidateSum;
        contributorSum += partial.contributorSum;
        maxProcessedCandidates = max(maxProcessedCandidates, partial.maxProcessedCandidates);
    }
    for (uint index = localIndex; index < gaussianPartialCount; index += 256u) {
        TrainingGaussianValidationPartial partial = gaussianValidationPartials[index];
        nonFiniteGaussianCount += partial.nonFiniteGaussianCount;
        nonFinitePositionCount += partial.nonFinitePositionCount;
        nonFiniteOpacityCount += partial.nonFiniteOpacityCount;
        nonFiniteRawScaleCount += partial.nonFiniteRawScaleCount;
        nonFiniteActivatedScaleCount += partial.nonFiniteActivatedScaleCount;
        nonFiniteRotationCount += partial.nonFiniteRotationCount;
        nonFiniteSHCount += partial.nonFiniteSHCount;
        firstNonFiniteGaussianIndex = min(firstNonFiniteGaussianIndex,
                                          partial.firstNonFiniteGaussianIndex);
    }

    validationLossSum[localIndex] = lossSum;
    validationMaxLoss[localIndex] = maxLoss;
    validationAlphaSum[localIndex] = alphaSum;
    validationLuminanceSum[localIndex] = luminanceSum;
    validationInvalidLossCount[localIndex] = invalidLossCount;
    validationInvalidRenderedCount[localIndex] = invalidRenderedCount;
    validationValidRenderedCount[localIndex] = validRenderedCount;
    validationProcessedCandidateSum[localIndex] = processedCandidateSum;
    validationContributorSum[localIndex] = contributorSum;
    validationMaxProcessedCandidates[localIndex] = maxProcessedCandidates;
    validationNonFiniteGaussianCount[localIndex] = nonFiniteGaussianCount;
    validationNonFinitePositionCount[localIndex] = nonFinitePositionCount;
    validationNonFiniteOpacityCount[localIndex] = nonFiniteOpacityCount;
    validationNonFiniteRawScaleCount[localIndex] = nonFiniteRawScaleCount;
    validationNonFiniteActivatedScaleCount[localIndex] = nonFiniteActivatedScaleCount;
    validationNonFiniteRotationCount[localIndex] = nonFiniteRotationCount;
    validationNonFiniteSHCount[localIndex] = nonFiniteSHCount;
    validationFirstNonFiniteGaussianIndex[localIndex] = firstNonFiniteGaussianIndex;
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
            validationProcessedCandidateSum[localIndex] += validationProcessedCandidateSum[localIndex + stride];
            validationContributorSum[localIndex] += validationContributorSum[localIndex + stride];
            validationMaxProcessedCandidates[localIndex] = max(
                validationMaxProcessedCandidates[localIndex],
                validationMaxProcessedCandidates[localIndex + stride]);
            validationNonFiniteGaussianCount[localIndex] +=
                validationNonFiniteGaussianCount[localIndex + stride];
            validationNonFinitePositionCount[localIndex] +=
                validationNonFinitePositionCount[localIndex + stride];
            validationNonFiniteOpacityCount[localIndex] +=
                validationNonFiniteOpacityCount[localIndex + stride];
            validationNonFiniteRawScaleCount[localIndex] +=
                validationNonFiniteRawScaleCount[localIndex + stride];
            validationNonFiniteActivatedScaleCount[localIndex] +=
                validationNonFiniteActivatedScaleCount[localIndex + stride];
            validationNonFiniteRotationCount[localIndex] +=
                validationNonFiniteRotationCount[localIndex + stride];
            validationNonFiniteSHCount[localIndex] +=
                validationNonFiniteSHCount[localIndex + stride];
            validationFirstNonFiniteGaussianIndex[localIndex] = min(
                validationFirstNonFiniteGaussianIndex[localIndex],
                validationFirstNonFiniteGaussianIndex[localIndex + stride]);
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
        result.nonFinitePositionCount = validationNonFinitePositionCount[0];
        result.nonFiniteOpacityCount = validationNonFiniteOpacityCount[0];
        result.nonFiniteRawScaleCount = validationNonFiniteRawScaleCount[0];
        result.nonFiniteActivatedScaleCount = validationNonFiniteActivatedScaleCount[0];
        result.nonFiniteRotationCount = validationNonFiniteRotationCount[0];
        result.nonFiniteSHCount = validationNonFiniteSHCount[0];
        result.firstNonFiniteGaussianIndex = validationFirstNonFiniteGaussianIndex[0];
        result.processedCandidateSum = validationProcessedCandidateSum[0];
        result.contributorSum = validationContributorSum[0];
        result.maxProcessedCandidates = validationMaxProcessedCandidates[0];
        result.padding0 = 0u;
        validationFinalResult[0] = result;
    }
}
