#include "../common/train_common.slang"
#include "../common/train_backward_types.slang"
[[vk::binding(6, 0)]] StructuredBuffer<float4> renderedColor;
[[vk::binding(8, 0)]] StructuredBuffer<float> lossBuffer;
[[vk::binding(14, 0)]] StructuredBuffer<PixelBlendState> pixelBlendStates;
[[vk::binding(29, 0)]] RWStructuredBuffer<TrainingPixelValidationPartial> pixelValidationPartials;
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
groupshared uint4 validationProcessedCandidateBucketsLow[256];
groupshared uint4 validationProcessedCandidateBucketsHigh[256];

uint trainProcessedCandidateBucket(uint processedCandidateCount) {
    if (processedCandidateCount == 0u) {
        return 0u;
    }
    if (processedCandidateCount <= 32u) {
        return 1u;
    }
    if (processedCandidateCount <= 64u) {
        return 2u;
    }
    if (processedCandidateCount <= 128u) {
        return 3u;
    }
    if (processedCandidateCount <= 256u) {
        return 4u;
    }
    if (processedCandidateCount <= 512u) {
        return 5u;
    }
    if (processedCandidateCount <= 1024u) {
        return 6u;
    }
    return 7u;
}

[shader("compute")]
[numthreads(16,16,1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID,
          uint3 groupThreadID : SV_GroupThreadID, uint3 groupID : SV_GroupID) {
    uint2 pixel = dispatchThreadID.xy;
    bool active = pixel.x < params.width && pixel.y < params.height;
    uint index = active ? pixel.y * params.width + pixel.x : 0u;
    float pixelLoss = active ? lossBuffer[index] : 0.0;
    float4 rendered = active ? renderedColor[index] : float4(0.0);
    uint processedCandidateCount = active ? pixelBlendStates[index].processedCount : 0u;
    uint contributorCount = active ? pixelBlendStates[index].contributionCount : 0u;
    bool finiteLoss = active && isfinite(pixelLoss);
    bool finiteRendered = active && all(isfinite(rendered));
    uint reductionIndex = groupThreadID.y * 16u + groupThreadID.x;
    validationLossSum[reductionIndex] = finiteLoss ? pixelLoss : 0.0;
    validationMaxLoss[reductionIndex] = finiteLoss ? pixelLoss : 0.0;
    validationAlphaSum[reductionIndex] = finiteRendered ? rendered.w : 0.0;
    validationLuminanceSum[reductionIndex] = finiteRendered
        ? (rendered.x + rendered.y + rendered.z) / 3.0
        : 0.0;
    validationInvalidLossCount[reductionIndex] = active && !finiteLoss ? 1u : 0u;
    validationInvalidRenderedCount[reductionIndex] = active && !finiteRendered ? 1u : 0u;
    validationValidRenderedCount[reductionIndex] = finiteRendered ? 1u : 0u;
    validationProcessedCandidateSum[reductionIndex] = float(processedCandidateCount);
    validationContributorSum[reductionIndex] = float(contributorCount);
    validationMaxProcessedCandidates[reductionIndex] = processedCandidateCount;
    uint4 processedCandidateBucketsLow = uint4(0u);
    uint4 processedCandidateBucketsHigh = uint4(0u);
    uint processedCandidateBucket = trainProcessedCandidateBucket(processedCandidateCount);
    if (processedCandidateBucket < 4u) {
        processedCandidateBucketsLow[processedCandidateBucket] = active ? 1u : 0u;
    } else {
        processedCandidateBucketsHigh[processedCandidateBucket - 4u] = active ? 1u : 0u;
    }
    validationProcessedCandidateBucketsLow[reductionIndex] = processedCandidateBucketsLow;
    validationProcessedCandidateBucketsHigh[reductionIndex] = processedCandidateBucketsHigh;
    GroupMemoryBarrierWithGroupSync();

    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (reductionIndex < stride) {
            validationLossSum[reductionIndex] += validationLossSum[reductionIndex + stride];
            validationMaxLoss[reductionIndex] = max(validationMaxLoss[reductionIndex],
                                                    validationMaxLoss[reductionIndex + stride]);
            validationAlphaSum[reductionIndex] += validationAlphaSum[reductionIndex + stride];
            validationLuminanceSum[reductionIndex] += validationLuminanceSum[reductionIndex + stride];
            validationInvalidLossCount[reductionIndex] += validationInvalidLossCount[reductionIndex + stride];
            validationInvalidRenderedCount[reductionIndex] += validationInvalidRenderedCount[reductionIndex + stride];
            validationValidRenderedCount[reductionIndex] += validationValidRenderedCount[reductionIndex + stride];
            validationProcessedCandidateSum[reductionIndex] += validationProcessedCandidateSum[reductionIndex + stride];
            validationContributorSum[reductionIndex] += validationContributorSum[reductionIndex + stride];
            validationMaxProcessedCandidates[reductionIndex] = max(
                validationMaxProcessedCandidates[reductionIndex],
                validationMaxProcessedCandidates[reductionIndex + stride]);
            validationProcessedCandidateBucketsLow[reductionIndex] +=
                validationProcessedCandidateBucketsLow[reductionIndex + stride];
            validationProcessedCandidateBucketsHigh[reductionIndex] +=
                validationProcessedCandidateBucketsHigh[reductionIndex + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (reductionIndex == 0u) {
        TrainingPixelValidationPartial partial;
        partial.lossSum = validationLossSum[0];
        partial.maxLoss = validationMaxLoss[0];
        partial.alphaSum = validationAlphaSum[0];
        partial.luminanceSum = validationLuminanceSum[0];
        partial.invalidLossCount = validationInvalidLossCount[0];
        partial.invalidRenderedPixelCount = validationInvalidRenderedCount[0];
        partial.validRenderedPixelCount = validationValidRenderedCount[0];
        partial.padding0 = 0u;
        partial.processedCandidateSum = validationProcessedCandidateSum[0];
        partial.contributorSum = validationContributorSum[0];
        partial.maxProcessedCandidates = validationMaxProcessedCandidates[0];
        partial.padding1 = 0u;
        partial.processedCandidateBucketsLow = validationProcessedCandidateBucketsLow[0];
        partial.processedCandidateBucketsHigh = validationProcessedCandidateBucketsHigh[0];
        uint validationGroupCountX = (params.width + 15u) / 16u;
        pixelValidationPartials[groupID.x + groupID.y * validationGroupCountX] = partial;
    }
}
