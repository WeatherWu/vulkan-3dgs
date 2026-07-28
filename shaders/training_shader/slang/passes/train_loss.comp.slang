#include "../common/train_common.slang"
#include "../common/train_backward_types.slang"
#include "../common/train_loss_common.slang"
#include "../common/train_target_common.slang"

[[vk::binding(6, 0)]]
StructuredBuffer<float4> renderedColor;

[[vk::binding(7, 0)]]
StructuredBuffer<uint> targetColor;

[[vk::binding(8, 0)]]
RWStructuredBuffer<float> lossBuffer;

[[vk::binding(14, 0)]]
StructuredBuffer<PixelBlendState> pixelBlendStates;

[[vk::binding(28, 0)]]
RWStructuredBuffer<SsimBackwardState> ssimBackwardStates;

[[vk::binding(29, 0)]]
RWStructuredBuffer<TrainingPixelValidationPartial> pixelValidationPartials;

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

static const int SSIM_RADIUS = 5;
static const float SSIM_C1 = 0.0001;
static const float SSIM_C2 = 0.0009;
static const float SSIM_SIGMA = 1.5;

float ssimWindowWeight(int2 offset) {
    float2 value = float2(offset);
    return exp(-dot(value, value) / (2.0 * SSIM_SIGMA * SSIM_SIGMA));
}

bool pixelInBounds(int2 pixel) {
    return pixel.x >= 0 && pixel.y >= 0 &&
           pixel.x < int(params.width) && pixel.y < int(params.height);
}

uint pixelIndex(int2 pixel) {
    return uint(pixel.y * int(params.width) + pixel.x);
}

float ssimColor(uint centerIndex, out SsimBackwardState backwardState) {
    int width = int(params.width);
    int2 center = int2(int(centerIndex % params.width), int(centerIndex / params.width));

    float3 renderedMean = float3(0.0);
    float3 targetMean = float3(0.0);
    float3 renderedSqMean = float3(0.0);
    float3 targetSqMean = float3(0.0);
    float3 renderedTargetMean = float3(0.0);
    float weightSum = 0.0;

    for (int dy = -SSIM_RADIUS; dy <= SSIM_RADIUS; ++dy) {
        for (int dx = -SSIM_RADIUS; dx <= SSIM_RADIUS; ++dx) {
            float weight = ssimWindowWeight(int2(dx, dy));
            int2 samplePixel = center + int2(dx, dy);
            float3 rendered = float3(0.0);
            float3 target = float3(0.0);
            if (pixelInBounds(samplePixel)) {
                uint sampleIndex = pixelIndex(samplePixel);
                rendered = renderedColor[sampleIndex].xyz;
                target = unpackTargetColor(targetColor[sampleIndex]).xyz;
            }

            renderedMean += rendered * weight;
            targetMean += target * weight;
            renderedSqMean += rendered * rendered * weight;
            targetSqMean += target * target * weight;
            renderedTargetMean += rendered * target * weight;
            weightSum += weight;
        }
    }

    renderedMean /= weightSum;
    targetMean /= weightSum;
    renderedSqMean /= weightSum;
    targetSqMean /= weightSum;
    renderedTargetMean /= weightSum;

    float3 renderedVariance = renderedSqMean - renderedMean * renderedMean;
    float3 targetVariance = targetSqMean - targetMean * targetMean;
    float3 covariance = renderedTargetMean - renderedMean * targetMean;

    float3 A = 2.0 * renderedMean * targetMean + SSIM_C1;
    float3 B = 2.0 * covariance + SSIM_C2;
    float3 C = renderedMean * renderedMean + targetMean * targetMean + SSIM_C1;
    float3 D = renderedVariance + targetVariance + SSIM_C2;
    float3 numerator = A * B;
    float3 denominator = C * D;
    float3 denominatorSquared = denominator * denominator;
    float3 lossScale = float3(-1.0 / 3.0);
    backwardState.base = float4(lossScale *
        (2.0 * targetMean * (B - A) * denominator -
         2.0 * numerator * renderedMean * (D - C)) /
        denominatorSquared, 0.0);
    backwardState.targetCoefficient = float4(
        lossScale * (2.0 * A * denominator) / denominatorSquared, 0.0);
    backwardState.renderedCoefficient = float4(
        lossScale * (-2.0 * numerator * C) / denominatorSquared, 0.0);

    float3 ssim = numerator / denominator;
    return (ssim.x + ssim.y + ssim.z) / 3.0;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID,
          uint3 groupThreadID : SV_GroupThreadID,
          uint3 groupID : SV_GroupID) {
    uint index = dispatchThreadID.x;
    bool active = index < params.pixelCount;
    float pixelLoss = 0.0;
    float4 rendered = float4(0.0);
    uint processedCandidateCount = 0u;
    uint contributorCount = 0u;

    if (active) {
        SsimBackwardState backwardState;
        rendered = renderedColor[index];
        float l1 = l1Color(rendered.xyz, unpackTargetColor(targetColor[index]).xyz);
        float dssim = 1.0 - ssimColor(index, backwardState);
        ssimBackwardStates[index] = backwardState;
        float dssimWeight = clamp(params.lossDssimWeight, 0.0, 1.0);
        pixelLoss = ((1.0 - dssimWeight) * l1 + dssimWeight * dssim) /
                    max(float(params.pixelCount), 1.0);
        lossBuffer[index] = pixelLoss;
        PixelBlendState blendState = pixelBlendStates[index];
        processedCandidateCount = blendState.processedCount;
        contributorCount = blendState.contributionCount;
    }

    if (params.validationEnabled == 0u) {
        return;
    }

    bool finiteLoss = active && isfinite(pixelLoss);
    bool finiteRendered = active && all(isfinite(rendered));
    uint localIndex = groupThreadID.x;
    validationLossSum[localIndex] = finiteLoss ? pixelLoss : 0.0;
    validationMaxLoss[localIndex] = finiteLoss ? pixelLoss : 0.0;
    validationAlphaSum[localIndex] = finiteRendered ? rendered.w : 0.0;
    validationLuminanceSum[localIndex] = finiteRendered
        ? (rendered.x + rendered.y + rendered.z) / 3.0
        : 0.0;
    validationInvalidLossCount[localIndex] = active && !finiteLoss ? 1u : 0u;
    validationInvalidRenderedCount[localIndex] = active && !finiteRendered ? 1u : 0u;
    validationValidRenderedCount[localIndex] = finiteRendered ? 1u : 0u;
    validationProcessedCandidateSum[localIndex] = float(processedCandidateCount);
    validationContributorSum[localIndex] = float(contributorCount);
    validationMaxProcessedCandidates[localIndex] = processedCandidateCount;
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
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (localIndex == 0u) {
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
        pixelValidationPartials[groupID.x] = partial;
    }
}
