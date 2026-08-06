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
static const int SSIM_SHARED_WIDTH = 26;
static const int SSIM_SHARED_PIXEL_COUNT = SSIM_SHARED_WIDTH * SSIM_SHARED_WIDTH;
static const float SSIM_C1 = 0.0001;
static const float SSIM_C2 = 0.0009;
static const float SSIM_INV_WEIGHT_SUM = 0.0707622378;
static const float SSIM_GAUSSIAN_WEIGHTS[6] = {
    1.0,
    0.8007374029,
    0.4111122905,
    0.1353352832,
    0.0285655008,
    0.0038659201
};

groupshared float4 sharedRenderedColor[SSIM_SHARED_PIXEL_COUNT];
groupshared uint sharedTargetColor[SSIM_SHARED_PIXEL_COUNT];

float ssimWindowWeight(int2 offset) {
    return SSIM_GAUSSIAN_WEIGHTS[abs(offset.x)] * SSIM_GAUSSIAN_WEIGHTS[abs(offset.y)];
}

uint ssimSharedIndex(int2 pixel) {
    return uint(pixel.y * SSIM_SHARED_WIDTH + pixel.x);
}

void loadSsimNeighborhood(uint2 groupOrigin, uint localIndex) {
    for (uint sharedIndex = localIndex;
         sharedIndex < SSIM_SHARED_PIXEL_COUNT;
         sharedIndex += 256u) {
        int2 sharedPixel = int2(sharedIndex % SSIM_SHARED_WIDTH,
                                sharedIndex / SSIM_SHARED_WIDTH);
        int2 imagePixel = int2(groupOrigin) + sharedPixel - int2(SSIM_RADIUS);
        if (imagePixel.x >= 0 && imagePixel.y >= 0 &&
            imagePixel.x < int(params.width) && imagePixel.y < int(params.height)) {
            uint imageIndex = uint(imagePixel.y * int(params.width) + imagePixel.x);
            sharedRenderedColor[sharedIndex] = renderedColor[imageIndex];
            sharedTargetColor[sharedIndex] = targetColor[imageIndex];
        } else {
            sharedRenderedColor[sharedIndex] = float4(0.0);
            sharedTargetColor[sharedIndex] = 0u;
        }
    }
    GroupMemoryBarrierWithGroupSync();
}

float ssimColor(uint2 localPixel, out SsimBackwardState backwardState) {
    float3 renderedMean = float3(0.0);
    float3 targetMean = float3(0.0);
    float3 renderedSqMean = float3(0.0);
    float3 targetSqMean = float3(0.0);
    float3 renderedTargetMean = float3(0.0);

    for (int dy = -SSIM_RADIUS; dy <= SSIM_RADIUS; ++dy) {
        for (int dx = -SSIM_RADIUS; dx <= SSIM_RADIUS; ++dx) {
            float weight = ssimWindowWeight(int2(dx, dy));
            int2 samplePixel = int2(localPixel) + int2(dx + SSIM_RADIUS, dy + SSIM_RADIUS);
            float3 rendered = sharedRenderedColor[ssimSharedIndex(samplePixel)].xyz;
            float3 target = unpackTargetColor(sharedTargetColor[ssimSharedIndex(samplePixel)]).xyz;

            renderedMean += rendered * weight;
            targetMean += target * weight;
            renderedSqMean += rendered * rendered * weight;
            targetSqMean += target * target * weight;
            renderedTargetMean += rendered * target * weight;
        }
    }

    renderedMean *= SSIM_INV_WEIGHT_SUM;
    targetMean *= SSIM_INV_WEIGHT_SUM;
    renderedSqMean *= SSIM_INV_WEIGHT_SUM;
    targetSqMean *= SSIM_INV_WEIGHT_SUM;
    renderedTargetMean *= SSIM_INV_WEIGHT_SUM;

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
[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID,
          uint3 groupThreadID : SV_GroupThreadID,
          uint3 groupID : SV_GroupID) {
    uint2 localPixel = groupThreadID.xy;
    uint2 pixel = dispatchThreadID.xy;
    uint localIndex = groupThreadID.y * 16u + groupThreadID.x;
    bool active = pixel.x < params.width && pixel.y < params.height;
    uint index = active ? pixel.y * params.width + pixel.x : 0u;
    uint2 groupOrigin = groupID.xy * 16u;
    loadSsimNeighborhood(groupOrigin, localIndex);
    float pixelLoss = 0.0;
    float4 rendered = float4(0.0);
    uint processedCandidateCount = 0u;
    uint contributorCount = 0u;

    if (active) {
        SsimBackwardState backwardState;
        rendered = renderedColor[index];
        float l1 = l1Color(rendered.xyz, unpackTargetColor(targetColor[index]).xyz);
        float dssim = 1.0 - ssimColor(localPixel, backwardState);
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
        uint validationGroupCountX = (params.width + 15u) / 16u;
        pixelValidationPartials[groupID.x + groupID.y * validationGroupCountX] = partial;
    }
}
