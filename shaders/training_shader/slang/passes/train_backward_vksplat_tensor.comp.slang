#include "../common/train_backward_common.slang"

static const uint VKSPLAT_TENSOR_BATCH_SIZE = 16u;
static const uint VKSPLAT_TENSOR_REDUCTION_WIDTH = 16u;
static const uint VKSPLAT_TILE_PIXEL_COUNT = TRAIN_TILE_SIZE * TRAIN_TILE_SIZE;
static const uint VKSPLAT_TENSOR_GRAD_COMPONENT_COUNT = 9u;

[[vk::binding(3, 0)]]
StructuredBuffer<ProjectedGaussian> projectedGaussians;

[[vk::binding(4, 0)]]
StructuredBuffer<uint> tileItems;

[[vk::binding(5, 0)]]
StructuredBuffer<uint4> tileRanges;

[[vk::binding(12, 0)]]
StructuredBuffer<PixelGrad> pixelGrads;

[[vk::binding(13, 0)]]
RWStructuredBuffer<ProjectedGaussianGrad> projectedGrads;

[[vk::binding(14, 0)]]
StructuredBuffer<PixelBlendState> pixelBlendStates;

groupshared ProjectedGaussian sharedGaussians[VKSPLAT_TENSOR_BATCH_SIZE];
groupshared float2 sharedPairDerivatives[
    VKSPLAT_TENSOR_BATCH_SIZE * VKSPLAT_TILE_PIXEL_COUNT];
groupshared float sharedThreadGradComponents[
    VKSPLAT_TENSOR_GRAD_COMPONENT_COUNT * VKSPLAT_TILE_PIXEL_COUNT];
groupshared float sharedPixelGradRed[VKSPLAT_TILE_PIXEL_COUNT];
groupshared float sharedPixelGradGreen[VKSPLAT_TILE_PIXEL_COUNT];
groupshared float sharedPixelGradBlue[VKSPLAT_TILE_PIXEL_COUNT];

struct CompactProjectedGrad {
    float2 center;
    float4 conicOpacity;
    float3 color;
};

CompactProjectedGrad makeZeroCompactProjectedGrad() {
    CompactProjectedGrad grad;
    grad.center = float2(0.0);
    grad.conicOpacity = float4(0.0);
    grad.color = float3(0.0);
    return grad;
}

void addCompactProjectedGrad(inout CompactProjectedGrad destination,
                             CompactProjectedGrad source) {
    destination.center += source.center;
    destination.conicOpacity += source.conicOpacity;
    destination.color += source.color;
}

bool compactProjectedGradIsNonZero(CompactProjectedGrad grad) {
    return any(grad.center != float2(0.0)) ||
           any(grad.conicOpacity != float4(0.0)) ||
           any(grad.color != float3(0.0));
}

void storeSharedThreadGrad(uint threadIndex, CompactProjectedGrad grad) {
    sharedThreadGradComponents[0u * VKSPLAT_TILE_PIXEL_COUNT + threadIndex] =
        grad.center.x;
    sharedThreadGradComponents[1u * VKSPLAT_TILE_PIXEL_COUNT + threadIndex] =
        grad.center.y;
    sharedThreadGradComponents[2u * VKSPLAT_TILE_PIXEL_COUNT + threadIndex] =
        grad.conicOpacity.x;
    sharedThreadGradComponents[3u * VKSPLAT_TILE_PIXEL_COUNT + threadIndex] =
        grad.conicOpacity.y;
    sharedThreadGradComponents[4u * VKSPLAT_TILE_PIXEL_COUNT + threadIndex] =
        grad.conicOpacity.z;
    sharedThreadGradComponents[5u * VKSPLAT_TILE_PIXEL_COUNT + threadIndex] =
        grad.conicOpacity.w;
    sharedThreadGradComponents[6u * VKSPLAT_TILE_PIXEL_COUNT + threadIndex] =
        grad.color.x;
    sharedThreadGradComponents[7u * VKSPLAT_TILE_PIXEL_COUNT + threadIndex] =
        grad.color.y;
    sharedThreadGradComponents[8u * VKSPLAT_TILE_PIXEL_COUNT + threadIndex] =
        grad.color.z;
}

CompactProjectedGrad loadSharedThreadGrad(uint threadIndex) {
    CompactProjectedGrad grad = makeZeroCompactProjectedGrad();
    grad.center.x = sharedThreadGradComponents[
        0u * VKSPLAT_TILE_PIXEL_COUNT + threadIndex];
    grad.center.y = sharedThreadGradComponents[
        1u * VKSPLAT_TILE_PIXEL_COUNT + threadIndex];
    grad.conicOpacity.x = sharedThreadGradComponents[
        2u * VKSPLAT_TILE_PIXEL_COUNT + threadIndex];
    grad.conicOpacity.y = sharedThreadGradComponents[
        3u * VKSPLAT_TILE_PIXEL_COUNT + threadIndex];
    grad.conicOpacity.z = sharedThreadGradComponents[
        4u * VKSPLAT_TILE_PIXEL_COUNT + threadIndex];
    grad.conicOpacity.w = sharedThreadGradComponents[
        5u * VKSPLAT_TILE_PIXEL_COUNT + threadIndex];
    grad.color.x = sharedThreadGradComponents[
        6u * VKSPLAT_TILE_PIXEL_COUNT + threadIndex];
    grad.color.y = sharedThreadGradComponents[
        7u * VKSPLAT_TILE_PIXEL_COUNT + threadIndex];
    grad.color.z = sharedThreadGradComponents[
        8u * VKSPLAT_TILE_PIXEL_COUNT + threadIndex];
    return grad;
}

CompactProjectedGrad backwardTensorPair(ProjectedGaussian gaussian,
                                        float2 samplePosition,
                                        float dLoss_dPower,
                                        float3 dLoss_dColor) {
    CompactProjectedGrad grad = makeZeroCompactProjectedGrad();
    grad.color = dLoss_dColor;
    float2 delta = samplePosition - gaussian.centerRadius.xy;
    float a = gaussian.conicOpacity.x;
    float b = gaussian.conicOpacity.y;
    float c = gaussian.conicOpacity.z;
    grad.conicOpacity.x = dLoss_dPower * (-0.5 * delta.x * delta.x);
    grad.conicOpacity.y = dLoss_dPower * (-0.5 * delta.x * delta.y);
    grad.conicOpacity.z = dLoss_dPower * (-0.5 * delta.y * delta.y);
    grad.conicOpacity.w = dLoss_dPower / max(gaussian.conicOpacity.w, 1e-20);
    grad.center.x = dLoss_dPower * (a * delta.x + b * delta.y);
    grad.center.y = dLoss_dPower * (b * delta.x + c * delta.y);
    return grad;
}

void atomicAddCompactProjectedGrad(uint gaussianIndex,
                                   CompactProjectedGrad grad) {
    atomicAddFloat(projectedGrads[gaussianIndex].centerRadius.x, grad.center.x);
    atomicAddFloat(projectedGrads[gaussianIndex].centerRadius.y, grad.center.y);
    atomicAddFloat(projectedGrads[gaussianIndex].conicOpacity.x, grad.conicOpacity.x);
    atomicAddFloat(projectedGrads[gaussianIndex].conicOpacity.y, grad.conicOpacity.y);
    atomicAddFloat(projectedGrads[gaussianIndex].conicOpacity.z, grad.conicOpacity.z);
    atomicAddFloat(projectedGrads[gaussianIndex].conicOpacity.w, grad.conicOpacity.w);
    atomicAddFloat(projectedGrads[gaussianIndex].color.x, grad.color.x);
    atomicAddFloat(projectedGrads[gaussianIndex].color.y, grad.color.y);
    atomicAddFloat(projectedGrads[gaussianIndex].color.z, grad.color.z);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 groupThreadID : SV_GroupThreadID,
          uint3 groupID : SV_GroupID) {
    uint localIndex = groupThreadID.y * TRAIN_TILE_SIZE + groupThreadID.x;
    uint tileIndex = groupID.y * tileCountX() + groupID.x;
    TileRange range = unpackTileRange(tileRanges[tileIndex]);
    uint safeRangeStart = min(range.start, params.tileItemCount);
    uint rangeCount = min(range.count, params.tileItemCount - safeRangeStart);
    uint2 tileOrigin = groupID.xy * TRAIN_TILE_SIZE;
    uint2 localPixel = uint2(
        localIndex % TRAIN_TILE_SIZE,
        localIndex / TRAIN_TILE_SIZE);
    uint2 pixel = tileOrigin + localPixel;
    bool validPixel = pixel.x < params.width && pixel.y < params.height;

    PixelBlendState blendState;
    blendState.finalTransmittance = 1.0;
    blendState.accumulatedAlpha = 0.0;
    blendState.processedCount = 0u;
    blendState.contributionCount = 0u;
    float3 dLoss_dPixelColor = float3(0.0);
    if (validPixel) {
        uint pixelLinearIndex = pixelIndex(pixel);
        blendState = pixelBlendStates[pixelLinearIndex];
        dLoss_dPixelColor = pixelGrads[pixelLinearIndex].color.xyz;
    }
    uint processedCount = validPixel
        ? min(blendState.processedCount, rangeCount)
        : 0u;
    sharedPixelGradRed[localIndex] = dLoss_dPixelColor.x;
    sharedPixelGradGreen[localIndex] = dLoss_dPixelColor.y;
    sharedPixelGradBlue[localIndex] = dLoss_dPixelColor.z;
    float3 suffixColor = float3(0.0);
    float transmittanceBefore = blendState.finalTransmittance;
    uint remaining = rangeCount;
    while (remaining > 0u) {
        uint batchCount = min(remaining, VKSPLAT_TENSOR_BATCH_SIZE);
        uint batchStart = remaining - batchCount;
        if (localIndex < batchCount) {
            uint gaussianIndex = tileItems[safeRangeStart + batchStart + localIndex];
            if (gaussianIndex < params.gaussianCount) {
                sharedGaussians[localIndex] = projectedGaussians[gaussianIndex];
            } else {
                ProjectedGaussian zeroGaussian;
                zeroGaussian.centerRadius = float4(0.0);
                zeroGaussian.conicOpacity = float4(0.0);
                zeroGaussian.color = float4(0.0);
                sharedGaussians[localIndex] = zeroGaussian;
            }
        }
        GroupMemoryBarrierWithGroupSync();

        for (uint reverseIndex = 0u; reverseIndex < batchCount; ++reverseIndex) {
            uint gaussianLocalIndex = batchCount - 1u - reverseIndex;
            uint itemOffset = batchStart + gaussianLocalIndex;
            float2 pairDerivative = float2(0.0);
            if (validPixel && itemOffset < processedCount) {
                ProjectedGaussian gaussian = sharedGaussians[gaussianLocalIndex];
                GaussianSample sample = evaluateGaussianSample(
                    gaussian, pixelCenter(pixel));
                if (contributesToPixel(sample)) {
                    float oneMinusAlpha = max(1.0 - sample.alpha, 1e-6);
                    transmittanceBefore /= oneMinusAlpha;
                    float alphaTransmittance = transmittanceBefore * sample.alpha;
                    float dLoss_dAlpha = dot(
                        dLoss_dPixelColor,
                        transmittanceBefore * (sample.color - suffixColor));
                    float dLoss_dPower = dLoss_dAlpha *
                        gaussian.conicOpacity.w * sample.coverage;
                    pairDerivative = float2(alphaTransmittance, dLoss_dPower);
                    suffixColor = sample.alpha * sample.color +
                                  oneMinusAlpha * suffixColor;
                }
            }
            sharedPairDerivatives[
                gaussianLocalIndex * VKSPLAT_TILE_PIXEL_COUNT + localIndex] =
                    pairDerivative;
        }
        GroupMemoryBarrierWithGroupSync();

        uint gaussianLocalIndex = localIndex / VKSPLAT_TENSOR_REDUCTION_WIDTH;
        uint reductionLane = localIndex % VKSPLAT_TENSOR_REDUCTION_WIDTH;
        CompactProjectedGrad localGrad = makeZeroCompactProjectedGrad();
        if (gaussianLocalIndex < batchCount) {
            ProjectedGaussian gaussian = sharedGaussians[gaussianLocalIndex];
            for (uint pixelRank = reductionLane;
                 pixelRank < VKSPLAT_TILE_PIXEL_COUNT;
                 pixelRank += VKSPLAT_TENSOR_REDUCTION_WIDTH) {
                float2 pairDerivative = sharedPairDerivatives[
                    gaussianLocalIndex * VKSPLAT_TILE_PIXEL_COUNT + pixelRank];
                if (pairDerivative.x != 0.0 || pairDerivative.y != 0.0) {
                    uint2 pairLocalPixel = uint2(
                        pixelRank % TRAIN_TILE_SIZE,
                        pixelRank / TRAIN_TILE_SIZE);
                    uint2 pairPixel = tileOrigin + pairLocalPixel;
                    float3 pairPixelGrad = float3(
                        sharedPixelGradRed[pixelRank],
                        sharedPixelGradGreen[pixelRank],
                        sharedPixelGradBlue[pixelRank]);
                    float3 dLoss_dGaussianColor =
                        pairPixelGrad * pairDerivative.x;
                    CompactProjectedGrad grad = backwardTensorPair(
                        gaussian,
                        pixelCenter(pairPixel),
                        pairDerivative.y,
                        dLoss_dGaussianColor);
                    addCompactProjectedGrad(localGrad, grad);
                }
            }
        }
        CompactProjectedGrad tileGrad = makeZeroCompactProjectedGrad();
        uint outputGaussianLocalIndex = 0u;
        bool outputTileGrad = false;
        storeSharedThreadGrad(localIndex, localGrad);
        GroupMemoryBarrierWithGroupSync();
        if (localIndex < batchCount) {
            uint base = localIndex * VKSPLAT_TENSOR_REDUCTION_WIDTH;
            for (uint lane = 0u; lane < VKSPLAT_TENSOR_REDUCTION_WIDTH; ++lane) {
                addCompactProjectedGrad(tileGrad, loadSharedThreadGrad(base + lane));
            }
            outputGaussianLocalIndex = localIndex;
            outputTileGrad = true;
        }
        if (outputTileGrad) {
            uint gaussianIndex = tileItems[
                safeRangeStart + batchStart + outputGaussianLocalIndex];
            if (gaussianIndex < params.gaussianCount &&
                compactProjectedGradIsNonZero(tileGrad)) {
                atomicAddCompactProjectedGrad(gaussianIndex, tileGrad);
            }
        }
        GroupMemoryBarrierWithGroupSync();
        remaining = batchStart;
    }
}
