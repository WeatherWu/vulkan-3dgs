#include "../common/train_backward_common.slang"

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

groupshared uint sharedMaxProcessedCount;
groupshared uint sharedGaussianIndices[256];
groupshared float4 sharedCenterRadius[256];
groupshared float4 sharedConicOpacity[256];
groupshared float4 sharedColor[256];

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID,
          uint3 groupThreadID : SV_GroupThreadID) {
    uint2 pixel = dispatchThreadID.xy;
    uint localIndex = groupThreadID.y * TRAIN_TILE_SIZE + groupThreadID.x;
    bool validPixel = pixel.x < params.width && pixel.y < params.height;

    TileRange range = unpackTileRange(tileRanges[tileIndexForPixel(pixel)]);
    uint pixelLinearIndex = validPixel ? pixelIndex(pixel) : 0u;
    PixelBlendState blendState;
    blendState.finalTransmittance = 1.0;
    blendState.accumulatedAlpha = 0.0;
    blendState.processedCount = 0u;
    blendState.contributionCount = 0u;
    float3 dLoss_dPixelColor = float3(0.0);
    if (validPixel) {
        blendState = pixelBlendStates[pixelLinearIndex];
        dLoss_dPixelColor = pixelGrads[pixelLinearIndex].color.xyz;
    }

    uint processedCount = validPixel ? min(blendState.processedCount, range.count) : 0u;
    if (localIndex == 0u) {
        sharedMaxProcessedCount = 0u;
    }
    GroupMemoryBarrierWithGroupSync();
    uint ignoredMax = 0u;
    InterlockedMax(sharedMaxProcessedCount, processedCount, ignoredMax);
    GroupMemoryBarrierWithGroupSync();

    float2 samplePosition = pixelCenter(pixel);
    float3 suffixColor = float3(0.0);
    float transmittanceBefore = blendState.finalTransmittance;
    uint remaining = sharedMaxProcessedCount;

    while (remaining > 0u) {
        uint batchCount = min(remaining, 256u);
        uint batchStart = remaining - batchCount;
        if (localIndex < batchCount) {
            uint itemOffset = batchStart + localIndex;
            uint gaussianIndex = tileItems[range.start + itemOffset];
            sharedGaussianIndices[localIndex] = gaussianIndex;
            if (gaussianIndex < params.gaussianCount) {
                ProjectedGaussian gaussian = projectedGaussians[gaussianIndex];
                sharedCenterRadius[localIndex] = gaussian.centerRadius;
                sharedConicOpacity[localIndex] = gaussian.conicOpacity;
                sharedColor[localIndex] = gaussian.color;
            }
        }
        GroupMemoryBarrierWithGroupSync();

        if (validPixel) {
            for (uint reverseIndex = 0u; reverseIndex < batchCount; ++reverseIndex) {
                uint sharedIndex = batchCount - 1u - reverseIndex;
                uint itemOffset = batchStart + sharedIndex;
                if (itemOffset >= processedCount) {
                    continue;
                }

                uint gaussianIndex = sharedGaussianIndices[sharedIndex];
                if (gaussianIndex >= params.gaussianCount) {
                    continue;
                }

                ProjectedGaussian gaussian;
                gaussian.centerRadius = sharedCenterRadius[sharedIndex];
                gaussian.conicOpacity = sharedConicOpacity[sharedIndex];
                gaussian.color = sharedColor[sharedIndex];
                GaussianSample sample = evaluateGaussianSample(gaussian, samplePosition);
                if (!contributesToPixel(sample)) {
                    continue;
                }

                float oneMinusAlpha = max(1.0 - sample.alpha, 1e-6);
                transmittanceBefore /= oneMinusAlpha;

                float3 dLoss_dGaussianColor =
                    dLoss_dPixelColor * (transmittanceBefore * sample.alpha);
                float dLoss_dAlpha = dot(
                    dLoss_dPixelColor,
                    transmittanceBefore * (sample.color - suffixColor));
                ProjectedGaussianGrad grad = backwardGaussianSample(
                    gaussian, sample, dLoss_dAlpha, dLoss_dGaussianColor);
                atomicAddProjectedGaussianGrad(projectedGrads, gaussianIndex, grad);

                suffixColor = sample.alpha * sample.color + oneMinusAlpha * suffixColor;
            }
        }
        GroupMemoryBarrierWithGroupSync();
        remaining = batchStart;
    }
}
