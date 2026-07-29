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

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint2 pixel = dispatchThreadID.xy;
    bool validPixel = pixel.x < params.width && pixel.y < params.height;
    uint pixelLinearIndex = validPixel ? pixelIndex(pixel) : 0u;
    TileRange range = unpackTileRange(tileRanges[tileIndexForPixel(pixel)]);

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
    uint laneCount = WaveGetLaneCount();
    uint laneIndex = WaveGetLaneIndex();
    uint subgroupMaxProcessedCount = processedCount;
    for (uint laneOffset = 1u; laneOffset < laneCount; laneOffset <<= 1u) {
        uint sourceLane = laneIndex + laneOffset;
        uint sourceValue = WaveReadLaneAt(
            subgroupMaxProcessedCount, min(sourceLane, laneCount - 1u));
        if ((laneIndex % (laneOffset << 1u)) == 0u && sourceLane < laneCount) {
            subgroupMaxProcessedCount = max(subgroupMaxProcessedCount, sourceValue);
        }
    }
    subgroupMaxProcessedCount = WaveReadLaneAt(subgroupMaxProcessedCount, 0u);

    float2 samplePosition = pixelCenter(pixel);
    float3 suffixColor = float3(0.0);
    float transmittanceBefore = blendState.finalTransmittance;
    uint remaining = subgroupMaxProcessedCount;

    while (remaining > 0u) {
        uint batchCount = min(remaining, laneCount);
        uint batchStart = remaining - batchCount;
        uint loadedGaussianIndex = 0xffffffffu;
        float4 loadedCenterRadius = float4(0.0);
        float4 loadedConicOpacity = float4(0.0);
        float4 loadedColor = float4(0.0);
        if (laneIndex < batchCount) {
            uint itemOffset = batchStart + laneIndex;
            loadedGaussianIndex = tileItems[range.start + itemOffset];
            if (loadedGaussianIndex < params.gaussianCount) {
                ProjectedGaussian gaussian = projectedGaussians[loadedGaussianIndex];
                loadedCenterRadius = gaussian.centerRadius;
                loadedConicOpacity = gaussian.conicOpacity;
                loadedColor = gaussian.color;
            }
        }

        for (uint reverseIndex = 0u; reverseIndex < batchCount; ++reverseIndex) {
            uint sourceLane = batchCount - 1u - reverseIndex;
            uint itemOffset = batchStart + sourceLane;
            uint gaussianIndex = WaveReadLaneAt(loadedGaussianIndex, sourceLane);
            float4 centerRadius = WaveReadLaneAt(loadedCenterRadius, sourceLane);
            float4 conicOpacity = WaveReadLaneAt(loadedConicOpacity, sourceLane);
            float4 color = WaveReadLaneAt(loadedColor, sourceLane);
            if (!validPixel || itemOffset >= processedCount ||
                gaussianIndex >= params.gaussianCount) {
                continue;
            }

            ProjectedGaussian gaussian;
            gaussian.centerRadius = centerRadius;
            gaussian.conicOpacity = conicOpacity;
            gaussian.color = color;
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
        remaining = batchStart;
    }
}
