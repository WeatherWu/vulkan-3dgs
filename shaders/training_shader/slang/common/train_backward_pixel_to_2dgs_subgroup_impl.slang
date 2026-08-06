#include "train_backward_common.slang"

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

uint reducePixelTo2DGSUint(uint value, uint laneIndex, uint laneCount) {
    for (uint laneOffset = 1u; laneOffset < laneCount; laneOffset <<= 1u) {
        uint sourceLane = laneIndex + laneOffset;
        uint safeSourceLane = min(sourceLane, laneCount - 1u);
        uint sourceValue = WaveReadLaneAt(value, safeSourceLane);
        if ((laneIndex % (laneOffset << 1u)) == 0u && sourceLane < laneCount) {
            value += sourceValue;
        }
    }
    return WaveReadLaneAt(value, 0u);
}

uint reducePixelTo2DGSMax(uint value, uint laneIndex, uint laneCount) {
    for (uint laneOffset = 1u; laneOffset < laneCount; laneOffset <<= 1u) {
        uint sourceLane = laneIndex + laneOffset;
        uint safeSourceLane = min(sourceLane, laneCount - 1u);
        uint sourceValue = WaveReadLaneAt(value, safeSourceLane);
        if ((laneIndex % (laneOffset << 1u)) == 0u && sourceLane < laneCount) {
            value = max(value, sourceValue);
        }
    }
    return WaveReadLaneAt(value, 0u);
}

void reduceProjectedGaussianGrad(inout ProjectedGaussianGrad grad,
                                 uint laneIndex,
                                 uint laneCount) {
    for (uint laneOffset = 1u; laneOffset < laneCount; laneOffset <<= 1u) {
        uint sourceLane = laneIndex + laneOffset;
        uint safeSourceLane = min(sourceLane, laneCount - 1u);
        float2 sourceCenter = WaveReadLaneAt(grad.centerRadius.xy, safeSourceLane);
        float4 sourceConicOpacity = WaveReadLaneAt(grad.conicOpacity, safeSourceLane);
        float3 sourceColor = WaveReadLaneAt(grad.color.xyz, safeSourceLane);
        if ((laneIndex % (laneOffset << 1u)) == 0u && sourceLane < laneCount) {
            grad.centerRadius.xy += sourceCenter;
            grad.conicOpacity += sourceConicOpacity;
            grad.color.xyz += sourceColor;
        }
    }
}

void backpropPixelTo2DGSDirect(uint processedCount,
                               TileRange range,
                               float2 samplePosition,
                               float3 dLoss_dPixelColor,
                               float finalTransmittance) {
    float3 suffixColor = float3(0.0);
    float transmittanceBefore = finalTransmittance;
    for (uint reverseIndex = 0u; reverseIndex < processedCount; ++reverseIndex) {
        uint itemOffset = processedCount - 1u - reverseIndex;
        uint gaussianIndex = tileItems[range.start + itemOffset];
        if (gaussianIndex >= params.gaussianCount) {
            continue;
        }

        ProjectedGaussian gaussian = projectedGaussians[gaussianIndex];
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
    uint subgroupMaxProcessedCount = reducePixelTo2DGSMax(
        processedCount, laneIndex, laneCount);
    float2 samplePosition = pixelCenter(pixel);

#if TRAIN_PIXEL_TO_2DGS_ADAPTIVE
    if (subgroupMaxProcessedCount <= laneCount) {
        if (validPixel) {
            backpropPixelTo2DGSDirect(processedCount,
                                      range,
                                      samplePosition,
                                      dLoss_dPixelColor,
                                      blendState.finalTransmittance);
        }
        return;
    }
#endif

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
            ProjectedGaussianGrad grad = makeZeroProjectedGaussianGrad();
            uint contributionCount = 0u;
            if (validPixel && itemOffset < processedCount &&
                gaussianIndex < params.gaussianCount) {
                ProjectedGaussian gaussian;
                gaussian.centerRadius = centerRadius;
                gaussian.conicOpacity = conicOpacity;
                gaussian.color = color;
                GaussianSample sample = evaluateGaussianSample(gaussian, samplePosition);
                if (contributesToPixel(sample)) {
                    float oneMinusAlpha = max(1.0 - sample.alpha, 1e-6);
                    transmittanceBefore /= oneMinusAlpha;

                    float3 dLoss_dGaussianColor =
                        dLoss_dPixelColor * (transmittanceBefore * sample.alpha);
                    float dLoss_dAlpha = dot(
                        dLoss_dPixelColor,
                        transmittanceBefore * (sample.color - suffixColor));
                    grad = backwardGaussianSample(
                        gaussian, sample, dLoss_dAlpha, dLoss_dGaussianColor);
                    contributionCount = 1u;

                    suffixColor = sample.alpha * sample.color + oneMinusAlpha * suffixColor;
                }
            }

            uint subgroupContributionCount = reducePixelTo2DGSUint(
                contributionCount, laneIndex, laneCount);
            if (subgroupContributionCount > 0u) {
                reduceProjectedGaussianGrad(grad, laneIndex, laneCount);
                if (laneIndex == 0u) {
                    atomicAddProjectedGaussianGrad(projectedGrads, gaussianIndex, grad);
                }
            }
        }
        remaining = batchStart;
    }
}
