#include "train_backward_common.slang"

static const uint TRAIN_TILE_GAUSSIAN_BATCH_SIZE = 32u;
static const uint TRAIN_TILE_GAUSSIAN_MAX_SUBGROUPS = 16u;

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

groupshared uint sharedGaussianIndices[TRAIN_TILE_GAUSSIAN_BATCH_SIZE];
groupshared uint sharedSubgroupAllocator;
groupshared float4 sharedCenterRadius[TRAIN_TILE_GAUSSIAN_BATCH_SIZE];
groupshared float4 sharedConicOpacity[TRAIN_TILE_GAUSSIAN_BATCH_SIZE];
groupshared float4 sharedColor[TRAIN_TILE_GAUSSIAN_BATCH_SIZE];
groupshared ProjectedGaussianGrad sharedSubgroupGrads[
    TRAIN_TILE_GAUSSIAN_BATCH_SIZE * TRAIN_TILE_GAUSSIAN_MAX_SUBGROUPS];
groupshared uint sharedSubgroupContributionCounts[
    TRAIN_TILE_GAUSSIAN_BATCH_SIZE * TRAIN_TILE_GAUSSIAN_MAX_SUBGROUPS];

void addProjectedGrad(inout ProjectedGaussianGrad destination,
                      ProjectedGaussianGrad source) {
    destination.centerRadius.xy += source.centerRadius.xy;
    destination.conicOpacity += source.conicOpacity;
    destination.color.xyz += source.color.xyz;
}

void reduceTileGaussianSubgroupGrad(inout ProjectedGaussianGrad grad,
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

uint reduceTileGaussianSubgroupCount(uint value,
                                     uint laneIndex,
                                     uint laneCount) {
    for (uint laneOffset = 1u; laneOffset < laneCount; laneOffset <<= 1u) {
        uint sourceLane = laneIndex + laneOffset;
        uint safeSourceLane = min(sourceLane, laneCount - 1u);
        uint sourceValue = WaveReadLaneAt(value, safeSourceLane);
        if ((laneIndex % (laneOffset << 1u)) == 0u && sourceLane < laneCount) {
            value += sourceValue;
        }
    }
    return value;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID,
          uint3 groupThreadID : SV_GroupThreadID) {
    uint2 pixel = dispatchThreadID.xy;
    uint localIndex = groupThreadID.y * TRAIN_TILE_SIZE + groupThreadID.x;
    bool validPixel = pixel.x < params.width && pixel.y < params.height;
    TileRange range = unpackTileRange(tileRanges[tileIndexForPixel(pixel)]);
    uint safeRangeStart = min(range.start, params.tileItemCount);
    uint rangeCount = min(range.count, params.tileItemCount - safeRangeStart);

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

    uint processedCount = validPixel ? min(blendState.processedCount, rangeCount) : 0u;
    uint laneCount = WaveGetLaneCount();
    uint laneIndex = WaveGetLaneIndex();
    if (localIndex == 0u) {
        sharedSubgroupAllocator = 0u;
    }
    GroupMemoryBarrierWithGroupSync();
    uint subgroupIndex = 0u;
    if (laneIndex == 0u) {
        InterlockedAdd(sharedSubgroupAllocator, 1u, subgroupIndex);
    }
    subgroupIndex = WaveReadLaneFirst(subgroupIndex);
    GroupMemoryBarrierWithGroupSync();
    uint subgroupCount = sharedSubgroupAllocator;
    float2 samplePosition = pixelCenter(pixel);
    float3 suffixColor = float3(0.0);
    float transmittanceBefore = blendState.finalTransmittance;
    uint remaining = rangeCount;

    while (remaining > 0u) {
        uint batchCount = min(remaining, TRAIN_TILE_GAUSSIAN_BATCH_SIZE);
        uint batchStart = remaining - batchCount;
        if (localIndex < batchCount) {
            uint sortedPosition = safeRangeStart + batchStart + localIndex;
            uint gaussianIndex = tileItems[sortedPosition];
            sharedGaussianIndices[localIndex] = gaussianIndex;
            if (gaussianIndex < params.gaussianCount) {
                ProjectedGaussian gaussian = projectedGaussians[gaussianIndex];
                sharedCenterRadius[localIndex] = gaussian.centerRadius;
                sharedConicOpacity[localIndex] = gaussian.conicOpacity;
                sharedColor[localIndex] = gaussian.color;
            } else {
                sharedCenterRadius[localIndex] = float4(0.0);
                sharedConicOpacity[localIndex] = float4(0.0);
                sharedColor[localIndex] = float4(0.0);
            }
        }
        GroupMemoryBarrierWithGroupSync();

        for (uint reverseIndex = 0u; reverseIndex < batchCount; ++reverseIndex) {
            uint sharedIndex = batchCount - 1u - reverseIndex;
            uint itemOffset = batchStart + sharedIndex;
            uint gaussianIndex = sharedGaussianIndices[sharedIndex];
            ProjectedGaussianGrad grad = makeZeroProjectedGaussianGrad();
            uint contributionCount = 0u;
            if (validPixel && itemOffset < processedCount &&
                gaussianIndex < params.gaussianCount) {
                ProjectedGaussian gaussian;
                gaussian.centerRadius = sharedCenterRadius[sharedIndex];
                gaussian.conicOpacity = sharedConicOpacity[sharedIndex];
                gaussian.color = sharedColor[sharedIndex];
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
                    suffixColor = sample.alpha * sample.color +
                                  oneMinusAlpha * suffixColor;
                }
            }

            reduceTileGaussianSubgroupGrad(grad, laneIndex, laneCount);
            contributionCount = reduceTileGaussianSubgroupCount(
                contributionCount, laneIndex, laneCount);
            if (laneIndex == 0u && subgroupIndex < TRAIN_TILE_GAUSSIAN_MAX_SUBGROUPS) {
                uint partialIndex = subgroupIndex * TRAIN_TILE_GAUSSIAN_BATCH_SIZE + sharedIndex;
                sharedSubgroupGrads[partialIndex] = grad;
                sharedSubgroupContributionCounts[partialIndex] = contributionCount;
            }
        }
        GroupMemoryBarrierWithGroupSync();

        if (localIndex < batchCount) {
            ProjectedGaussianGrad tileGrad = makeZeroProjectedGaussianGrad();
            uint tileContributionCount = 0u;
            for (uint subgroup = 0u;
                 subgroup < min(subgroupCount, TRAIN_TILE_GAUSSIAN_MAX_SUBGROUPS);
                 ++subgroup) {
                uint partialIndex = subgroup * TRAIN_TILE_GAUSSIAN_BATCH_SIZE + localIndex;
                addProjectedGrad(tileGrad, sharedSubgroupGrads[partialIndex]);
                tileContributionCount += sharedSubgroupContributionCounts[partialIndex];
            }

            uint gaussianIndex = sharedGaussianIndices[localIndex];
            if (tileContributionCount > 0u && gaussianIndex < params.gaussianCount) {
                atomicAddProjectedGaussianGrad(projectedGrads, gaussianIndex, tileGrad);
            }
        }
        GroupMemoryBarrierWithGroupSync();
        remaining = batchStart;
    }
}
