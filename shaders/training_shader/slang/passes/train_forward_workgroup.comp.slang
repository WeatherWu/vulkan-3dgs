#include "../common/train_forward_common.slang"

[[vk::binding(3, 0)]]
StructuredBuffer<ProjectedGaussian> projectedGaussians;

[[vk::binding(4, 0)]]
StructuredBuffer<uint> tileItems;

[[vk::binding(5, 0)]]
StructuredBuffer<uint4> tileRanges;

[[vk::binding(6, 0)]]
RWStructuredBuffer<float4> renderedColor;

[[vk::binding(14, 0)]]
RWStructuredBuffer<PixelBlendState> pixelBlendStates;

groupshared uint sharedGaussianIndices[TRAIN_TILE_PIXEL_COUNT];
groupshared float4 sharedCenterRadius[TRAIN_TILE_PIXEL_COUNT];
groupshared float4 sharedConicOpacity[TRAIN_TILE_PIXEL_COUNT];
groupshared float4 sharedColor[TRAIN_TILE_PIXEL_COUNT];
groupshared uint sharedActivePixelCount;

[shader("compute")]
[numthreads(TRAIN_TILE_SIZE, TRAIN_TILE_SIZE, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID,
          uint3 groupThreadID : SV_GroupThreadID,
          uint3 groupID : SV_GroupID) {
    uint2 pixel = dispatchThreadID.xy;
    TileRange range = unpackTileRange(tileRanges[tileIndexForPixel(pixel)]);
    uint localIndex = groupThreadID.y * TRAIN_TILE_SIZE + groupThreadID.x;
    bool validPixel = pixel.x < params.width && pixel.y < params.height;
    uint pixelLinearIndex = validPixel ? pixelIndex(pixel) : 0u;
    float2 samplePosition = pixelCenter(pixel);
    PixelComposite composite = makeEmptyComposite();
    bool finished = !validPixel;
    if (localIndex == 0u) {
        sharedActivePixelCount = 0u;
    }
    GroupMemoryBarrierWithGroupSync();
    if (validPixel) {
        uint ignoredActiveCount = 0u;
        InterlockedAdd(sharedActivePixelCount, 1u, ignoredActiveCount);
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint batchStart = 0u; batchStart < range.count;
         batchStart += TRAIN_TILE_PIXEL_COUNT) {
        uint batchCount = min(range.count - batchStart, TRAIN_TILE_PIXEL_COUNT);
        if (localIndex < batchCount) {
            uint gaussianIndex = tileItems[range.start + batchStart + localIndex];
            sharedGaussianIndices[localIndex] = gaussianIndex;
            if (gaussianIndex < params.gaussianCount) {
                ProjectedGaussian gaussian = projectedGaussians[gaussianIndex];
                sharedCenterRadius[localIndex] = gaussian.centerRadius;
                sharedConicOpacity[localIndex] = gaussian.conicOpacity;
                sharedColor[localIndex] = gaussian.color;
            }
        }
        GroupMemoryBarrierWithGroupSync();

        if (!finished) {
            for (uint batchOffset = 0u; batchOffset < batchCount; ++batchOffset) {
                uint gaussianIndex = sharedGaussianIndices[batchOffset];
                if (gaussianIndex >= params.gaussianCount) {
                    continue;
                }

                ProjectedGaussian gaussian;
                gaussian.centerRadius = sharedCenterRadius[batchOffset];
                gaussian.conicOpacity = sharedConicOpacity[batchOffset];
                gaussian.color = sharedColor[batchOffset];
                GaussianSample sample = evaluateGaussianSample(gaussian, samplePosition);
                if (!contributesToPixel(sample)) {
                    continue;
                }

                float testTransmittance = composite.transmittance * (1.0 - sample.alpha);
                if (testTransmittance < TRAIN_TRANSMITTANCE_MIN) {
                    finished = true;
                    uint ignoredActiveCount = 0u;
                    InterlockedAdd(sharedActivePixelCount, 0xffffffffu, ignoredActiveCount);
                    break;
                }

                compositeGaussianFrontToBack(composite, sample);
                composite.processedCount = batchStart + batchOffset + 1u;
                composite.contributionCount += 1u;
            }
        }
        GroupMemoryBarrierWithGroupSync();
        if (sharedActivePixelCount == 0u) {
            break;
        }
    }

    if (!validPixel) {
        return;
    }

    renderedColor[pixelLinearIndex] = finalizeComposite(composite);
    PixelBlendState blendState;
    blendState.finalTransmittance = composite.transmittance;
    blendState.accumulatedAlpha = composite.accumulatedAlpha;
    blendState.processedCount = composite.processedCount;
    blendState.contributionCount = composite.contributionCount;
    pixelBlendStates[pixelLinearIndex] = blendState;
}
