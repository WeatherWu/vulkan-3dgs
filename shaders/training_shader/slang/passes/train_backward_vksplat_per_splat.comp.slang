#include "../common/train_backward_common.slang"

static const uint VKSPLAT_PER_SPLAT_MAX_BATCH_SIZE = 128u;
static const uint VKSPLAT_TILE_PIXEL_COUNT = TRAIN_TILE_SIZE * TRAIN_TILE_SIZE;

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

groupshared float4 sharedReversePixelState[VKSPLAT_TILE_PIXEL_COUNT];
groupshared float4 sharedPixelGrad[VKSPLAT_TILE_PIXEL_COUNT];
groupshared uint sharedProcessedCount[VKSPLAT_TILE_PIXEL_COUNT];
groupshared uint sharedSplatBatchSize;

void addProjectedGrad(inout ProjectedGaussianGrad destination,
                      ProjectedGaussianGrad source) {
    destination.centerRadius.xy += source.centerRadius.xy;
    destination.conicOpacity += source.conicOpacity;
    destination.color.xyz += source.color.xyz;
}

[shader("compute")]
[numthreads(128, 1, 1)]
void main(uint3 groupThreadID : SV_GroupThreadID,
          uint3 groupID : SV_GroupID) {
    uint threadRank = groupThreadID.x;
    uint tileIndex = groupID.y * tileCountX() + groupID.x;
    TileRange range = unpackTileRange(tileRanges[tileIndex]);
    uint safeRangeStart = min(range.start, params.tileItemCount);
    uint rangeCount = min(range.count, params.tileItemCount - safeRangeStart);
    uint2 tileOrigin = groupID.xy * TRAIN_TILE_SIZE;

    for (uint pixelRank = threadRank;
         pixelRank < VKSPLAT_TILE_PIXEL_COUNT;
         pixelRank += VKSPLAT_PER_SPLAT_MAX_BATCH_SIZE) {
        uint2 localPixel = uint2(
            pixelRank % TRAIN_TILE_SIZE,
            pixelRank / TRAIN_TILE_SIZE);
        uint2 pixel = tileOrigin + localPixel;
        bool validPixel = pixel.x < params.width && pixel.y < params.height;
        if (validPixel) {
            uint pixelLinearIndex = pixelIndex(pixel);
            PixelBlendState blendState = pixelBlendStates[pixelLinearIndex];
            sharedReversePixelState[pixelRank] = float4(
                0.0, 0.0, 0.0, blendState.finalTransmittance);
            sharedPixelGrad[pixelRank] = pixelGrads[pixelLinearIndex].color;
            sharedProcessedCount[pixelRank] = min(blendState.processedCount, rangeCount);
        } else {
            sharedReversePixelState[pixelRank] = float4(0.0, 0.0, 0.0, 1.0);
            sharedPixelGrad[pixelRank] = float4(0.0);
            sharedProcessedCount[pixelRank] = 0u;
        }
    }

    if (threadRank == 0u) {
        uint subgroupSize = max(WaveGetLaneCount(), 1u);
        uint estimatedBatch = uint(ceil(sqrt(
            float(max(rangeCount, 1u) * VKSPLAT_TILE_PIXEL_COUNT))));
        estimatedBatch = ((estimatedBatch + subgroupSize - 1u) / subgroupSize) * subgroupSize;
        sharedSplatBatchSize = clamp(
            estimatedBatch, subgroupSize, VKSPLAT_PER_SPLAT_MAX_BATCH_SIZE);
    }
    GroupMemoryBarrierWithGroupSync();

    uint batchSize = sharedSplatBatchSize;
    uint batchBase = 0u;
    while (batchBase < rangeCount) {
        uint batchCount = min(batchSize, rangeCount - batchBase);
        bool activeSplat = threadRank < batchCount;
        uint itemOffset = activeSplat
            ? rangeCount - 1u - (batchBase + threadRank)
            : 0u;
        uint gaussianIndex = activeSplat
            ? tileItems[safeRangeStart + itemOffset]
            : 0xffffffffu;
        ProjectedGaussian gaussian;
        gaussian.centerRadius = float4(0.0);
        gaussian.conicOpacity = float4(0.0);
        gaussian.color = float4(0.0);
        if (gaussianIndex < params.gaussianCount) {
            gaussian = projectedGaussians[gaussianIndex];
        }
        ProjectedGaussianGrad accumulatedGrad = makeZeroProjectedGaussianGrad();
        uint contributionCount = 0u;

        uint wavefrontSteps = batchCount + VKSPLAT_TILE_PIXEL_COUNT - 1u;
        for (uint step = 0u; step < wavefrontSteps; ++step) {
            int pixelRank = int(step) - int(threadRank);
            if (activeSplat && pixelRank >= 0 &&
                pixelRank < int(VKSPLAT_TILE_PIXEL_COUNT) &&
                itemOffset < sharedProcessedCount[uint(pixelRank)]) {
                uint2 localPixel = uint2(
                    uint(pixelRank) % TRAIN_TILE_SIZE,
                    uint(pixelRank) / TRAIN_TILE_SIZE);
                uint2 pixel = tileOrigin + localPixel;
                float4 reverseState = sharedReversePixelState[uint(pixelRank)];
                float3 suffixColor = reverseState.xyz;
                float transmittanceBefore = reverseState.w;
                GaussianSample sample = evaluateGaussianSample(
                    gaussian, pixelCenter(pixel));
                if (contributesToPixel(sample)) {
                    float oneMinusAlpha = max(1.0 - sample.alpha, 1e-6);
                    transmittanceBefore /= oneMinusAlpha;
                    float3 dLoss_dPixelColor = sharedPixelGrad[uint(pixelRank)].xyz;
                    float3 dLoss_dGaussianColor =
                        dLoss_dPixelColor * (transmittanceBefore * sample.alpha);
                    float dLoss_dAlpha = dot(
                        dLoss_dPixelColor,
                        transmittanceBefore * (sample.color - suffixColor));
                    ProjectedGaussianGrad grad = backwardGaussianSample(
                        gaussian, sample, dLoss_dAlpha, dLoss_dGaussianColor);
                    addProjectedGrad(accumulatedGrad, grad);
                    ++contributionCount;
                    suffixColor = sample.alpha * sample.color +
                                  oneMinusAlpha * suffixColor;
                    sharedReversePixelState[uint(pixelRank)] = float4(
                        suffixColor, transmittanceBefore);
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }

        if (activeSplat && contributionCount > 0u &&
            gaussianIndex < params.gaussianCount) {
            atomicAddProjectedGaussianGrad(
                projectedGrads, gaussianIndex, accumulatedGrad);
        }
        batchBase += batchCount;
        GroupMemoryBarrierWithGroupSync();
    }
}
