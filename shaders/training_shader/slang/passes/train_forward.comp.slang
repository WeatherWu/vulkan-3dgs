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

PixelComposite renderPixelFromTile(uint2 pixel) {
    PixelComposite composite = makeEmptyComposite();
    float2 samplePosition = pixelCenter(pixel);
    TileRange range = unpackTileRange(tileRanges[tileIndexForPixel(pixel)]);

    for (uint itemOffset = 0; itemOffset < range.count; ++itemOffset) {
        uint gaussianIndex = tileItems[range.start + itemOffset];
        if (gaussianIndex >= params.gaussianCount) {
            continue;
        }

        GaussianSample sample = evaluateGaussianSample(projectedGaussians[gaussianIndex], samplePosition);
        if (!contributesToPixel(sample)) {
            continue;
        }

        float testTransmittance = composite.transmittance * (1.0 - sample.alpha);
        if (testTransmittance < TRAIN_TRANSMITTANCE_MIN) {
            break;
        }

        compositeGaussianFrontToBack(composite, sample);
        composite.processedCount = itemOffset + 1u;
        composite.contributionCount += 1u;
    }

    return composite;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint2 pixel = dispatchThreadID.xy;
    if (pixel.x >= params.width || pixel.y >= params.height) {
        return;
    }

    uint index = pixelIndex(pixel);
    PixelComposite composite = renderPixelFromTile(pixel);
    renderedColor[index] = finalizeComposite(composite);

    PixelBlendState blendState;
    blendState.finalTransmittance = composite.transmittance;
    blendState.accumulatedAlpha = composite.accumulatedAlpha;
    blendState.processedCount = composite.processedCount;
    blendState.contributionCount = composite.contributionCount;
    pixelBlendStates[index] = blendState;
}
