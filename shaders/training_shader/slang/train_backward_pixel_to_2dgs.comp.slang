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

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint2 pixel = dispatchThreadID.xy;
    if (pixel.x >= params.width || pixel.y >= params.height) {
        return;
    }

    float3 dLoss_dPixelColor = pixelGrads[pixelIndex(pixel)].color.xyz;
    float2 samplePosition = pixelCenter(pixel);
    TileRange range = unpackTileRange(tileRanges[tileIndexForPixel(pixel)]);
    float transmittance = 1.0;

    for (uint itemOffset = 0; itemOffset < range.count; ++itemOffset) {
        uint gaussianIndex = tileItems[range.start + itemOffset];
        if (gaussianIndex >= params.gaussianCount) {
            continue;
        }

        ProjectedGaussian gaussian = projectedGaussians[gaussianIndex];
        GaussianSample sample = evaluateGaussianSample(gaussian, samplePosition);
        if (!contributesToPixel(sample)) {
            continue;
        }

        float3 suffixColor = renderSuffixColor(projectedGaussians, tileItems, range, itemOffset + 1, samplePosition);
        float3 dLoss_dGaussianColor = dLoss_dPixelColor * (transmittance * sample.alpha);
        float dLoss_dAlpha = dot(dLoss_dPixelColor, transmittance * (sample.color - suffixColor));
        ProjectedGaussianGrad grad = backwardGaussianSample(gaussian, sample, dLoss_dAlpha, dLoss_dGaussianColor);
        atomicAddProjectedGaussianGrad(projectedGrads, gaussianIndex, grad);

        transmittance *= 1.0 - sample.alpha;
        if (transmittance <= TRAIN_TRANSMITTANCE_MIN) {
            break;
        }
    }
}
