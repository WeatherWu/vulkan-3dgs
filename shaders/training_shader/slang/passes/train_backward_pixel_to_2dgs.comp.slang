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
    if (pixel.x >= params.width || pixel.y >= params.height) {
        return;
    }

    uint pixelLinearIndex = pixelIndex(pixel);
    float3 dLoss_dPixelColor = pixelGrads[pixelLinearIndex].color.xyz;
    float2 samplePosition = pixelCenter(pixel);
    TileRange range = unpackTileRange(tileRanges[tileIndexForPixel(pixel)]);
    PixelBlendState blendState = pixelBlendStates[pixelLinearIndex];
    float finalTransmittance = blendState.finalTransmittance;
    uint processedCount = min(blendState.processedCount, range.count);

    float3 suffixColor = float3(0.0);
    float transmittanceBefore = finalTransmittance;

    for (uint reverseIndex = 0; reverseIndex < processedCount; ++reverseIndex) {
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

        float3 dLoss_dGaussianColor = dLoss_dPixelColor * (transmittanceBefore * sample.alpha);
        float dLoss_dAlpha = dot(dLoss_dPixelColor, transmittanceBefore * (sample.color - suffixColor));
        ProjectedGaussianGrad grad = backwardGaussianSample(gaussian, sample, dLoss_dAlpha, dLoss_dGaussianColor);
        atomicAddProjectedGaussianGrad(projectedGrads, gaussianIndex, grad);

        suffixColor = sample.alpha * sample.color + oneMinusAlpha * suffixColor;
    }
}
