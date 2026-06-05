#include "train_common.slang"

[[vk::binding(3, 0)]]
StructuredBuffer<ProjectedGaussian> projectedGaussians;

[[vk::binding(6, 0)]]
RWStructuredBuffer<float4> renderedColor;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint2 pixel = dispatchThreadID.xy;
    if (pixel.x >= params.width || pixel.y >= params.height) {
        return;
    }

    uint index = pixelIndex(pixel);
    float4 color = float4(0.0, 0.0, 0.0, 1.0);
    if (params.gaussianCount > 0) {
        color = projectedGaussians[0].color;
        color.w = 1.0;
    }
    renderedColor[index] = color;
}
