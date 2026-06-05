#include "train_common.slang"

[[vk::binding(1, 0)]]
RWStructuredBuffer<GaussianGrad> gaussianGrads;

[[vk::binding(6, 0)]]
RWStructuredBuffer<float4> renderedColor;

[[vk::binding(8, 0)]]
RWStructuredBuffer<float> lossBuffer;

[[vk::binding(9, 0)]]
RWStructuredBuffer<uint4> counters;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint index = dispatchThreadID.x;

    if (index < params.gaussianCount) {
        GaussianGrad zeroGrad;
        zeroGrad.positionOpacity = 0.0;
        zeroGrad.scale = 0.0;
        zeroGrad.rotation = 0.0;
        for (uint i = 0; i < 16; ++i) {
            zeroGrad.sh[i] = 0.0;
        }
        gaussianGrads[index] = zeroGrad;
    }

    if (index < params.pixelCount) {
        renderedColor[index] = float4(0.0, 0.0, 0.0, 1.0);
    }

    if (index == 0) {
        lossBuffer[0] = 0.0;
        counters[0] = uint4(0, 0, 0, 0);
    }
}
