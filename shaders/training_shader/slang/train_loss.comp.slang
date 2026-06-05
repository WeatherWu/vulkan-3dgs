#include "train_common.slang"
#include "train_loss_common.slang"

[[vk::binding(6, 0)]]
StructuredBuffer<float4> renderedColor;

[[vk::binding(7, 0)]]
StructuredBuffer<float4> targetColor;

[[vk::binding(8, 0)]]
RWStructuredBuffer<float> lossBuffer;

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    if (dispatchThreadID.x != 0 || params.pixelCount == 0) {
        return;
    }

    lossBuffer[0] = l1Color(renderedColor[0].xyz, targetColor[0].xyz);
}
