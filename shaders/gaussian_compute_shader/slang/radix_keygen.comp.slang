#include "gaussian_common.slang"

struct RadixParams {
    uint count;
    uint shift;
    uint groupCount;
    uint padding;
};

[[vk::binding(0, 0)]]
RWStructuredBuffer<uint> indicesOut;

[[vk::binding(1, 0)]]
RWStructuredBuffer<uint> keysOut;

[[vk::binding(6, 0)]]
StructuredBuffer<GaussianInstance> instances;

[[vk::binding(7, 0)]]
ConstantBuffer<UniformBufferObject> ubo;

[[vk::binding(8, 0)]]
RWStructuredBuffer<uint> drawArgs;

[[vk::push_constant]]
ConstantBuffer<RadixParams> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint index = dispatchThreadID.x;
    if (index >= params.count) {
        return;
    }

    if (index == 0u) {
        drawArgs[0] = 4u;
        drawArgs[2] = 0u;
        drawArgs[3] = 0u;
        drawArgs[4] = 0u;
    }

    float4 centerClip = mul(mul(mul(ubo.projection, ubo.view), ubo.model), float4(instances[index].position.xyz, 1.0));
    if (centerClip.w <= 1e-6) {
        return;
    }

    float3 ndc = centerClip.xyz / centerClip.w;
    static const float CULL_MARGIN_NDC = 2.0;
    if (ndc.x < -CULL_MARGIN_NDC || ndc.x > CULL_MARGIN_NDC ||
        ndc.y < -CULL_MARGIN_NDC || ndc.y > CULL_MARGIN_NDC ||
        ndc.z < 0.0 || ndc.z > 1.0) {
        return;
    }

    uint visibleIndex = 0;
    InterlockedAdd(drawArgs[1], 1u, visibleIndex);
    indicesOut[visibleIndex] = index;
    keysOut[visibleIndex] = asuint(1.0 - ndc.z);
}
