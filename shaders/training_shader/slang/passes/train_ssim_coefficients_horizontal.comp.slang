#include "../common/train_common.slang"
#include "../common/train_backward_types.slang"
#include "../common/train_ssim_separable.slang"

[[vk::binding(28, 0)]] StructuredBuffer<SsimBackwardState> ssimBackwardStates;
[[vk::binding(42, 0)]] RWStructuredBuffer<float4> ssimScratch;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 thread : SV_DispatchThreadID) {
    if (thread.x >= params.width || thread.y >= params.height) return;
    float3 base = float3(0.0);
    float3 targetCoefficient = float3(0.0);
    float3 renderedCoefficient = float3(0.0);
    for (int dx = -SEPARABLE_SSIM_RADIUS; dx <= SEPARABLE_SSIM_RADIUS; ++dx) {
        int x = int(thread.x) + dx;
        if (x < 0 || x >= int(params.width)) continue;
        SsimBackwardState state = ssimBackwardStates[thread.y * params.width + uint(x)];
        float weight = SEPARABLE_SSIM_WEIGHTS[abs(dx)];
        base += state.base.xyz * weight;
        targetCoefficient += state.targetCoefficient.xyz * weight;
        renderedCoefficient += state.renderedCoefficient.xyz * weight;
    }
    uint index = thread.y * params.width + thread.x;
    ssimScratch[index] = float4(base, 0.0);
    ssimScratch[params.pixelCount + index] = float4(targetCoefficient, 0.0);
    ssimScratch[2u * params.pixelCount + index] = float4(renderedCoefficient, 0.0);
}
