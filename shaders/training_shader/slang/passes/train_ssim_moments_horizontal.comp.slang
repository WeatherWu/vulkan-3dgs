#include "../common/train_common.slang"
#include "../common/train_target_common.slang"
#include "../common/train_ssim_separable.slang"

[[vk::binding(6, 0)]] StructuredBuffer<float4> renderedColor;
[[vk::binding(7, 0)]] StructuredBuffer<uint> targetColor;
[[vk::binding(42, 0)]] RWStructuredBuffer<float4> ssimScratch;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 thread : SV_DispatchThreadID) {
    if (thread.x >= params.width || thread.y >= params.height) return;
    float3 renderedMean = float3(0.0);
    float3 targetMean = float3(0.0);
    float3 renderedSqMean = float3(0.0);
    float3 targetSqMean = float3(0.0);
    float3 renderedTargetMean = float3(0.0);
    for (int dx = -SEPARABLE_SSIM_RADIUS; dx <= SEPARABLE_SSIM_RADIUS; ++dx) {
        int x = int(thread.x) + dx;
        if (x < 0 || x >= int(params.width)) continue;
        uint sampleIndex = thread.y * params.width + uint(x);
        float3 rendered = renderedColor[sampleIndex].xyz;
        float3 target = unpackTargetColor(targetColor[sampleIndex]).xyz;
        float weight = SEPARABLE_SSIM_WEIGHTS[abs(dx)];
        renderedMean += rendered * weight;
        targetMean += target * weight;
        renderedSqMean += rendered * rendered * weight;
        targetSqMean += target * target * weight;
        renderedTargetMean += rendered * target * weight;
    }
    uint index = thread.y * params.width + thread.x;
    ssimScratch[index] = float4(renderedMean, 0.0);
    ssimScratch[params.pixelCount + index] = float4(targetMean, 0.0);
    ssimScratch[2u * params.pixelCount + index] = float4(renderedSqMean, 0.0);
    ssimScratch[3u * params.pixelCount + index] = float4(targetSqMean, 0.0);
    ssimScratch[4u * params.pixelCount + index] = float4(renderedTargetMean, 0.0);
}
