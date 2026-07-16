#include "../common/train_backward_common.slang"
#include "../common/train_target_common.slang"

[[vk::binding(6, 0)]]
StructuredBuffer<float4> renderedColor;

[[vk::binding(7, 0)]]
StructuredBuffer<uint> targetColor;

[[vk::binding(12, 0)]]
RWStructuredBuffer<PixelGrad> pixelGrads;

[[vk::binding(28, 0)]]
StructuredBuffer<SsimBackwardState> ssimBackwardStates;

static const int BACKWARD_SSIM_RADIUS = 5;
static const float BACKWARD_SSIM_SIGMA = 1.5;
static const float BACKWARD_SSIM_INV_WEIGHT_SUM = 0.0707622378;

float backwardSsimWindowWeight(int2 offset) {
    float2 value = float2(offset);
    return exp(-dot(value, value) / (2.0 * BACKWARD_SSIM_SIGMA * BACKWARD_SSIM_SIGMA));
}

uint backwardPixelIndex(int2 pixel) {
    return uint(pixel.y * int(params.width) + pixel.x);
}

float3 dssimDerivative(uint sampleIndex) {
    int2 samplePixel = int2(int(sampleIndex % params.width), int(sampleIndex / params.width));
    float3 rendered = renderedColor[sampleIndex].xyz;
    float3 target = unpackTargetColor(targetColor[sampleIndex]).xyz;
    int2 minCenter = max(samplePixel - int2(BACKWARD_SSIM_RADIUS), int2(0));
    int2 maxCenter = min(samplePixel + int2(BACKWARD_SSIM_RADIUS), int2(int(params.width) - 1, int(params.height) - 1));
    float3 derivative = float3(0.0);

    for (int centerY = minCenter.y; centerY <= maxCenter.y; ++centerY) {
        for (int centerX = minCenter.x; centerX <= maxCenter.x; ++centerX) {
            int2 center = int2(centerX, centerY);
            int2 offset = samplePixel - center;
            float sampleWeight = backwardSsimWindowWeight(offset) * BACKWARD_SSIM_INV_WEIGHT_SUM;
            SsimBackwardState state = ssimBackwardStates[backwardPixelIndex(center)];
            derivative += sampleWeight *
                (state.base.xyz +
                 state.targetCoefficient.xyz * target +
                 state.renderedCoefficient.xyz * rendered);
        }
    }

    return derivative;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint index = dispatchThreadID.x;
    if (index >= params.pixelCount) {
        return;
    }

    float3 rendered = renderedColor[index].xyz;
    float3 target = unpackTargetColor(targetColor[index]).xyz;
    float3 l1Grad = float3(
        l1Derivative(rendered.x, target.x),
        l1Derivative(rendered.y, target.y),
        l1Derivative(rendered.z, target.z)
    ) / 3.0;

    float dssimWeight = clamp(params.lossDssimWeight, 0.0, 1.0);
    float invPixelCount = 1.0 / max(float(params.pixelCount), 1.0);

    PixelGrad grad;
    grad.color = float4(((1.0 - dssimWeight) * l1Grad + dssimWeight * dssimDerivative(index)) *
                        invPixelCount,
                        0.0);
    pixelGrads[index] = grad;
}
