#include "../common/train_common.slang"
#include "../common/train_backward_types.slang"
#include "../common/train_loss_common.slang"
#include "../common/train_target_common.slang"

[[vk::binding(6, 0)]]
StructuredBuffer<float4> renderedColor;

[[vk::binding(7, 0)]]
StructuredBuffer<uint> targetColor;

[[vk::binding(8, 0)]]
RWStructuredBuffer<float> lossBuffer;

[[vk::binding(28, 0)]]
RWStructuredBuffer<SsimBackwardState> ssimBackwardStates;

static const int SSIM_RADIUS = 5;
static const float SSIM_C1 = 0.0001;
static const float SSIM_C2 = 0.0009;
static const float SSIM_SIGMA = 1.5;

float ssimWindowWeight(int2 offset) {
    float2 value = float2(offset);
    return exp(-dot(value, value) / (2.0 * SSIM_SIGMA * SSIM_SIGMA));
}

bool pixelInBounds(int2 pixel) {
    return pixel.x >= 0 && pixel.y >= 0 &&
           pixel.x < int(params.width) && pixel.y < int(params.height);
}

uint pixelIndex(int2 pixel) {
    return uint(pixel.y * int(params.width) + pixel.x);
}

float ssimColor(uint centerIndex, out SsimBackwardState backwardState) {
    int width = int(params.width);
    int2 center = int2(int(centerIndex % params.width), int(centerIndex / params.width));

    float3 renderedMean = float3(0.0);
    float3 targetMean = float3(0.0);
    float3 renderedSqMean = float3(0.0);
    float3 targetSqMean = float3(0.0);
    float3 renderedTargetMean = float3(0.0);
    float weightSum = 0.0;

    for (int dy = -SSIM_RADIUS; dy <= SSIM_RADIUS; ++dy) {
        for (int dx = -SSIM_RADIUS; dx <= SSIM_RADIUS; ++dx) {
            float weight = ssimWindowWeight(int2(dx, dy));
            int2 samplePixel = center + int2(dx, dy);
            float3 rendered = float3(0.0);
            float3 target = float3(0.0);
            if (pixelInBounds(samplePixel)) {
                uint sampleIndex = pixelIndex(samplePixel);
                rendered = renderedColor[sampleIndex].xyz;
                target = unpackTargetColor(targetColor[sampleIndex]).xyz;
            }

            renderedMean += rendered * weight;
            targetMean += target * weight;
            renderedSqMean += rendered * rendered * weight;
            targetSqMean += target * target * weight;
            renderedTargetMean += rendered * target * weight;
            weightSum += weight;
        }
    }

    renderedMean /= weightSum;
    targetMean /= weightSum;
    renderedSqMean /= weightSum;
    targetSqMean /= weightSum;
    renderedTargetMean /= weightSum;

    float3 renderedVariance = max(renderedSqMean - renderedMean * renderedMean, float3(0.0));
    float3 targetVariance = max(targetSqMean - targetMean * targetMean, float3(0.0));
    float3 covariance = renderedTargetMean - renderedMean * targetMean;

    float3 A = 2.0 * renderedMean * targetMean + SSIM_C1;
    float3 B = 2.0 * covariance + SSIM_C2;
    float3 C = renderedMean * renderedMean + targetMean * targetMean + SSIM_C1;
    float3 D = renderedVariance + targetVariance + SSIM_C2;
    float3 rawSsim = (A * B) / max(C * D, float3(1e-8));
    float3 clampMask = float3(
        rawSsim.x > 0.0 && rawSsim.x < 1.0 ? 1.0 : 0.0,
        rawSsim.y > 0.0 && rawSsim.y < 1.0 ? 1.0 : 0.0,
        rawSsim.z > 0.0 && rawSsim.z < 1.0 ? 1.0 : 0.0
    );
    float3 derivativeScale = -clampMask * rawSsim / 3.0;
    float3 safeA = max(A, float3(1e-8));
    float3 safeB = max(B, float3(1e-8));
    float3 safeC = max(C, float3(1e-8));
    float3 safeD = max(D, float3(1e-8));
    backwardState.base = float4(derivativeScale *
        (2.0 * targetMean / safeA -
         2.0 * targetMean / safeB -
         2.0 * renderedMean / safeC +
         2.0 * renderedMean / safeD), 0.0);
    backwardState.targetCoefficient = float4(derivativeScale * (2.0 / safeB), 0.0);
    backwardState.renderedCoefficient = float4(derivativeScale * (-2.0 / safeD), 0.0);

    float3 ssim = clamp(rawSsim, float3(0.0), float3(1.0));

    return (ssim.x + ssim.y + ssim.z) / 3.0;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint index = dispatchThreadID.x;
    if (index >= params.pixelCount) {
        return;
    }

    SsimBackwardState backwardState;
    float l1 = l1Color(renderedColor[index].xyz, unpackTargetColor(targetColor[index]).xyz);
    float dssim = clamp(1.0 - ssimColor(index, backwardState), 0.0, 1.0);
    ssimBackwardStates[index] = backwardState;
    float dssimWeight = clamp(params.lossDssimWeight, 0.0, 1.0);
    lossBuffer[index] = ((1.0 - dssimWeight) * l1 + dssimWeight * dssim) /
                        max(float(params.pixelCount), 1.0);
}
