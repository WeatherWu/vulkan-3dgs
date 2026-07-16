#include "../common/train_backward_common.slang"
#include "../common/train_target_common.slang"

[[vk::binding(6, 0)]]
StructuredBuffer<float4> renderedColor;

[[vk::binding(7, 0)]]
StructuredBuffer<uint> targetColor;

[[vk::binding(12, 0)]]
RWStructuredBuffer<PixelGrad> pixelGrads;

static const int BACKWARD_SSIM_RADIUS = 5;
static const float BACKWARD_SSIM_C1 = 0.0001;
static const float BACKWARD_SSIM_C2 = 0.0009;
static const float BACKWARD_SSIM_SIGMA = 1.5;

float backwardSsimWindowWeight(int2 offset) {
    float2 value = float2(offset);
    return exp(-dot(value, value) / (2.0 * BACKWARD_SSIM_SIGMA * BACKWARD_SSIM_SIGMA));
}

uint backwardClampedPixelIndex(int2 pixel) {
    int x = clamp(pixel.x, 0, int(params.width) - 1);
    int y = clamp(pixel.y, 0, int(params.height) - 1);
    return uint(y * int(params.width) + x);
}

void ssimWindowStats(uint centerIndex,
                     out float3 renderedMean,
                     out float3 targetMean,
                     out float3 renderedSqMean,
                     out float3 targetSqMean,
                     out float3 renderedTargetMean,
                     out float weightSum) {
    int2 center = int2(int(centerIndex % params.width), int(centerIndex / params.width));
    renderedMean = float3(0.0);
    targetMean = float3(0.0);
    renderedSqMean = float3(0.0);
    targetSqMean = float3(0.0);
    renderedTargetMean = float3(0.0);
    weightSum = 0.0;

    for (int dy = -BACKWARD_SSIM_RADIUS; dy <= BACKWARD_SSIM_RADIUS; ++dy) {
        for (int dx = -BACKWARD_SSIM_RADIUS; dx <= BACKWARD_SSIM_RADIUS; ++dx) {
            float weight = backwardSsimWindowWeight(int2(dx, dy));
            uint sampleIndex = backwardClampedPixelIndex(center + int2(dx, dy));
            float3 rendered = renderedColor[sampleIndex].xyz;
            float3 target = unpackTargetColor(targetColor[sampleIndex]).xyz;

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
}

float3 ssimWindowDerivativeForSample(uint centerIndex, uint sampleIndex, float sampleWeight) {
    float3 renderedMean;
    float3 targetMean;
    float3 renderedSqMean;
    float3 targetSqMean;
    float3 renderedTargetMean;
    float weightSum;
    ssimWindowStats(centerIndex, renderedMean, targetMean, renderedSqMean, targetSqMean, renderedTargetMean, weightSum);

    float3 rendered = renderedColor[sampleIndex].xyz;
    float3 target = unpackTargetColor(targetColor[sampleIndex]).xyz;
    float3 renderedVariance = max(renderedSqMean - renderedMean * renderedMean, float3(0.0));
    float3 targetVariance = max(targetSqMean - targetMean * targetMean, float3(0.0));
    float3 covariance = renderedTargetMean - renderedMean * targetMean;

    float3 A = 2.0 * renderedMean * targetMean + BACKWARD_SSIM_C1;
    float3 B = 2.0 * covariance + BACKWARD_SSIM_C2;
    float3 C = renderedMean * renderedMean + targetMean * targetMean + BACKWARD_SSIM_C1;
    float3 D = renderedVariance + targetVariance + BACKWARD_SSIM_C2;
    float3 rawSsim = (A * B) / max(C * D, float3(1e-8));
    float3 clampMask = float3(
        rawSsim.x > 0.0 && rawSsim.x < 1.0 ? 1.0 : 0.0,
        rawSsim.y > 0.0 && rawSsim.y < 1.0 ? 1.0 : 0.0,
        rawSsim.z > 0.0 && rawSsim.z < 1.0 ? 1.0 : 0.0
    );

    float normalizedSampleWeight = sampleWeight / max(weightSum, 1e-8);
    float3 dMean = float3(normalizedSampleWeight);
    float3 dRenderedSqMean = 2.0 * rendered * normalizedSampleWeight;
    float3 dRenderedTargetMean = target * normalizedSampleWeight;
    float3 dVariance = dRenderedSqMean - 2.0 * renderedMean * dMean;
    float3 dCovariance = dRenderedTargetMean - targetMean * dMean;

    float3 dA = 2.0 * targetMean * dMean;
    float3 dB = 2.0 * dCovariance;
    float3 dC = 2.0 * renderedMean * dMean;
    float3 dD = dVariance;

    return clampMask * rawSsim * (dA / max(A, float3(1e-8)) +
                                  dB / max(B, float3(1e-8)) -
                                  dC / max(C, float3(1e-8)) -
                                  dD / max(D, float3(1e-8)));
}

float sampleWeightInWindow(uint centerIndex, uint sampleIndex) {
    int2 center = int2(int(centerIndex % params.width), int(centerIndex / params.width));
    float weightSum = 0.0;
    for (int dy = -BACKWARD_SSIM_RADIUS; dy <= BACKWARD_SSIM_RADIUS; ++dy) {
        for (int dx = -BACKWARD_SSIM_RADIUS; dx <= BACKWARD_SSIM_RADIUS; ++dx) {
            if (backwardClampedPixelIndex(center + int2(dx, dy)) == sampleIndex) {
                weightSum += backwardSsimWindowWeight(int2(dx, dy));
            }
        }
    }
    return weightSum;
}

float3 dssimDerivative(uint sampleIndex) {
    int2 samplePixel = int2(int(sampleIndex % params.width), int(sampleIndex / params.width));
    int2 minCenter = max(samplePixel - int2(BACKWARD_SSIM_RADIUS), int2(0));
    int2 maxCenter = min(samplePixel + int2(BACKWARD_SSIM_RADIUS), int2(int(params.width) - 1, int(params.height) - 1));
    float3 derivative = float3(0.0);

    for (int centerY = minCenter.y; centerY <= maxCenter.y; ++centerY) {
        for (int centerX = minCenter.x; centerX <= maxCenter.x; ++centerX) {
            uint centerIndex = uint(centerY * int(params.width) + centerX);
            float sampleWeight = sampleWeightInWindow(centerIndex, sampleIndex);
            if (sampleWeight > 0.0) {
                derivative += -ssimWindowDerivativeForSample(centerIndex, sampleIndex, sampleWeight) / 3.0;
            }
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
