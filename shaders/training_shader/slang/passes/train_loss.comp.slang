#include "../common/train_common.slang"
#include "../common/train_loss_common.slang"

[[vk::binding(6, 0)]]
StructuredBuffer<float4> renderedColor;

[[vk::binding(7, 0)]]
StructuredBuffer<float4> targetColor;

[[vk::binding(8, 0)]]
RWStructuredBuffer<float> lossBuffer;

static const int SSIM_RADIUS = 5;
static const float SSIM_C1 = 0.0001;
static const float SSIM_C2 = 0.0009;
static const float SSIM_SIGMA = 1.5;

float ssimWindowWeight(int2 offset) {
    float2 value = float2(offset);
    return exp(-dot(value, value) / (2.0 * SSIM_SIGMA * SSIM_SIGMA));
}

uint clampedPixelIndex(int2 pixel) {
    int width = int(params.width);
    int height = int(params.height);
    int x = clamp(pixel.x, 0, width - 1);
    int y = clamp(pixel.y, 0, height - 1);
    return uint(y * width + x);
}

float ssimColor(uint centerIndex) {
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
            uint sampleIndex = clampedPixelIndex(center + int2(dx, dy));
            float3 rendered = renderedColor[sampleIndex].xyz;
            float3 target = targetColor[sampleIndex].xyz;

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

    float3 numerator = (2.0 * renderedMean * targetMean + SSIM_C1) * (2.0 * covariance + SSIM_C2);
    float3 denominator = (renderedMean * renderedMean + targetMean * targetMean + SSIM_C1) *
                         (renderedVariance + targetVariance + SSIM_C2);
    float3 ssim = numerator / max(denominator, float3(1e-8));
    ssim = clamp(ssim, float3(0.0), float3(1.0));

    return (ssim.x + ssim.y + ssim.z) / 3.0;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint index = dispatchThreadID.x;
    if (index >= params.pixelCount) {
        return;
    }

    float l1 = l1Color(renderedColor[index].xyz, targetColor[index].xyz);
    float dssim = clamp(1.0 - ssimColor(index), 0.0, 1.0);
    float dssimWeight = clamp(params.lossDssimWeight, 0.0, 1.0);
    lossBuffer[index] = ((1.0 - dssimWeight) * l1 + dssimWeight * dssim) /
                        max(float(params.pixelCount), 1.0);
}
