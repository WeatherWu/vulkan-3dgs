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
static const int BACKWARD_SSIM_SHARED_WIDTH = 26;
static const int BACKWARD_SSIM_SHARED_PIXEL_COUNT =
    BACKWARD_SSIM_SHARED_WIDTH * BACKWARD_SSIM_SHARED_WIDTH;
static const float BACKWARD_SSIM_INV_WEIGHT_SUM = 0.0707622378;
static const float BACKWARD_SSIM_GAUSSIAN_WEIGHTS[6] = {
    1.0,
    0.8007374029,
    0.4111122905,
    0.1353352832,
    0.0285655008,
    0.0038659201
};

groupshared SsimBackwardState sharedSsimBackwardStates[BACKWARD_SSIM_SHARED_PIXEL_COUNT];

float backwardSsimWindowWeight(int2 offset) {
    return BACKWARD_SSIM_GAUSSIAN_WEIGHTS[abs(offset.x)] *
           BACKWARD_SSIM_GAUSSIAN_WEIGHTS[abs(offset.y)];
}

uint backwardSsimSharedIndex(int2 pixel) {
    return uint(pixel.y * BACKWARD_SSIM_SHARED_WIDTH + pixel.x);
}

void loadBackwardSsimNeighborhood(uint2 groupOrigin, uint localIndex) {
    for (uint sharedIndex = localIndex;
         sharedIndex < BACKWARD_SSIM_SHARED_PIXEL_COUNT;
         sharedIndex += 256u) {
        int2 sharedPixel = int2(sharedIndex % BACKWARD_SSIM_SHARED_WIDTH,
                                sharedIndex / BACKWARD_SSIM_SHARED_WIDTH);
        int2 imagePixel = int2(groupOrigin) + sharedPixel - int2(BACKWARD_SSIM_RADIUS);
        SsimBackwardState state;
        state.base = float4(0.0);
        state.targetCoefficient = float4(0.0);
        state.renderedCoefficient = float4(0.0);
        if (imagePixel.x >= 0 && imagePixel.y >= 0 &&
            imagePixel.x < int(params.width) && imagePixel.y < int(params.height)) {
            uint imageIndex = uint(imagePixel.y * int(params.width) + imagePixel.x);
            state = ssimBackwardStates[imageIndex];
        }
        sharedSsimBackwardStates[sharedIndex] = state;
    }
    GroupMemoryBarrierWithGroupSync();
}

float3 dssimDerivative(uint2 pixel, uint2 localPixel, float3 rendered, float3 target) {
    int2 samplePixel = int2(pixel);
    int2 groupOrigin = samplePixel - int2(localPixel);
    int2 minCenter = max(samplePixel - int2(BACKWARD_SSIM_RADIUS), int2(0));
    int2 maxCenter = min(samplePixel + int2(BACKWARD_SSIM_RADIUS),
                          int2(int(params.width) - 1, int(params.height) - 1));
    float3 derivative = float3(0.0);

    for (int centerY = minCenter.y; centerY <= maxCenter.y; ++centerY) {
        for (int centerX = minCenter.x; centerX <= maxCenter.x; ++centerX) {
            int2 center = int2(centerX, centerY);
            int2 centerOffset = center - groupOrigin;
            int2 offset = int2(localPixel) - centerOffset;
            int2 sharedCenter = centerOffset + int2(BACKWARD_SSIM_RADIUS);
            float sampleWeight = backwardSsimWindowWeight(offset) * BACKWARD_SSIM_INV_WEIGHT_SUM;
            SsimBackwardState state = sharedSsimBackwardStates[backwardSsimSharedIndex(sharedCenter)];
            derivative += sampleWeight *
                (state.base.xyz +
                 state.targetCoefficient.xyz * target +
                 state.renderedCoefficient.xyz * rendered);
        }
    }

    return derivative;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint2 pixel = dispatchThreadID.xy;
    uint2 localPixel = uint2(pixel.x % 16u, pixel.y % 16u);
    uint localIndex = localPixel.y * 16u + localPixel.x;
    uint2 groupOrigin = pixel - localPixel;
    bool active = pixel.x < params.width && pixel.y < params.height;
    uint index = active ? pixel.y * params.width + pixel.x : 0u;
    loadBackwardSsimNeighborhood(groupOrigin, localIndex);
    if (!active) {
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
    grad.color = float4(((1.0 - dssimWeight) * l1Grad +
                         dssimWeight * dssimDerivative(pixel, localPixel, rendered, target)) *
                        invPixelCount,
                        0.0);
    pixelGrads[index] = grad;
}
