#include "../common/train_common.slang"

[[vk::binding(1, 0)]]
RWStructuredBuffer<GaussianGrad> gaussianGrads;

[[vk::binding(6, 0)]]
RWStructuredBuffer<float4> renderedColor;

[[vk::binding(8, 0)]]
RWStructuredBuffer<float> lossBuffer;

[[vk::binding(9, 0)]]
RWStructuredBuffer<uint4> counters;

[[vk::binding(14, 0)]]
RWStructuredBuffer<PixelBlendState> pixelBlendStates;

[[vk::binding(15, 0)]]
RWStructuredBuffer<GaussianVisibilityState> gaussianVisibility;

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

        GaussianVisibilityState visibility;
        visibility.contributionCount = 0;
        visibility.alphaSum = 0.0;
        clearVisibilityMaxScreenRadius(visibility);
        visibility.padding0 = 0;
        gaussianVisibility[index] = visibility;
    }

    if (index < params.pixelCount) {
        renderedColor[index] = float4(0.0, 0.0, 0.0, 1.0);
        PixelBlendState blendState;
        blendState.finalTransmittance = 1.0;
        blendState.accumulatedAlpha = 0.0;
        blendState.processedCount = 0;
        blendState.lastContributor = 0xFFFFFFFFu;
        pixelBlendStates[index] = blendState;
    }

    if (index == 0) {
        lossBuffer[0] = 0.0;
        counters[0] = uint4(0, 0, 0, 0);
    }
}
