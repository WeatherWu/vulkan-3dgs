#ifndef TRAIN_DIRECT_INTRINSICS
#define TRAIN_DIRECT_INTRINSICS 1
#endif
#if TRAIN_DIRECT_INTRINSICS
#include "../common/train_backward_project_direct_intrinsics.slang"
#else
#include "../common/train_backward_project_common.slang"
#endif
[[vk::binding(0, 0)]] StructuredBuffer<GaussianTrainParam> source;
[[vk::binding(3, 0)]] RWStructuredBuffer<ProjectedGaussian> projected;
[[vk::binding(65, 0)]] StructuredBuffer<float4> centerGrad;
[[vk::binding(66, 0)]] RWStructuredBuffer<float4> positionGrad;
[[vk::binding(67, 0)]] RWStructuredBuffer<float4> cameraCoordinates;
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 thread : SV_DispatchThreadID) {
    if (thread.x >= params.gaussianCount) return;
    uint index = thread.x;
    GaussianTrainParam gaussian = source[index];
    ProjectionResult result = trainProjectGaussian(gaussian);
    projected[index] = trainPackProjectedGaussian(result);
    ProjectionLocalState state = trainBuildProjectionLocalState(gaussian);
    cameraCoordinates[index] = state.cameraPosition;
    positionGrad[index] = result.visible
        ? float4(backwardPixelCenterToPosition(state, centerGrad[index].xy), 0.0) : float4(0.0);
}
