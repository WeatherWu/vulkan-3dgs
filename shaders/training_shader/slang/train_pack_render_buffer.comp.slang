#include "train_common.slang"

[[vk::binding(0, 0)]]
StructuredBuffer<GaussianTrainParam> gaussianParams;

[[vk::binding(10, 0)]]
RWStructuredBuffer<GaussianTrainParam> previewInstances;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint index = dispatchThreadID.x;
    if (index >= params.gaussianCount) {
        return;
    }

    previewInstances[index] = gaussianParams[index];
}
