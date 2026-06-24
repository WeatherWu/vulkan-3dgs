#include "../common/train_forward_project_common.slang"

[[vk::binding(0, 0)]]
StructuredBuffer<GaussianTrainParam> gaussianParams;

[[vk::binding(3, 0)]]
RWStructuredBuffer<ProjectedGaussian> projectedGaussians;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint index = dispatchThreadID.x;
    if (index >= params.gaussianCount) {
        return;
    }

    ProjectionResult projection = trainProjectGaussian(gaussianParams[index]);
    projectedGaussians[index] = trainPackProjectedGaussian(projection);
}
