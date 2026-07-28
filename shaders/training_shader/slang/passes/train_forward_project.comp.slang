#include "../common/train_forward_project_common.slang"

[[vk::binding(0, 0)]]
StructuredBuffer<GaussianTrainParam> gaussianParams;

[[vk::binding(3, 0)]]
RWStructuredBuffer<ProjectedGaussian> projectedGaussians;

[[vk::binding(16, 0)]]
RWStructuredBuffer<GaussianDensificationState> densificationStates;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint index = dispatchThreadID.x;
    if (index >= params.gaussianCount) {
        return;
    }

    ProjectionResult projection = trainProjectGaussian(gaussianParams[index]);
    projectedGaussians[index] = trainPackProjectedGaussian(projection);
    if (projection.visible) {
        GaussianDensificationState densification = densificationStates[index];
        densification.maxScreenRadius = max(densification.maxScreenRadius, projection.radius);
        densificationStates[index] = densification;
    }
}
