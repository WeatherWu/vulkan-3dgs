#include "../vksplat/native.slang"
[[vk::binding(0,0)]] RWStructuredBuffer<GaussianTrainParam> canonicalParams;
[[vk::binding(2,0)]] RWStructuredBuffer<AdamState> canonicalMoments;
[[vk::binding(16,0)]] RWStructuredBuffer<GaussianDensificationState> canonicalStatistics;
[[vk::binding(78,0)]] RWStructuredBuffer<float2> nativeRunningGrad;
[[vk::binding(79,0)]] RWStructuredBuffer<float> nativeRunningRadius;
[shader("compute")]
[numthreads(256,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint g=tid.x;
    if (g>=params.gaussianCount) return;
    canonicalParams[g]=nativeCanonicalParam(g);
    AdamState state;
    state.firstMoment=nativeCanonicalMoment(g,false);
    state.secondMoment=nativeCanonicalMoment(g,true);
    canonicalMoments[g]=state;
    GaussianDensificationState statistics;
    statistics.screenGradSum = nativeRunningGrad[g].x;
    statistics.screenGradCount = uint(nativeRunningGrad[g].y);
    statistics.maxScreenRadius = nativeRunningRadius[g];
    statistics.padding0 = 0;
    canonicalStatistics[g] = statistics;
}
