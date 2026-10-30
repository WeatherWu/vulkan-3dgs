#include "../vksplat/native.slang"
[[vk::binding(0,0)]] RWStructuredBuffer<GaussianTrainParam> canonicalParams;
[[vk::binding(2,0)]] RWStructuredBuffer<AdamState> canonicalMoments;
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
}
