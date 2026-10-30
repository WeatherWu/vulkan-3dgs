#include "../vksplat/native.slang"
[[vk::binding(0,0)]] StructuredBuffer<GaussianTrainParam> canonicalParams;
[[vk::binding(2,0)]] StructuredBuffer<AdamState> canonicalMoments;
[shader("compute")]
[numthreads(256,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint g=tid.x;
    if (g>=params.gaussianCount) return;
    GaussianTrainParam value=canonicalParams[g];
    write_mean(value.positionOpacity.xyz,g,xyz_ws);
    rotations[g]=value.rotation.wxyz;
    nativePadding[3u*g]=value.scale.w;
    nativePadding[3u*g+1u]=canonicalMoments[g].firstMoment.scale.w;
    nativePadding[3u*g+2u]=canonicalMoments[g].secondMoment.scale.w;
    scales_opacs[g]=float4(exp(value.scale.xyz),1.0/(1.0+exp(-value.positionOpacity.w)));
    for (uint b=0;b<12;++b) {
        float4 v;
        for (uint lane=0;lane<4;++lane) {
            uint scalar=b*4+lane,c=scalar%3,coefficient=scalar/3;
            v[lane]=value.sh[c*4+coefficient/4][coefficient%4];
        }
        sh_coeffs[nativeSHAddress(g,b)]=v;
    }
    nativeStoreMoment(g,canonicalMoments[g].firstMoment,false);
    nativeStoreMoment(g,canonicalMoments[g].secondMoment,true);
}
