#ifndef VKSPLAT_PROJECTION_SOA
#define VKSPLAT_PROJECTION_SOA
[[vk::binding(80,0)]] RWStructuredBuffer<float2> soaXY;
[[vk::binding(81,0)]] RWStructuredBuffer<float> soaDepth;
[[vk::binding(82,0)]] RWStructuredBuffer<float4> soaConic;
[[vk::binding(83,0)]] RWStructuredBuffer<float> soaRGB;
[[vk::binding(84,0)]] RWStructuredBuffer<uint2> soaRect;
[[vk::binding(85,0)]] RWStructuredBuffer<int> soaRadius;
[[vk::binding(86,0)]] RWStructuredBuffer<uint> soaCount;
ProjectedGaussian soaRasterGaussian(uint g) {
    ProjectedGaussian p;
    p.centerRadius=float4(soaXY[g],0,0);
    p.conicOpacity=soaConic[g];
    p.color=float4(soaRGB[3*g],soaRGB[3*g+1],soaRGB[3*g+2],1);
    return p;
}
ProjectedGaussian soaGaussian(uint g) {
    ProjectedGaussian p;
    p.centerRadius=float4(soaXY[g],soaDepth[g],float(soaRadius[g]));
    p.conicOpacity=soaConic[g];
    p.color=float4(soaRGB[3*g],soaRGB[3*g+1],soaRGB[3*g+2],1);
    return p;
}
#endif
