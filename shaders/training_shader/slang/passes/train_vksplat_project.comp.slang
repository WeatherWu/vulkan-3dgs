#include "../vksplat/native.slang"
[[vk::binding(3,0)]] RWStructuredBuffer<ProjectedGaussian> projectedGaussians;
[[vk::binding(32,0)]] RWStructuredBuffer<uint2> gaussianTileRanges;
[[vk::binding(16,0)]] RWStructuredBuffer<GaussianDensificationState> densificationStates;

[shader("compute")]
[numthreads(32,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint g = tid.x;
    if (g >= params.gaussianCount) return;
    ProjectedGaussian output;
    output.centerRadius = output.conicOpacity = output.color = float4(0);
    projectedGaussians[g] = output;
    gaussianTileRanges[g] = uint2(0);
    nativeRectangles[g] = uint4(0);
    Gaussian_3D gaussian = nativeGaussian(g);
    Splat_2D_Vertex splat = project_gaussian_to_camera(gaussian, nativeCameraValue(), params.activeSHDegree);
    if (splat.xyz_vs.z <= NEAR_CLIP || splat.inv_cov_vs_opac.w <= ALPHA_THRESHOLD || splat.det <= 0.0) return;
    float extend = extend_from_opacity(gaussian.opacities);
    float2 radii = sqrt(splat.cov_vs.xz) * extend;
    uint gridWidth = (params.width + 15u) / 16u, gridHeight = (params.height + 15u) / 16u;
    rectangle rect = get_rectangle_tile_space(splat.xyz_vs.xy, radii, gridHeight, gridWidth);
    uint count = count_ellipse_grid_overlaps(splat.xyz_vs.xy, splat.inv_cov_vs_opac.xyz / (extend*extend), rect);
    if (!count) return;
    nativeRectangles[g] = uint4(rect.min_x, rect.min_y, rect.max_x, rect.max_y);
    gaussianTileRanges[g] = uint2(0,count);
    output.centerRadius = float4(splat.xyz_vs, ceil(max(radii.x,radii.y)));
    output.conicOpacity = splat.inv_cov_vs_opac;
    output.color = float4(splat.rgb,1);
    projectedGaussians[g] = output;
    GaussianDensificationState state = densificationStates[g];
    state.maxScreenRadius = max(state.maxScreenRadius, output.centerRadius.w);
    densificationStates[g] = state;
}
