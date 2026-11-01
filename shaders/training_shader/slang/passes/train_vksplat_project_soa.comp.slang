#include "../vksplat/native.slang"
#include "../vksplat/projection_soa.slang"
[shader("compute")]
[numthreads(32,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint g = tid.x;
    if (g >= params.gaussianCount) return;
    soaCount[g]=0;
    soaRect[g]=uint2(0);
    soaRadius[g]=0;
    Gaussian_3D gaussian = nativeGaussian(g);
    Splat_2D_Vertex splat = project_gaussian_to_camera(gaussian, nativeCameraValue(), params.activeSHDegree);
    if (splat.xyz_vs.z <= NEAR_CLIP || splat.inv_cov_vs_opac.w <= ALPHA_THRESHOLD || splat.det <= 0.0) return;
    float extend = extend_from_opacity(gaussian.opacities);
    float2 radii = sqrt(splat.cov_vs.xz) * extend;
    uint gridWidth = (params.width + 15u) / 16u, gridHeight = (params.height + 15u) / 16u;
    rectangle rect = get_rectangle_tile_space(splat.xyz_vs.xy, radii, gridHeight, gridWidth);
    if (rect.min_x == rect.max_x || rect.min_y == rect.max_y) return;
    soaRadius[g] = int(ceil(max(radii.x,radii.y)));
    uint count = count_ellipse_grid_overlaps(splat.xyz_vs.xy, splat.inv_cov_vs_opac.xyz / (extend*extend), rect);
    if (!count) return;
    soaRect[g] = uint2(uint(rect.min_x) | (uint(rect.min_y)<<16), uint(rect.max_x) | (uint(rect.max_y)<<16));
    soaCount[g] = count;
    soaXY[g]=splat.xyz_vs.xy;
    soaDepth[g]=splat.xyz_vs.z;
    soaConic[g]=splat.inv_cov_vs_opac;
    soaRGB[3*g]=splat.rgb.x; soaRGB[3*g+1]=splat.rgb.y; soaRGB[3*g+2]=splat.rgb.z;
}
