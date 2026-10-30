#include "../common/train_common.slang"
#include "../common/train_backward_types.slang"
#define NO_MAIN
#include "vertex_shader.slang"
#undef NO_MAIN
import strategy_utils;

struct NativeCameraUniform {
    float4x4 view;
    float4x4 projection;
    float4x4 model;
    float4 viewport;
    float4 focalTan;
    float4 cameraPosition;
};
[[vk::binding(11,0)]] ConstantBuffer<NativeCameraUniform> nativeCamera;
[[vk::binding(64,0)]] ConstantBuffer<float4> nativeIntrinsics;
[[vk::binding(66,0)]] RWStructuredBuffer<float> xyz_ws;
[[vk::binding(67,0)]] RWStructuredBuffer<float4> sh_coeffs;
[[vk::binding(68,0)]] RWStructuredBuffer<float4> rotations;
[[vk::binding(69,0)]] RWStructuredBuffer<float4> scales_opacs;
[[vk::binding(70,0)]] RWStructuredBuffer<float> g_xyz_ws;
[[vk::binding(71,0)]] RWStructuredBuffer<float4> g_sh_coeffs_1;
[[vk::binding(72,0)]] RWStructuredBuffer<float4> g_sh_coeffs_2;
[[vk::binding(73,0)]] RWStructuredBuffer<float4> g_rotations;
[[vk::binding(74,0)]] RWStructuredBuffer<float4> g_scales_opacs;
[[vk::binding(75,0)]] RWStructuredBuffer<uint4> nativeRectangles;
[[vk::binding(76,0)]] RWStructuredBuffer<float> nativePadding;

uint nativeSHAddress(uint g, uint block) {
    return (g / 32u) * 384u + block * 32u + g % 32u;
}

Camera nativeCameraValue() {
    return load_camera(nativeCamera.view, nativeIntrinsics.x, nativeIntrinsics.y,
                       nativeIntrinsics.z, nativeIntrinsics.w, 0, float4(0),
                       params.height, params.width);
}

Gaussian_3D nativeGaussian(uint g) {
    return {read_mean(g, xyz_ws), read_spherical_harmonics_coeffs(g, sh_coeffs, params.activeSHDegree),
            rotations[g], scales_opacs[g].xyz, scales_opacs[g].w};
}

GaussianTrainParam nativeCanonicalParam(uint g) {
    GaussianTrainParam result;
    float4 activated = scales_opacs[g];
    result.positionOpacity = float4(read_mean(g, xyz_ws),
        log(max(activated.w, 1e-8) / max(1.0 - activated.w, 1e-8)));
    result.scale = float4(log(activated.xyz), nativePadding[3u*g]);
    result.rotation = rotations[g].yzwx;
    for (uint c = 0; c < 3; ++c) for (uint block = 0; block < 4; ++block) {
        float4 v;
        for (uint lane = 0; lane < 4; ++lane) {
            uint coefficient = block * 4u + lane;
            uint scalar = coefficient * 3u + c;
            v[lane] = sh_coeffs[nativeSHAddress(g, scalar / 4u)][scalar % 4u];
        }
        result.sh[c * 4u + block] = v;
    }
    return result;
}

GaussianGrad nativeCanonicalMoment(uint g, bool second) {
    uint m = second ? 1u : 0u;
    float4 so = g_scales_opacs[2u * g + m];
    GaussianGrad result;
    result.positionOpacity = float4(read_mean(g, g_xyz_ws, 2, m), so.w);
    result.scale = float4(so.xyz, nativePadding[3u*g+(second?2u:1u)]);
    result.rotation = g_rotations[2u * g + m].yzwx;
    for (uint c = 0; c < 3; ++c) for (uint block = 0; block < 4; ++block) {
        float4 v;
        for (uint lane = 0; lane < 4; ++lane) {
            uint scalar = (block * 4u + lane) * 3u + c;
            uint address = nativeSHAddress(g, scalar / 4u);
            v[lane] = second ? g_sh_coeffs_2[address][scalar % 4u] : g_sh_coeffs_1[address][scalar % 4u];
        }
        result.sh[c * 4u + block] = v;
    }
    return result;
}

void nativeStoreMoment(uint g, GaussianGrad value, bool second) {
    uint m = second ? 1u : 0u;
    write_mean(value.positionOpacity.xyz, g, g_xyz_ws, 2, m);
    g_scales_opacs[2u * g + m] = float4(value.scale.xyz, value.positionOpacity.w);
    g_rotations[2u * g + m] = value.rotation.wxyz;
    for (uint b = 0; b < 12; ++b) {
        float4 v;
        for (uint lane = 0; lane < 4; ++lane) {
            uint scalar = b * 4u + lane, c = scalar % 3u, coefficient = scalar / 3u;
            v[lane] = value.sh[c * 4u + coefficient / 4u][coefficient % 4u];
        }
        if (second) g_sh_coeffs_2[nativeSHAddress(g,b)] = v;
        else g_sh_coeffs_1[nativeSHAddress(g,b)] = v;
    }
}
