#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 1) in float fragAlpha;
layout(location = 2) flat in vec2 fragScreenPos;      // 像素空间中心
layout(location = 3) flat in mat2 fragCovariance2D;   // 像素空间2D协方差矩阵（占用location 3和4）
layout(location = 5) flat in float fragMaxAxis;   // 最大轴长（使用flat避免插值问题）
layout(location = 6) in vec2 fragGaussianUV;      // 椭圆局部坐标

layout(location = 0) out vec4 outColor;

// Uniform: 屏幕分辨率
layout(binding = 1) uniform ScreenInfo {
    vec2 screenSize; // width, height
} screenInfo;

vec3 srgbToLinear(vec3 color) {
    color = max(color, vec3(0.0));
    bvec3 cutoff = lessThanEqual(color, vec3(0.04045));
    vec3 lower = color / 12.92;
    vec3 higher = pow((color + vec3(0.055)) / 1.055, vec3(2.4));
    return mix(higher, lower, cutoff);
}

float gaussianAlphaWeight(vec2 uv) {
    float A = dot(uv, uv);
    if (A > 8.0) {
        return 0.0;
    }

    return exp(-0.5 * A);
}

void main() {
    const float MIN_ALPHA = 1.0 / 255.0;

    float weight = gaussianAlphaWeight(fragGaussianUV);
    if (weight <= 0.0) {
        discard;
    }
    
    // Alpha混合：原始alpha × SuperSplat式归一化高斯权重。
    float finalAlpha = clamp(fragAlpha, 0.0, 1.0) * weight;
    
    if (finalAlpha < MIN_ALPHA) {
        discard;
    }
    
    vec3 linearColor = srgbToLinear(fragColor);
    outColor = vec4(linearColor * finalAlpha, finalAlpha);
}
