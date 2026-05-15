#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 1) in float fragAlpha;
layout(location = 2) flat in vec2 fragScreenPos;      // 像素空间中心
layout(location = 3) flat in mat2 fragCovariance2D;   // 像素空间2D协方差矩阵（占用location 3和4）
layout(location = 5) flat in float fragMaxAxis;   // 最大轴长（使用flat避免插值问题）
layout(location = 6) in vec2 fragGaussianUV;      // 椭圆局部坐标

layout(location = 0) out vec4 outColor;

void main() {
    float gaussianAlpha = exp(-0.5 * dot(fragGaussianUV, fragGaussianUV));
    float finalAlpha = clamp(fragAlpha, 0.0, 1.0) * gaussianAlpha;
    
    outColor = vec4(fragColor, finalAlpha);
}
