#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 1) in float fragAlpha;
layout(location = 2) in vec2 fragScreenPos;      // 屏幕空间中心（NDC）
layout(location = 3) in mat2 fragCovariance2D;   // 2D协方差矩阵（占用location 3和4）
layout(location = 5) flat in float fragMaxAxis;   // 最大轴长（使用flat避免插值问题）

layout(location = 0) out vec4 outColor;

// Uniform: 屏幕分辨率
layout(binding = 1) uniform ScreenInfo {
    vec2 screenSize; // width, height
} screenInfo;

// 计算2D高斯分布的概率密度函数
float gaussianPDF(vec2 x, vec2 mean, mat2 covariance) {
    vec2 diff = x - mean;
    
    // 计算协方差矩阵的逆
    float det = covariance[0][0] * covariance[1][1] - covariance[0][1] * covariance[1][0];
    if (det <= 0.0) return 0.0;
    
    mat2 invCov = mat2(
        covariance[1][1], -covariance[0][1],
        -covariance[1][0], covariance[0][0]
    ) / det;
    
    // 马氏距离平方: d^2 = (x-μ)^T × Σ^-1 × (x-μ)
    vec2 temp = invCov * diff;
    float mahalanobisDistSq = dot(diff, temp);
    
    // 高斯PDF: exp(-0.5 * d^2)
    return exp(-0.5 * mahalanobisDistSq);
}

void main() {
    // 当前片段的屏幕空间位置（像素坐标转NDC）
    vec2 pixelCoord = gl_FragCoord.xy;
    vec2 ndcCoord = (pixelCoord / screenInfo.screenSize) * 2.0 - 1.0;
    
    // 计算高斯分布值
    float pdf = gaussianPDF(ndcCoord, fragScreenPos, fragCovariance2D);
    
    // 阈值裁剪（避免绘制尾部噪声）
    if (pdf < 0.01) {
        discard;
    }
    
    // Alpha混合：原始alpha × 高斯权重
    float finalAlpha = fragAlpha * pdf;
    
    if (finalAlpha < 0.01) {
        discard;
    }
    
    outColor = vec4(fragColor, finalAlpha);
}
