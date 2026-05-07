#version 450

// 预定义的四边形顶点（-1到1的标准正方形）
layout(location = 0) in vec2 quadVertex;  // (-1,-1), (1,-1), (-1,1), (1,1)

// Uniform Buffer
layout(binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 projection;
    vec4 cameraPosition_time; // xyz: camera position, w: time
    vec4 focal;               // xy: pixel focal lengths, zw: screen size
} ubo;

// SSBO: 存储所有高斯点的实例数据（包含完整SH系数）
struct GaussianInstance {
    vec4 position;      // xyz: 位置
    vec4 scale;         // xyz: 缩放
    vec4 rotation;      // xyzw: 四元数
    
    // 球谐函数系数（完整3阶SH）
    vec4 sh0_alpha;     // xyz: 0阶SH - DC项, w: alpha
    vec4 sh1[3];        // xyz: 1阶SH - 3个基函数
    vec4 sh2[5];        // xyz: 2阶SH - 5个基函数
    vec4 sh3[7];        // xyz: 3阶SH - 7个基函数
    
    // 全部使用vec4，避免CPU侧glm::vec3与GLSL std430 vec3/vec3数组步长不一致。
};

layout(std430, binding = 2) readonly buffer GaussianInstances {
    GaussianInstance instances[];
};

// GPU深度排序后的实例索引。绘制实例号先查这里，再读取Gaussian数据。
layout(std430, binding = 3) readonly buffer SortedIndices {
    uint sortedIndices[];
};

// 输出到片段着色器
layout(location = 0) out vec3 fragColor;
layout(location = 1) out float fragAlpha;
layout(location = 2) flat out vec2 fragScreenPos;   // 像素空间中心
layout(location = 3) flat out mat2 fragCovariance2D; // 像素空间2D协方差矩阵
layout(location = 5) out float fragMaxAxis;      // 最大轴长
layout(location = 6) out vec2 fragGaussianUV;    // 椭圆局部坐标，用于SuperSplat式alpha衰减

// 从四元数构建旋转矩阵
mat3 quatToMat3(vec4 q) {
    return mat3(
        1.0 - 2.0 * q.y * q.y - 2.0 * q.z * q.z,
        2.0 * q.x * q.y + 2.0 * q.z * q.w,
        2.0 * q.x * q.z - 2.0 * q.y * q.w,
        
        2.0 * q.x * q.y - 2.0 * q.z * q.w,
        1.0 - 2.0 * q.x * q.x - 2.0 * q.z * q.z,
        2.0 * q.y * q.z + 2.0 * q.x * q.w,
        
        2.0 * q.x * q.z + 2.0 * q.y * q.w,
        2.0 * q.y * q.z - 2.0 * q.x * q.w,
        1.0 - 2.0 * q.x * q.x - 2.0 * q.y * q.y
    );
}

// 球谐函数评估（3阶SH，共16个基函数）
vec3 restore3DGSColor(vec3 sh_color) {
    // 3DGS stores SH coefficients in a color space restored by adding 0.5
    // after SH evaluation. Only negative values are clamped.
    return max(sh_color + vec3(0.5), vec3(0.0));
}

vec3 evaluateSH(const GaussianInstance instance, vec3 view_direction) {
    const float SH_C0 = 0.28209479177387814f;
    const float SH_C1 = 0.4886025119029199f;
    const float SH_C2[5] = float[](
        1.0925484305920792f,
        -1.0925484305920792f,
        0.31539156525252005f,
        -1.0925484305920792f,
        0.5462742152960396f
    );
    const float SH_C3[7] = float[](
        -0.5900435899266435f,
        2.890611442640554f,
        -0.4570457994644658f,
        0.3731763325901154f,
        -0.4570457994644658f,
        1.445305721320277f,
        -0.5900435899266435f
    );
    
    // 0阶SH (DC项)
    vec3 dc = instance.sh0_alpha.xyz * SH_C0;
    vec3 result = dc;
    
    if (view_direction.x == 0.0f && view_direction.y == 0.0f && view_direction.z == 0.0f) {
        return restore3DGSColor(result);
    }
    
    float x = view_direction.x;
    float y = view_direction.y;
    float z = view_direction.z;
    
    // 1阶SH (l=1, m=-1,0,1)
    result += SH_C1 * (-y * instance.sh1[0].xyz + z * instance.sh1[1].xyz - x * instance.sh1[2].xyz);
    
    // 2阶SH (l=2, m=-2,-1,0,1,2)
    float xx = x * x, yy = y * y, zz = z * z;
    float xy = x * y, yz = y * z, xz = x * z;
    result += 
        SH_C2[0] * xy * instance.sh2[0].xyz +
        SH_C2[1] * yz * instance.sh2[1].xyz +
        SH_C2[2] * (2.0f * zz - xx - yy) * instance.sh2[2].xyz +
        SH_C2[3] * xz * instance.sh2[3].xyz +
        SH_C2[4] * (xx - yy) * instance.sh2[4].xyz;
    
    // 3阶SH (l=3, m=-3,-2,-1,0,1,2,3)
    result +=
        SH_C3[0] * y * (3.0f * xx - yy) * instance.sh3[0].xyz +
        SH_C3[1] * xy * z * instance.sh3[1].xyz +
        SH_C3[2] * y * (4.0f * zz - xx - yy) * instance.sh3[2].xyz +
        SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * instance.sh3[3].xyz +
        SH_C3[4] * x * (4.0f * zz - xx - yy) * instance.sh3[4].xyz +
        SH_C3[5] * z * (xx - yy) * instance.sh3[5].xyz +
        SH_C3[6] * x * (xx - 3.0f * yy) * instance.sh3[6].xyz;
    
    return restore3DGSColor(result);
}

void main() {
    // 通过排序索引间接读取实例数据，使GPU排序结果真正决定绘制顺序。
    uint instanceIdx = sortedIndices[gl_InstanceIndex];
    if (instanceIdx == 0xFFFFFFFFu) {
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        return;
    }
    GaussianInstance instance = instances[instanceIdx];
    
    // 1. 构建3D协方差矩阵 Σ = R × S × S^T × R^T
    mat3 rotationMatrix = quatToMat3(instance.rotation);
    mat3 scaleMatrix = mat3(
        instance.scale.x * instance.scale.x, 0.0, 0.0,
        0.0, instance.scale.y * instance.scale.y, 0.0,
        0.0, 0.0, instance.scale.z * instance.scale.z
    );
    
    mat3 covariance3D = rotationMatrix * scaleMatrix * transpose(rotationMatrix);
    
    // 2. 将3D协方差变换到相机空间（仅旋转部分，不含平移）
    mat3 viewMatrix3x3 = mat3(ubo.view);
    
    // 将高斯中心变换到相机空间
    vec4 centerCamera = ubo.view * vec4(instance.position.xyz, 1.0);
    
    // 计算投影后的深度
    vec4 centerClip = ubo.projection * centerCamera;
    float depth = centerClip.w;
    
    // 避免除零
    if (depth <= 0.001) {
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0); // 剔除近裁剪面内的点
        return;
    }
    
    // 3. Project world covariance to screen-space pixels. This is the same
    // 3DGS Jacobian path as PlayCanvas, written explicitly for GLSL's
    // column-vector convention to avoid row/column transposition mistakes.
    mat3 covarianceCamera = viewMatrix3x3 * covariance3D * transpose(viewMatrix3x3);
    float z = max(depth, 1e-6);
    float fx = ubo.focal.x;
    float fy = ubo.focal.y;

    mat3x2 J = mat3x2(
        fx / z,                         0.0,
        0.0,                            fy / z,
        fx * centerCamera.x / (z * z),  fy * centerCamera.y / (z * z)
    );

    mat2 cov = J * covarianceCamera * transpose(J);
    float covA = cov[0][0];
    float covB = 0.5 * (cov[0][1] + cov[1][0]);
    float covC = cov[1][1];

    // Low-pass blur from the 3DGS rasterization reference path.
    float detOrig = max(covA * covC - covB * covB, 0.0);
    float diagonal1 = covA + 0.3;
    float offDiagonal = covB;
    float diagonal2 = covC + 0.3;
    float detBlur = max(diagonal1 * diagonal2 - offDiagonal * offDiagonal, 1e-12);
    mat2 covariance2D = mat2(diagonal1, offDiagonal, offDiagonal, diagonal2);
    
    // 6. 特征分解获取椭圆的轴和方向
    float mid = 0.5 * (diagonal1 + diagonal2);
    float radius = length(vec2((diagonal1 - diagonal2) * 0.5, offDiagonal));
    float lambda1 = mid + radius;
    float lambda2 = mid - radius;
    if (lambda2 <= 0.0) {
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        return;
    }

    vec2 screenSize = max(ubo.focal.zw, vec2(1.0));
    const float SQRT8 = 2.8284271247461903;
    float vmin = min(1024.0, min(screenSize.x, screenSize.y));
    float l1 = min(SQRT8 * sqrt(lambda1), vmin);
    float l2 = min(SQRT8 * sqrt(lambda2), vmin);

    float maxAxis = max(l1, l2);

    // SuperSplat keeps sub-2px splats and relies on AA compensation plus the
    // alpha contribution cutoff. A hard 2px cull removes fine fur/hair splats.
    if (maxAxis <= 0.0) {
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        return;
    }

    vec2 diagonalVector = vec2(offDiagonal, lambda1 - diagonal1);
    diagonalVector = dot(diagonalVector, diagonalVector) > 1e-8
        ? normalize(diagonalVector)
        : vec2(1.0, 0.0);
    vec2 majorAxis = l1 * diagonalVector;
    vec2 minorAxis = l2 * vec2(diagonalVector.y, -diagonalVector.x);

    float alpha = clamp(instance.sh0_alpha.w, 0.0, 1.0);
    if (255.0 * alpha <= 1.0) {
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        return;
    }

    vec2 gaussianUV = quadVertex * SQRT8;
    vec2 clipOffset = (quadVertex.x * majorAxis + quadVertex.y * minorAxis) * (2.0 * centerClip.ww) / screenSize;
    
    // 9. 计算最终位置（保持在裁剪空间）
    // 将NDC空间的偏移量转换回裁剪空间
    vec2 ndcCenter = centerClip.xy / centerClip.w;
    gl_Position = centerClip + vec4(clipOffset, 0.0, 0.0);
    vec2 pixelCenter = (ndcCenter * 0.5 + 0.5) * screenSize;
    
    // 10. 计算视角相关的颜色（使用完整SH评估）
    // 标准3DGS使用世界空间中从相机指向高斯中心的方向。
    vec3 viewVector = instance.position.xyz - ubo.cameraPosition_time.xyz;
    vec3 viewDirection = dot(viewVector, viewVector) > 1e-12 ? normalize(viewVector) : vec3(0.0);
    fragColor = evaluateSH(instance, viewDirection);
    
    // 11. 传递其他数据到片段着色器
    fragAlpha = alpha;
    fragScreenPos = pixelCenter;
    fragCovariance2D = covariance2D;
    fragMaxAxis = 0.5 * max(l1, l2);
    fragGaussianUV = gaussianUV;
}
