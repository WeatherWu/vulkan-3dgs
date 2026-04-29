#version 450

// 预定义的四边形顶点（-1到1的标准正方形）
layout(location = 0) in vec2 quadVertex;  // (-1,-1), (1,-1), (-1,1), (1,1)

// 实例数据（每个高斯点的3D参数）
layout(location = 1) in vec3 instancePosition;
layout(location = 2) in vec3 instanceScale;
layout(location = 3) in vec4 instanceRotation; // 四元数 (x, y, z, w)
layout(location = 4) in vec3 instanceColor;
layout(location = 5) in float instanceAlpha;

// Uniform Buffer
layout(binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 projection;
    vec3 cameraPosition;
    float time;
} ubo;

// 输出到片段着色器
layout(location = 0) out vec3 fragColor;
layout(location = 1) out float fragAlpha;
layout(location = 2) out vec2 fragScreenPos;   // 屏幕空间中心
layout(location = 3) out mat2 fragCovariance2D; // 2D协方差矩阵
layout(location = 5) out float fragMaxAxis;      // 最大轴长

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

void main() {
    // 1. 构建3D协方差矩阵 Σ = R × S × S^T × R^T
    mat3 rotationMatrix = quatToMat3(instanceRotation);
    mat3 scaleMatrix = mat3(
        instanceScale.x, 0.0, 0.0,
        0.0, instanceScale.y, 0.0,
        0.0, 0.0, instanceScale.z
    );
    
    mat3 covariance3D = rotationMatrix * scaleMatrix * transpose(rotationMatrix);
    
    // 2. 将3D协方差变换到相机空间（仅旋转部分，不含平移）
    mat3 viewMatrix3x3 = mat3(ubo.view);
    
    // 将高斯中心变换到相机空间
    vec4 centerCamera = ubo.view * vec4(instancePosition, 1.0);
    
    // 计算投影后的深度
    vec4 centerClip = ubo.projection * centerCamera;
    float depth = centerClip.w;
    
    // 避免除零
    if (depth <= 0.001) {
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0); // 剔除近裁剪面内的点
        return;
    }
    
    // 3. 计算从相机空间到NDC的雅可比矩阵 J
    // J = d(NDC)/d(Camera) ≈ [f_x/z, 0; 0, f_y/z]（透视投影近似）
    float fx = ubo.projection[0][0]; // 水平焦距
    float fy = ubo.projection[1][1]; // 垂直焦距
    float invZ = 1.0 / depth;        // 深度倒数
    
    mat2 J = mat2(
        fx * invZ, 0.0,
        0.0, fy * invZ
    );
    
    // 4. 将3D协方差先变换到相机空间
    mat3 covarianceCamera = viewMatrix3x3 * covariance3D * transpose(viewMatrix3x3);
    
    // 5. 通过雅可比矩阵投影到2D：Σ_2D = J × Σ_camera(xy) × J^T
    // 提取3D协方差的xy部分（忽略z轴）
    mat2 covariance3D_xy = mat2(
        covarianceCamera[0][0], covarianceCamera[0][1],
        covarianceCamera[1][0], covarianceCamera[1][1]
    );
    
    mat2 covariance2D = J * covariance3D_xy * transpose(J);
    
    // 6. 特征分解获取椭圆的轴和方向
    float a = covariance2D[0][0];
    float b = covariance2D[0][1];
    float c = covariance2D[1][1];
    
    float trace = a + c;
    float det = a * c - b * b;
    float discriminant = trace * trace - 4.0 * det;
    
    vec2 eigenvalues;
    mat2 eigenvectors;
    
    if (discriminant < 0.0) {
        eigenvalues = vec2(a, c);
        eigenvectors = mat2(1.0, 0.0, 0.0, 1.0);
    } else {
        float sqrtDisc = sqrt(discriminant);
        eigenvalues.x = (trace + sqrtDisc) * 0.5;
        eigenvalues.y = (trace - sqrtDisc) * 0.5;
        
        if (abs(b) > 1e-6) {
            float theta = 0.5 * atan(2.0 * b, (a - c));
            float cosTheta = cos(theta);
            float sinTheta = sin(theta);
            eigenvectors = mat2(cosTheta, -sinTheta, sinTheta, cosTheta);
        } else {
            eigenvectors = mat2(1.0, 0.0, 0.0, 1.0);
        }
    }
    
    // 7. 计算轴长（3σ范围）
    vec2 axes = 3.0 * sqrt(max(eigenvalues, vec2(1e-6)));
    
    // 8. 变换四边形顶点到椭圆包围盒
    vec2 localOffset = quadVertex * axes;
    vec2 worldOffset = eigenvectors * localOffset;
    
    // 9. 计算最终位置
    vec2 screenCenter = centerClip.xy / centerClip.w;
    gl_Position = vec4(screenCenter + worldOffset, centerClip.z / centerClip.w, 1.0);
    
    // 10. 传递数据到片段着色器
    fragColor = instanceColor;
    fragAlpha = instanceAlpha;
    fragScreenPos = screenCenter;
    fragCovariance2D = covariance2D;
    fragMaxAxis = max(axes.x, axes.y);
}
