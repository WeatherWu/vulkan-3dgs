#include "gaussian_common.slang"

[[vk::binding(0, 0)]]
ConstantBuffer<UniformBufferObject> ubo;

[[vk::binding(2, 0)]]
StructuredBuffer<GaussianInstance> instances;

[[vk::binding(3, 0)]]
StructuredBuffer<uint> sortedIndices;

struct VertexInput {
    float2 quadVertex : TEXCOORD0;
    uint instanceIndex : SV_InstanceID;
};

struct VertexOutput {
    float4 position : SV_Position;
    float3 color : TEXCOORD0;
    float alpha : TEXCOORD1;
    float2 gaussianUV : TEXCOORD2;
};

float3x3 mat3FromMat4(float4x4 value) {
    return float3x3(
        value[0][0], value[0][1], value[0][2],
        value[1][0], value[1][1], value[1][2],
        value[2][0], value[2][1], value[2][2]
    );
}

float2x2 mat2FromMat3(float3x3 value) {
    return float2x2(
        value[0][0], value[0][1],
        value[1][0], value[1][1]
    );
}

VertexOutput cullOutput() {
    VertexOutput output;
    output.position = float4(0.0, 0.0, 2.0, 1.0);
    output.color = float3(0.0, 0.0, 0.0);
    output.alpha = 0.0;
    output.gaussianUV = float2(0.0, 0.0);
    return output;
}

[shader("vertex")]
VertexOutput main(VertexInput input) {
    uint instanceIdx = sortedIndices[input.instanceIndex];
    if (instanceIdx == 0xffffffffu) {
        return cullOutput();
    }

    GaussianInstance instance = instances[instanceIdx];

    float3x3 rotationMatrix = quatToMat3(instance.rotation);
    float3x3 scaleMatrix = float3x3(
        instance.scale.x * instance.scale.x, 0.0, 0.0,
        0.0, instance.scale.y * instance.scale.y, 0.0,
        0.0, 0.0, instance.scale.z * instance.scale.z
    );

    float3x3 covariance3D = mul(mul(rotationMatrix, scaleMatrix), transpose(rotationMatrix));
    float3x3 modelMatrix3x3 = mat3FromMat4(ubo.model);
    float3x3 covarianceWorld = mul(mul(modelMatrix3x3, covariance3D), transpose(modelMatrix3x3));

    float3x3 viewMatrix3x3 = mat3FromMat4(ubo.view);
    float3 worldPosition = mul(ubo.model, float4(instance.position.xyz, 1.0)).xyz;
    float4 centerCamera = mul(ubo.view, float4(worldPosition, 1.0));
    float4 centerClip = mul(ubo.projection, centerCamera);
    float depth = centerClip.w;
    if (depth <= 0.001) {
        return cullOutput();
    }

    float3 ndcCenter3 = centerClip.xyz / centerClip.w;
    if (ndcCenter3.z < 0.0 || ndcCenter3.z > 1.0) {
        return cullOutput();
    }

    float2 screenSize = max(ubo.focal.zw, float2(1.0, 1.0));
    float3x3 covarianceCamera = mul(mul(viewMatrix3x3, covarianceWorld), transpose(viewMatrix3x3));
    float cameraZ = centerCamera.z;
    float cameraZ2 = max(cameraZ * cameraZ, 1e-12);
    float cameraRadius = max(length(centerCamera.xyz), 1e-6);
    float3x3 J = float3x3(
        -1.0 / cameraZ, 0.0, centerCamera.x / cameraZ2,
        0.0, -1.0 / cameraZ, centerCamera.y / cameraZ2,
        -2.0 * centerCamera.x / cameraRadius,
        -2.0 * centerCamera.y / cameraRadius,
        -2.0 * centerCamera.z / cameraRadius
    );

    float3x3 projectedCovariance = mul(mul(J, covarianceCamera), transpose(J));
    float2x2 projectionScale = float2x2(
        ubo.projection[0][0], ubo.projection[0][1],
        ubo.projection[1][0], ubo.projection[1][1]
    );
    float2x2 covNdc = mul(mul(projectionScale, mat2FromMat3(projectedCovariance)), projectionScale);
    covNdc[0][0] += 1.0 / (screenSize.x * screenSize.x);
    covNdc[1][1] += 1.0 / (screenSize.y * screenSize.y);

    float diagonal1 = covNdc[0][0];
    float offDiagonal = covNdc[1][0];
    float diagonal2 = covNdc[1][1];
    float D = sqrt((diagonal1 - diagonal2) * (diagonal1 - diagonal2) + 4.0 * offDiagonal * offDiagonal);
    float lambda1 = 0.5 * (diagonal1 + diagonal2 + D);
    float lambda2 = 0.5 * (diagonal1 + diagonal2 - D);
    if (lambda2 <= 0.0 || D <= 1e-12) {
        return cullOutput();
    }

    float s0 = sqrt(lambda1);
    float s1 = sqrt(lambda2);
    float maxAxis = max(s0, s1);
    if (maxAxis <= 0.0) {
        return cullOutput();
    }

    float sin2t = 2.0 * offDiagonal / D;
    float cos2t = (diagonal1 - diagonal2) / D;
    float theta = 0.5 * atan2(sin2t, cos2t);
    float cosTheta = cos(theta);
    float sinTheta = sin(theta);
    float2x2 rotScale = float2x2(
        s0 * cosTheta, -s1 * sinTheta,
        s0 * sinTheta, s1 * cosTheta
    );

    float alpha = clamp(instance.position.w, 0.0, 1.0);
    if (255.0 * alpha <= 1.0) {
        return cullOutput();
    }

    static const float CONFIDENCE_RADIUS = 3.0;
    float2 gaussianUV = input.quadVertex * CONFIDENCE_RADIUS;
    float2 ndcOffset = mul(rotScale, input.quadVertex) * CONFIDENCE_RADIUS;

    VertexOutput output;
    output.position = float4(ndcCenter3 + float3(ndcOffset, 0.0), 1.0);

    float3 modelTranslation = float3(ubo.model[0][3], ubo.model[1][3], ubo.model[2][3]);
    float3 cameraModelPosition = mul(transpose(modelMatrix3x3), ubo.cameraPosition_time.xyz - modelTranslation);
    float3 viewVector = instance.position.xyz - cameraModelPosition;
    float3 viewDirection = dot(viewVector, viewVector) > 1e-12 ? normalize(viewVector) : float3(0.0, 0.0, 0.0);

    output.color = evaluateSH(instance, viewDirection);
    output.alpha = alpha;
    output.gaussianUV = gaussianUV;
    return output;
}
