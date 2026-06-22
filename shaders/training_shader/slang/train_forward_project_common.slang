#include "train_common.slang"

struct TrainingForwardCamera {
    float4x4 view;
    float4x4 projection;
    float4x4 model;
    float4 viewport;
    float4 cameraPosition;
};

struct ProjectionResult {
    bool visible;
    float2 center;
    float depth;
    float radius;
    float4 conicOpacity;
    float4 color;
};

[[vk::binding(11, 0)]]
ConstantBuffer<TrainingForwardCamera> trainingCamera;

float3x3 trainMat3FromMat4(float4x4 value) {
    return float3x3(
        value[0][0], value[0][1], value[0][2],
        value[1][0], value[1][1], value[1][2],
        value[2][0], value[2][1], value[2][2]
    );
}

float2x2 trainMat2FromMat3(float3x3 value) {
    return float2x2(
        value[0][0], value[0][1],
        value[1][0], value[1][1]
    );
}

float3x3 trainQuatToMat3(float4 q) {
    return float3x3(
        1.0 - 2.0 * q.y * q.y - 2.0 * q.z * q.z,
        2.0 * q.x * q.y - 2.0 * q.z * q.w,
        2.0 * q.x * q.z + 2.0 * q.y * q.w,

        2.0 * q.x * q.y + 2.0 * q.z * q.w,
        1.0 - 2.0 * q.x * q.x - 2.0 * q.z * q.z,
        2.0 * q.y * q.z - 2.0 * q.x * q.w,

        2.0 * q.x * q.z - 2.0 * q.y * q.w,
        2.0 * q.y * q.z + 2.0 * q.x * q.w,
        1.0 - 2.0 * q.x * q.x - 2.0 * q.y * q.y
    );
}

float3x3 trainCovariance3D(GaussianTrainParam gaussian) {
    float3 scale = max(abs(gaussian.scale.xyz), float3(1e-6));
    float3x3 rotation = trainQuatToMat3(normalize(gaussian.rotation));
    float3x3 scaleMatrix = float3x3(
        scale.x * scale.x, 0.0, 0.0,
        0.0, scale.y * scale.y, 0.0,
        0.0, 0.0, scale.z * scale.z
    );
    return mul(mul(rotation, scaleMatrix), transpose(rotation));
}

float3x3 trainProjectionJacobian(float3 cameraPosition) {
    float cameraZ = cameraPosition.z;
    float cameraZ2 = max(cameraZ * cameraZ, 1e-12);
    float cameraRadius = max(length(cameraPosition), 1e-6);
    return float3x3(
        -1.0 / cameraZ, 0.0, cameraPosition.x / cameraZ2,
        0.0, -1.0 / cameraZ, cameraPosition.y / cameraZ2,
        -2.0 * cameraPosition.x / cameraRadius,
        -2.0 * cameraPosition.y / cameraRadius,
        -2.0 * cameraPosition.z / cameraRadius
    );
}

float2x2 trainCovariance2DPixel(float3 cameraPosition, float3x3 covarianceWorld) {
    float3x3 viewMatrix = trainMat3FromMat4(trainingCamera.view);
    float3x3 covarianceCamera = mul(mul(viewMatrix, covarianceWorld), transpose(viewMatrix));
    float3x3 jacobian = trainProjectionJacobian(cameraPosition);
    float3x3 projectedCovariance = mul(mul(jacobian, covarianceCamera), transpose(jacobian));

    float2x2 projectionScale = float2x2(
        trainingCamera.projection[0][0], trainingCamera.projection[0][1],
        trainingCamera.projection[1][0], trainingCamera.projection[1][1]
    );
    float2x2 covNdc = mul(mul(projectionScale, trainMat2FromMat3(projectedCovariance)), projectionScale);

    float2 screenSize = max(trainingCamera.viewport.zw, float2(1.0));
    covNdc[0][0] += 1.0 / (screenSize.x * screenSize.x);
    covNdc[1][1] += 1.0 / (screenSize.y * screenSize.y);

    float2x2 ndcToPixel = float2x2(
        0.5 * screenSize.x, 0.0,
        0.0, 0.5 * screenSize.y
    );
    return mul(mul(ndcToPixel, covNdc), ndcToPixel);
}

float4 trainConicOpacity(float2x2 covariancePixel, float opacity) {
    float a = covariancePixel[0][0];
    float b = covariancePixel[0][1];
    float c = covariancePixel[1][1];
    float determinant = max(a * c - b * b, 1e-12);
    return float4(c / determinant, -b / determinant, a / determinant, clamp(opacity, 0.0, 1.0));
}

float trainGaussianRadius(float2x2 covariancePixel) {
    float a = covariancePixel[0][0];
    float b = covariancePixel[0][1];
    float c = covariancePixel[1][1];
    float delta = sqrt(max((a - c) * (a - c) + 4.0 * b * b, 0.0));
    float lambdaMax = max(0.5 * (a + c + delta), 0.0);
    return ceil(3.0 * sqrt(lambdaMax));
}

float2 trainNdcToPixel(float2 ndc) {
    float2 screenSize = trainingCamera.viewport.zw;
    return float2(
        (ndc.x * 0.5 + 0.5) * screenSize.x,
        (0.5 - ndc.y * 0.5) * screenSize.y
    );
}

float4 trainProjectedColor(GaussianTrainParam gaussian, float3 modelPosition) {
    float3 viewVector = modelPosition - trainingCamera.cameraPosition.xyz;
    float3 viewDirection = dot(viewVector, viewVector) > 1e-12 ? normalize(viewVector) : float3(0.0, 0.0, 1.0);
    float4 color = evaluateSH(gaussian, viewDirection);
    return float4(max(color.xyz, float3(0.0)), 1.0);
}

ProjectionResult trainProjectGaussian(GaussianTrainParam gaussian) {
    ProjectionResult result;
    result.visible = false;
    result.center = float2(0.0);
    result.depth = 0.0;
    result.radius = 0.0;
    result.conicOpacity = float4(0.0);
    result.color = float4(0.0);

    float4 modelPosition4 = mul(trainingCamera.model, float4(gaussian.positionOpacity.xyz, 1.0));
    float4 cameraPosition4 = mul(trainingCamera.view, modelPosition4);
    float4 clipPosition = mul(trainingCamera.projection, cameraPosition4);
    if (clipPosition.w <= 1e-6) {
        return result;
    }

    float3 ndc = clipPosition.xyz / clipPosition.w;
    if (ndc.x < -1.3 || ndc.x > 1.3 || ndc.y < -1.3 || ndc.y > 1.3 || ndc.z < 0.0 || ndc.z > 1.0) {
        return result;
    }

    float3x3 modelMatrix = trainMat3FromMat4(trainingCamera.model);
    float3x3 covarianceWorld = mul(mul(modelMatrix, trainCovariance3D(gaussian)), transpose(modelMatrix));
    float2x2 covariancePixel = trainCovariance2DPixel(cameraPosition4.xyz, covarianceWorld);
    float radius = trainGaussianRadius(covariancePixel);
    if (radius <= 0.0 || gaussian.positionOpacity.w <= 0.0) {
        return result;
    }

    result.visible = true;
    result.center = trainNdcToPixel(ndc.xy);
    result.depth = cameraPosition4.z;
    result.radius = radius;
    result.conicOpacity = trainConicOpacity(covariancePixel, gaussian.positionOpacity.w);
    result.color = trainProjectedColor(gaussian, gaussian.positionOpacity.xyz);
    return result;
}

ProjectedGaussian trainPackProjectedGaussian(ProjectionResult projection) {
    ProjectedGaussian projected;
    projected.centerRadius = float4(projection.center, projection.depth, projection.radius);
    projected.conicOpacity = projection.conicOpacity;
    projected.color = projection.color;
    return projected;
}
