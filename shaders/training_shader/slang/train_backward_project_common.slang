#include "train_forward_project_common.slang"
#include "train_backward_types.slang"

struct ProjectionLocalState {
    float4 modelPosition;
    float4 cameraPosition;
    float4 clipPosition;
    float3 ndc;
    float2x2 covariancePixel;
    float3x3 covarianceWorld;
    float3x3 covariance3D;
};

ProjectionLocalState trainBuildProjectionLocalState(GaussianTrainParam gaussian) {
    ProjectionLocalState state;
    state.modelPosition = mul(trainingCamera.model, float4(gaussian.positionOpacity.xyz, 1.0));
    state.cameraPosition = mul(trainingCamera.view, state.modelPosition);
    state.clipPosition = mul(trainingCamera.projection, state.cameraPosition);
    state.ndc = state.clipPosition.xyz / max(state.clipPosition.w, 1e-6);

    float3x3 modelMatrix = trainMat3FromMat4(trainingCamera.model);
    state.covariance3D = trainCovariance3D(gaussian);
    state.covarianceWorld = mul(mul(modelMatrix, state.covariance3D), transpose(modelMatrix));
    state.covariancePixel = trainCovariance2DPixel(state.cameraPosition.xyz, state.covarianceWorld);
    return state;
}

float3 backwardPixelCenterToPosition(ProjectionLocalState state, float2 dLoss_dCenter) {
    float2 screenSize = max(trainingCamera.viewport.zw, float2(1.0));
    float3 dLoss_dNdc = float3(
        dLoss_dCenter.x * 0.5 * screenSize.x,
        dLoss_dCenter.y * -0.5 * screenSize.y,
        0.0
    );

    float invW = 1.0 / max(state.clipPosition.w, 1e-6);
    float4 dLoss_dClip = float4(0.0);
    dLoss_dClip.x += dLoss_dNdc.x * invW;
    dLoss_dClip.y += dLoss_dNdc.y * invW;
    dLoss_dClip.z += dLoss_dNdc.z * invW;
    dLoss_dClip.w -= dot(dLoss_dNdc, state.clipPosition.xyz) * invW * invW;

    float4 dLoss_dCamera = mul(transpose(trainingCamera.projection), dLoss_dClip);
    float4 dLoss_dModel = mul(transpose(trainingCamera.view), dLoss_dCamera);
    float4 dLoss_dObject = mul(transpose(trainingCamera.model), dLoss_dModel);
    return dLoss_dObject.xyz;
}

float3 backwardDepthToPosition(float dLoss_dDepth) {
    float4 dLoss_dCamera = float4(0.0, 0.0, dLoss_dDepth, 0.0);
    float4 dLoss_dModel = mul(transpose(trainingCamera.view), dLoss_dCamera);
    float4 dLoss_dObject = mul(transpose(trainingCamera.model), dLoss_dModel);
    return dLoss_dObject.xyz;
}

float3x3 backwardConicToCovariance(float2x2 covariancePixel, float3 dLoss_dConic) {
    float a = covariancePixel[0][0];
    float b = covariancePixel[0][1];
    float c = covariancePixel[1][1];
    float determinant = max(a * c - b * b, 1e-12);

    float dLoss_dDeterminant =
        -(dLoss_dConic.x * c - dLoss_dConic.y * b + dLoss_dConic.z * a) /
        (determinant * determinant);

    float dLoss_da = dLoss_dConic.z / determinant + dLoss_dDeterminant * c;
    float dLoss_db = -dLoss_dConic.y / determinant - 2.0 * b * dLoss_dDeterminant;
    float dLoss_dc = dLoss_dConic.x / determinant + dLoss_dDeterminant * a;

    return float3x3(
        dLoss_da, dLoss_db, 0.0,
        dLoss_db, dLoss_dc, 0.0,
        0.0, 0.0, 0.0
    );
}

float3x3 backwardCovariancePixelToCovariance3D(ProjectionLocalState state, float3x3 dLoss_dCovariancePixel3) {
    float3x3 viewMatrix = trainMat3FromMat4(trainingCamera.view);
    float3x3 modelMatrix = trainMat3FromMat4(trainingCamera.model);
    float3x3 jacobian = trainProjectionJacobian(state.cameraPosition.xyz);
    float2 screenSize = max(trainingCamera.viewport.zw, float2(1.0));

    float3x3 projectionScale3 = float3x3(
        trainingCamera.projection[0][0], trainingCamera.projection[0][1], 0.0,
        trainingCamera.projection[1][0], trainingCamera.projection[1][1], 0.0,
        0.0, 0.0, 1.0
    );
    float3x3 ndcToPixel3 = float3x3(
        0.5 * screenSize.x, 0.0, 0.0,
        0.0, 0.5 * screenSize.y, 0.0,
        0.0, 0.0, 1.0
    );

    float3x3 chain = mul(mul(mul(ndcToPixel3, projectionScale3), jacobian), viewMatrix);
    float3x3 dLoss_dCovarianceWorld = mul(mul(transpose(chain), dLoss_dCovariancePixel3), chain);
    return mul(mul(transpose(modelMatrix), dLoss_dCovarianceWorld), modelMatrix);
}

float3 backwardCovariance3DToScale(GaussianTrainParam gaussian, float3x3 dLoss_dCovariance3D) {
    float4 normalizedRotation = normalize(gaussian.rotation);
    float3x3 rotation = trainQuatToMat3(normalizedRotation);
    float3x3 localGrad = mul(mul(transpose(rotation), dLoss_dCovariance3D), rotation);
    float3 scale = max(abs(gaussian.scale.xyz), float3(1e-6));
    return 2.0 * scale * float3(localGrad[0][0], localGrad[1][1], localGrad[2][2]);
}

void backwardProjectedColorToSH(inout GaussianGrad gaussianGrad,
                                GaussianTrainParam gaussian,
                                float3 dLoss_dColor) {
    static const float SH_C0 = 0.28209479177387814;
    static const float SH_C1 = 0.4886025119029199;

    float3 viewVector = gaussian.positionOpacity.xyz - trainingCamera.cameraPosition.xyz;
    float3 viewDirection = dot(viewVector, viewVector) > 1e-12 ? normalize(viewVector) : float3(0.0, 0.0, 1.0);
    float4 unclampedColor = evaluateSH(gaussian, viewDirection);
    float3 active = float3(unclampedColor.x > 0.0 ? 1.0 : 0.0,
                          unclampedColor.y > 0.0 ? 1.0 : 0.0,
                          unclampedColor.z > 0.0 ? 1.0 : 0.0);
    float3 gradColor = dLoss_dColor * active;

    gaussianGrad.sh[0].xyz += gradColor * SH_C0;
    gaussianGrad.sh[1].xyz += gradColor * (-viewDirection.y * SH_C1);
    gaussianGrad.sh[2].xyz += gradColor * ( viewDirection.z * SH_C1);
    gaussianGrad.sh[3].xyz += gradColor * (-viewDirection.x * SH_C1);
}
