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

float3 backwardCameraVectorToPosition(float3 dLoss_dCamera) {
    float4 dLoss_dModel = mul(transpose(trainingCamera.view), float4(dLoss_dCamera, 0.0));
    float4 dLoss_dObject = mul(transpose(trainingCamera.model), dLoss_dModel);
    return dLoss_dObject.xyz;
}

float3x3 backwardConicToCovariance(float2x2 covariancePixel, float3 dLoss_dConic) {
    float a = covariancePixel[0][0];
    float b = covariancePixel[0][1];
    float c = covariancePixel[1][1];
    float determinant = a * c - b * b;
    float determinant2Inv = 1.0 / (determinant * determinant + 1e-7);

    float dLoss_da = determinant2Inv *
        (-c * c * dLoss_dConic.x +
          2.0 * b * c * dLoss_dConic.y +
          (determinant - a * c) * dLoss_dConic.z);
    float dLoss_db = determinant2Inv * 2.0 *
        (b * c * dLoss_dConic.x -
         (determinant + 2.0 * b * b) * dLoss_dConic.y +
         a * b * dLoss_dConic.z);
    float dLoss_dc = determinant2Inv *
        (-a * a * dLoss_dConic.z +
          2.0 * a * b * dLoss_dConic.y +
          (determinant - a * c) * dLoss_dConic.x);

    return float3x3(
        dLoss_da, 0.5 * dLoss_db, 0.0,
        0.5 * dLoss_db, dLoss_dc, 0.0,
        0.0, 0.0, 0.0
    );
}

float3 backwardCovariancePixelToPosition(ProjectionLocalState state, float3x3 dLoss_dCovariancePixel3) {
    float3x3 viewMatrix = trainMat3FromMat4(trainingCamera.view);
    float3x3 covarianceCamera = mul(mul(viewMatrix, state.covarianceWorld), transpose(viewMatrix));
    ProjectionClampState clampState = trainClampCameraPositionForProjectionWithGrad(state.cameraPosition.xyz);
    float3x3 jacobian = trainProjectionJacobian(clampState.cameraPosition);
    float3x3 dLoss_dJacobian = mul(mul(dLoss_dCovariancePixel3 + transpose(dLoss_dCovariancePixel3),
                                       jacobian),
                                   covarianceCamera);

    float cameraZ = max(state.cameraPosition.z, 1e-6);
    float cameraZ2 = cameraZ * cameraZ;
    float cameraZ3 = cameraZ2 * cameraZ;
    float fx = trainingCamera.focalTan.x;
    float fy = trainingCamera.focalTan.y;

    float dLoss_dCameraX = clampState.gradMask.x *
        (-fx / cameraZ2) * dLoss_dJacobian[0][2];
    float dLoss_dCameraY = clampState.gradMask.y *
        (-fy / cameraZ2) * dLoss_dJacobian[1][2];
    float dLoss_dCameraZ =
        (-fx / cameraZ2) * dLoss_dJacobian[0][0] +
        (-fy / cameraZ2) * dLoss_dJacobian[1][1] +
        (2.0 * fx * clampState.cameraPosition.x / cameraZ3) * dLoss_dJacobian[0][2] +
        (2.0 * fy * clampState.cameraPosition.y / cameraZ3) * dLoss_dJacobian[1][2];

    return backwardCameraVectorToPosition(float3(dLoss_dCameraX, dLoss_dCameraY, dLoss_dCameraZ));
}

float3x3 backwardCovariancePixelToCovariance3D(ProjectionLocalState state, float3x3 dLoss_dCovariancePixel3) {
    float3x3 viewMatrix = trainMat3FromMat4(trainingCamera.view);
    float3x3 modelMatrix = trainMat3FromMat4(trainingCamera.model);
    float3x3 jacobian = trainProjectionJacobian(trainClampCameraPositionForProjection(state.cameraPosition.xyz));

    float3x3 chain = mul(jacobian, viewMatrix);
    float3x3 dLoss_dCovarianceWorld = mul(mul(transpose(chain), dLoss_dCovariancePixel3), chain);
    return mul(mul(transpose(modelMatrix), dLoss_dCovarianceWorld), modelMatrix);
}

float3 backwardCovariance3DToScale(GaussianTrainParam gaussian, float3x3 dLoss_dCovariance3D) {
    float4 normalizedRotation = trainNormalizeQuaternion(gaussian.rotation);
    float3x3 rotation = trainQuatToMat3(normalizedRotation);
    float3x3 localGrad = mul(mul(transpose(rotation), dLoss_dCovariance3D), rotation);
    float3 scale = trainScaleActivation(gaussian.scale.xyz);
    float3 dLoss_dActivatedScale = 2.0 * scale * float3(localGrad[0][0], localGrad[1][1], localGrad[2][2]);
    return dLoss_dActivatedScale * trainScaleActivationBackward(gaussian.scale.xyz);
}

float4 backwardNormalizeQuat(float4 q, float4 dLoss_dNormalizedQ) {
    float lengthQ = length(q);
    if (lengthQ <= 1e-12) {
        return dLoss_dNormalizedQ / 1e-12;
    }
    float4 normalizedQ = q / lengthQ;
    return (dLoss_dNormalizedQ - normalizedQ * dot(normalizedQ, dLoss_dNormalizedQ)) / lengthQ;
}

float4 backwardRotationMatrixToQuaternion(float4 normalizedQ, float3x3 dLoss_dRotation) {
    float x = normalizedQ.x;
    float y = normalizedQ.y;
    float z = normalizedQ.z;
    float w = normalizedQ.w;

    float4 grad = float4(0.0);

    grad.x += dLoss_dRotation[0][1] * (2.0 * y);
    grad.x += dLoss_dRotation[0][2] * (2.0 * z);
    grad.x += dLoss_dRotation[1][0] * (2.0 * y);
    grad.x += dLoss_dRotation[1][1] * (-4.0 * x);
    grad.x += dLoss_dRotation[1][2] * (-2.0 * w);
    grad.x += dLoss_dRotation[2][0] * (2.0 * z);
    grad.x += dLoss_dRotation[2][1] * (2.0 * w);
    grad.x += dLoss_dRotation[2][2] * (-4.0 * x);

    grad.y += dLoss_dRotation[0][0] * (-4.0 * y);
    grad.y += dLoss_dRotation[0][1] * (2.0 * x);
    grad.y += dLoss_dRotation[0][2] * (2.0 * w);
    grad.y += dLoss_dRotation[1][0] * (2.0 * x);
    grad.y += dLoss_dRotation[1][2] * (2.0 * z);
    grad.y += dLoss_dRotation[2][0] * (-2.0 * w);
    grad.y += dLoss_dRotation[2][1] * (2.0 * z);
    grad.y += dLoss_dRotation[2][2] * (-4.0 * y);

    grad.z += dLoss_dRotation[0][0] * (-4.0 * z);
    grad.z += dLoss_dRotation[0][1] * (-2.0 * w);
    grad.z += dLoss_dRotation[0][2] * (2.0 * x);
    grad.z += dLoss_dRotation[1][0] * (2.0 * w);
    grad.z += dLoss_dRotation[1][1] * (-4.0 * z);
    grad.z += dLoss_dRotation[1][2] * (2.0 * y);
    grad.z += dLoss_dRotation[2][0] * (2.0 * x);
    grad.z += dLoss_dRotation[2][1] * (2.0 * y);

    grad.w += dLoss_dRotation[0][1] * (-2.0 * z);
    grad.w += dLoss_dRotation[0][2] * (2.0 * y);
    grad.w += dLoss_dRotation[1][0] * (2.0 * z);
    grad.w += dLoss_dRotation[1][2] * (-2.0 * x);
    grad.w += dLoss_dRotation[2][0] * (-2.0 * y);
    grad.w += dLoss_dRotation[2][1] * (2.0 * x);

    return grad;
}

float4 backwardCovariance3DToRotation(GaussianTrainParam gaussian, float3x3 dLoss_dCovariance3D) {
    float4 normalizedRotation = trainNormalizeQuaternion(gaussian.rotation);
    float3x3 rotation = trainQuatToMat3(normalizedRotation);
    float3 scale = trainScaleActivation(gaussian.scale.xyz);
    float3x3 scaleMatrix = float3x3(
        scale.x * scale.x, 0.0, 0.0,
        0.0, scale.y * scale.y, 0.0,
        0.0, 0.0, scale.z * scale.z
    );

    float3x3 dLoss_dRotation = mul(dLoss_dCovariance3D + transpose(dLoss_dCovariance3D), mul(rotation, scaleMatrix));
    return backwardNormalizeQuat(gaussian.rotation, backwardRotationMatrixToQuaternion(normalizedRotation, dLoss_dRotation));
}

float3 activeProjectedColorGrad(GaussianTrainParam gaussian,
                                float3 dLoss_dColor,
                                out float3 viewDirection) {
    float3 viewVector = gaussian.positionOpacity.xyz - trainingCamera.cameraPosition.xyz;
    viewDirection = dot(viewVector, viewVector) > 1e-12 ? normalize(viewVector) : float3(0.0, 0.0, 1.0);
    float4 unclampedColor = evaluateSH(gaussian, viewDirection);
    float3 restoredColor = unclampedColor.xyz + float3(0.5);
    float3 active = float3(restoredColor.x > 0.0 ? 1.0 : 0.0,
                          restoredColor.y > 0.0 ? 1.0 : 0.0,
                          restoredColor.z > 0.0 ? 1.0 : 0.0);
    return dLoss_dColor * active;
}

void backwardProjectedColorToSH(inout GaussianGrad gaussianGrad,
                                GaussianTrainParam gaussian,
                                float3 dLoss_dColor) {
    static const float SH_C0 = 0.28209479177387814;
    static const float SH_C1 = 0.4886025119029199;
    static const float SH_C2[5] = {
        1.0925484305920792,
       -1.0925484305920792,
        0.31539156525252005,
       -1.0925484305920792,
        0.5462742152960396
    };
    static const float SH_C3[7] = {
       -0.5900435899266435,
        2.890611442640554,
       -0.4570457994644658,
        0.3731763325901154,
       -0.4570457994644658,
        1.445305721320277,
       -0.5900435899266435
    };

    float3 viewDirection;
    float3 gradColor = activeProjectedColorGrad(gaussian, dLoss_dColor, viewDirection);
    float x = viewDirection.x;
    float y = viewDirection.y;
    float z = viewDirection.z;
    float xx = x * x;
    float yy = y * y;
    float zz = z * z;
    float xy = x * y;
    float yz = y * z;
    float xz = x * z;

    trainAddSHGradient(gaussianGrad, 0u, gradColor * SH_C0);
    if (params.activeSHDegree < 1) {
        return;
    }

    trainAddSHGradient(gaussianGrad, 1u, gradColor * (-viewDirection.y * SH_C1));
    trainAddSHGradient(gaussianGrad, 2u, gradColor * ( viewDirection.z * SH_C1));
    trainAddSHGradient(gaussianGrad, 3u, gradColor * (-viewDirection.x * SH_C1));
    if (params.activeSHDegree < 2) {
        return;
    }

    trainAddSHGradient(gaussianGrad, 4u, gradColor * (SH_C2[0] * xy));
    trainAddSHGradient(gaussianGrad, 5u, gradColor * (SH_C2[1] * yz));
    trainAddSHGradient(gaussianGrad, 6u, gradColor * (SH_C2[2] * (2.0 * zz - xx - yy)));
    trainAddSHGradient(gaussianGrad, 7u, gradColor * (SH_C2[3] * xz));
    trainAddSHGradient(gaussianGrad, 8u, gradColor * (SH_C2[4] * (xx - yy)));
    if (params.activeSHDegree < 3) {
        return;
    }

    trainAddSHGradient(gaussianGrad, 9u, gradColor * (SH_C3[0] * y * (3.0 * xx - yy)));
    trainAddSHGradient(gaussianGrad, 10u, gradColor * (SH_C3[1] * xy * z));
    trainAddSHGradient(gaussianGrad, 11u, gradColor * (SH_C3[2] * y * (4.0 * zz - xx - yy)));
    trainAddSHGradient(gaussianGrad, 12u, gradColor * (SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy)));
    trainAddSHGradient(gaussianGrad, 13u, gradColor * (SH_C3[4] * x * (4.0 * zz - xx - yy)));
    trainAddSHGradient(gaussianGrad, 14u, gradColor * (SH_C3[5] * z * (xx - yy)));
    trainAddSHGradient(gaussianGrad, 15u, gradColor * (SH_C3[6] * x * (xx - 3.0 * yy)));
}

float3 backwardProjectedColorToPosition(GaussianTrainParam gaussian, float3 dLoss_dColor) {
    static const float SH_C1 = 0.4886025119029199;
    static const float SH_C2[5] = {
        1.0925484305920792,
       -1.0925484305920792,
        0.31539156525252005,
       -1.0925484305920792,
        0.5462742152960396
    };
    static const float SH_C3[7] = {
       -0.5900435899266435,
        2.890611442640554,
       -0.4570457994644658,
        0.3731763325901154,
       -0.4570457994644658,
        1.445305721320277,
       -0.5900435899266435
    };

    float3 viewDirection;
    float3 gradColor = activeProjectedColorGrad(gaussian, dLoss_dColor, viewDirection);
    if (params.activeSHDegree < 1) {
        return float3(0.0);
    }

    float3 sh1 = trainGetSHCoefficient(gaussian, 1u);
    float3 sh2 = trainGetSHCoefficient(gaussian, 2u);
    float3 sh3 = trainGetSHCoefficient(gaussian, 3u);
    float3 dLoss_dViewDirection = float3(
        dot(gradColor, -sh3 * SH_C1),
        dot(gradColor, -sh1 * SH_C1),
        dot(gradColor,  sh2 * SH_C1));

    float x = viewDirection.x;
    float y = viewDirection.y;
    float z = viewDirection.z;
    float xx = x * x;
    float yy = y * y;
    float zz = z * z;

    if (params.activeSHDegree >= 2) {
    float3 sh4 = trainGetSHCoefficient(gaussian, 4u);
    float3 sh5 = trainGetSHCoefficient(gaussian, 5u);
    float3 sh6 = trainGetSHCoefficient(gaussian, 6u);
    float3 sh7 = trainGetSHCoefficient(gaussian, 7u);
    float3 sh8 = trainGetSHCoefficient(gaussian, 8u);
    dLoss_dViewDirection.x += dot(gradColor, sh4) * (SH_C2[0] * y);
    dLoss_dViewDirection.y += dot(gradColor, sh4) * (SH_C2[0] * x);

    dLoss_dViewDirection.y += dot(gradColor, sh5) * (SH_C2[1] * z);
    dLoss_dViewDirection.z += dot(gradColor, sh5) * (SH_C2[1] * y);

    dLoss_dViewDirection.x += dot(gradColor, sh6) * (-2.0 * SH_C2[2] * x);
    dLoss_dViewDirection.y += dot(gradColor, sh6) * (-2.0 * SH_C2[2] * y);
    dLoss_dViewDirection.z += dot(gradColor, sh6) * (4.0 * SH_C2[2] * z);

    dLoss_dViewDirection.x += dot(gradColor, sh7) * (SH_C2[3] * z);
    dLoss_dViewDirection.z += dot(gradColor, sh7) * (SH_C2[3] * x);

    dLoss_dViewDirection.x += dot(gradColor, sh8) * (2.0 * SH_C2[4] * x);
    dLoss_dViewDirection.y += dot(gradColor, sh8) * (-2.0 * SH_C2[4] * y);
    }

    if (params.activeSHDegree >= 3) {
    float3 sh9 = trainGetSHCoefficient(gaussian, 9u);
    float3 sh10 = trainGetSHCoefficient(gaussian, 10u);
    float3 sh11 = trainGetSHCoefficient(gaussian, 11u);
    float3 sh12 = trainGetSHCoefficient(gaussian, 12u);
    float3 sh13 = trainGetSHCoefficient(gaussian, 13u);
    float3 sh14 = trainGetSHCoefficient(gaussian, 14u);
    float3 sh15 = trainGetSHCoefficient(gaussian, 15u);
    dLoss_dViewDirection.x += dot(gradColor, sh9) * (6.0 * SH_C3[0] * x * y);
    dLoss_dViewDirection.y += dot(gradColor, sh9) * (3.0 * SH_C3[0] * (xx - yy));

    dLoss_dViewDirection.x += dot(gradColor, sh10) * (SH_C3[1] * y * z);
    dLoss_dViewDirection.y += dot(gradColor, sh10) * (SH_C3[1] * x * z);
    dLoss_dViewDirection.z += dot(gradColor, sh10) * (SH_C3[1] * x * y);

    dLoss_dViewDirection.x += dot(gradColor, sh11) * (-2.0 * SH_C3[2] * x * y);
    dLoss_dViewDirection.y += dot(gradColor, sh11) * (SH_C3[2] * (4.0 * zz - xx - 3.0 * yy));
    dLoss_dViewDirection.z += dot(gradColor, sh11) * (8.0 * SH_C3[2] * y * z);

    dLoss_dViewDirection.x += dot(gradColor, sh12) * (-6.0 * SH_C3[3] * x * z);
    dLoss_dViewDirection.y += dot(gradColor, sh12) * (-6.0 * SH_C3[3] * y * z);
    dLoss_dViewDirection.z += dot(gradColor, sh12) * (SH_C3[3] * (6.0 * zz - 3.0 * xx - 3.0 * yy));

    dLoss_dViewDirection.x += dot(gradColor, sh13) * (SH_C3[4] * (4.0 * zz - 3.0 * xx - yy));
    dLoss_dViewDirection.y += dot(gradColor, sh13) * (-2.0 * SH_C3[4] * x * y);
    dLoss_dViewDirection.z += dot(gradColor, sh13) * (8.0 * SH_C3[4] * x * z);

    dLoss_dViewDirection.x += dot(gradColor, sh14) * (2.0 * SH_C3[5] * x * z);
    dLoss_dViewDirection.y += dot(gradColor, sh14) * (-2.0 * SH_C3[5] * y * z);
    dLoss_dViewDirection.z += dot(gradColor, sh14) * (SH_C3[5] * (xx - yy));

    dLoss_dViewDirection.x += dot(gradColor, sh15) * (3.0 * SH_C3[6] * (xx - yy));
    dLoss_dViewDirection.y += dot(gradColor, sh15) * (-6.0 * SH_C3[6] * x * y);
    }

    float3 viewVector = gaussian.positionOpacity.xyz - trainingCamera.cameraPosition.xyz;
    float viewLength = length(viewVector);
    if (viewLength <= 1e-6) {
        return float3(0.0);
    }

    return (dLoss_dViewDirection - viewDirection * dot(viewDirection, dLoss_dViewDirection)) / viewLength;
}

bool projectedGaussianHasBackwardSupport(ProjectedGaussian projectedGaussian) {
    return projectedGaussian.centerRadius.w > 0.0 &&
           isfinite(projectedGaussian.centerRadius.w);
}

void accumulateProjectedGradientToGaussian(
    GaussianTrainParam gaussian,
    ProjectedGaussianGrad projectedGrad,
    inout GaussianGrad grad,
    inout GaussianDensificationState densification) {
    ProjectionLocalState projectionState = trainBuildProjectionLocalState(gaussian);
    float2 dLoss_dCenter = projectedGrad.centerRadius.xy;
    grad.positionOpacity.xyz += backwardPixelCenterToPosition(
        projectionState, dLoss_dCenter);
    grad.positionOpacity.xyz += backwardDepthToPosition(
        projectedGrad.centerRadius.z);
    grad.positionOpacity.w += projectedGrad.conicOpacity.w *
                              trainOpacityActivationBackward(
                                  gaussian.positionOpacity.w);

    float3x3 dLoss_dCovariancePixel = backwardConicToCovariance(
        projectionState.covariancePixel, projectedGrad.conicOpacity.xyz);
    grad.positionOpacity.xyz += backwardCovariancePixelToPosition(
        projectionState, dLoss_dCovariancePixel);
    float3x3 dLoss_dCovariance3D = backwardCovariancePixelToCovariance3D(
        projectionState, dLoss_dCovariancePixel);
    grad.scale.xyz += backwardCovariance3DToScale(
        gaussian, dLoss_dCovariance3D);
    grad.rotation += backwardCovariance3DToRotation(
        gaussian, dLoss_dCovariance3D);

    grad.positionOpacity.xyz += backwardProjectedColorToPosition(
        gaussian, projectedGrad.color.xyz);
    backwardProjectedColorToSH(grad, gaussian, projectedGrad.color.xyz);

    float2 densificationGrad = dLoss_dCenter *
        (0.5 * max(trainingCamera.viewport.zw, float2(1.0)));
    densification.screenGradSum += length(densificationGrad);
    densification.screenGradCount += 1u;
}
