struct TrainingPushConstants {
    uint gaussianCount;
    uint pixelCount;
    uint width;
    uint height;
    uint trainingIteration;
    uint activeSHDegree;
    uint maxSHDegree;
    uint tileItemCount;
    float positionLearningRate;
    float positionLearningRateFinal;
    float positionLearningRateDelayMult;
    float positionLearningRateMaxSteps;
    float featureLearningRate;
    float featureRestLearningRate;
    float opacityLearningRate;
    float scaleLearningRate;
    float rotationLearningRate;
    float optimizerBeta1;
    float optimizerBeta2;
    float optimizerEpsilon;
    float optimizerGradClip;
    float lossDssimWeight;
    uint validationEnabled;
    uint validationIteration;
};

struct TrainingPixelValidationPartial {
    float lossSum;
    float maxLoss;
    float alphaSum;
    float luminanceSum;
    uint invalidLossCount;
    uint invalidRenderedPixelCount;
    uint validRenderedPixelCount;
    uint padding0;
};

struct TrainingGaussianValidationPartial {
    uint nonFiniteGaussianCount;
    uint padding0;
    uint padding1;
    uint padding2;
};

struct TrainingValidationGpuResult {
    float lossSum;
    float maxLoss;
    float alphaSum;
    float luminanceSum;
    uint invalidLossCount;
    uint invalidRenderedPixelCount;
    uint validRenderedPixelCount;
    uint nonFiniteGaussianCount;
    uint validationIteration;
    uint padding0;
    uint padding1;
    uint padding2;
    uint padding3;
    uint padding4;
    uint padding5;
    uint padding6;
};

struct GaussianTrainParam {
    float4 positionOpacity;
    float4 scale;
    float4 rotation;
    float4 sh[16];
};

struct ProjectedGaussian {
    float4 centerRadius;
    float4 conicOpacity;
    float4 color;
};

struct PixelBlendState {
    float finalTransmittance;
    float accumulatedAlpha;
    uint processedCount;
    uint lastContributor;
};

#ifndef TRAIN_USE_UINT_RADIUS
#define TRAIN_USE_UINT_RADIUS 0
#endif

struct GaussianVisibilityState {
    uint contributionCount;
    float alphaSum;
#if TRAIN_USE_UINT_RADIUS
    uint maxScreenRadiusBits;
#else
    float maxScreenRadius;
#endif
    uint padding0;
};

struct GaussianDensificationState {
    float screenGradSum;
    uint screenGradCount;
#if TRAIN_USE_UINT_RADIUS
    uint maxScreenRadiusBits;
#else
    float maxScreenRadius;
#endif
    uint padding0;
};

struct GaussianGrad {
    float4 positionOpacity;
    float4 scale;
    float4 rotation;
    float4 sh[16];
};

struct AdamState {
    GaussianGrad firstMoment;
    GaussianGrad secondMoment;
};

[[vk::push_constant]]
ConstantBuffer<TrainingPushConstants> params;

float4 evaluateSH(GaussianTrainParam gaussian, float3 viewDirection) {
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

    float4 result = gaussian.sh[0] * SH_C0;
    float x = viewDirection.x;
    float y = viewDirection.y;
    float z = viewDirection.z;
    float xx = x * x;
    float yy = y * y;
    float zz = z * z;
    float xy = x * y;
    float yz = y * z;
    float xz = x * z;

    if (params.activeSHDegree < 1) {
        return result;
    }

    result += SH_C1 * (
        -y * gaussian.sh[1] +
         z * gaussian.sh[2] -
         x * gaussian.sh[3]
    );

    if (params.activeSHDegree < 2) {
        return result;
    }

    result +=
        SH_C2[0] * xy * gaussian.sh[4] +
        SH_C2[1] * yz * gaussian.sh[5] +
        SH_C2[2] * (2.0 * zz - xx - yy) * gaussian.sh[6] +
        SH_C2[3] * xz * gaussian.sh[7] +
        SH_C2[4] * (xx - yy) * gaussian.sh[8];

    if (params.activeSHDegree < 3) {
        return result;
    }

    result +=
        SH_C3[0] * y * (3.0 * xx - yy) * gaussian.sh[9] +
        SH_C3[1] * xy * z * gaussian.sh[10] +
        SH_C3[2] * y * (4.0 * zz - xx - yy) * gaussian.sh[11] +
        SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * gaussian.sh[12] +
        SH_C3[4] * x * (4.0 * zz - xx - yy) * gaussian.sh[13] +
        SH_C3[5] * z * (xx - yy) * gaussian.sh[14] +
        SH_C3[6] * x * (xx - 3.0 * yy) * gaussian.sh[15];
    return result;
}

uint pixelIndex(uint2 pixel) {
    return pixel.y * params.width + pixel.x;
}

void clearVisibilityMaxScreenRadius(inout GaussianVisibilityState state) {
#if TRAIN_USE_UINT_RADIUS
    state.maxScreenRadiusBits = 0u;
#else
    state.maxScreenRadius = 0.0;
#endif
}

void clearDensificationMaxScreenRadius(inout GaussianDensificationState state) {
#if TRAIN_USE_UINT_RADIUS
    state.maxScreenRadiusBits = 0u;
#else
    state.maxScreenRadius = 0.0;
#endif
}

void atomicMaxVisibilityScreenRadius(inout GaussianVisibilityState state, float radius) {
#if TRAIN_USE_UINT_RADIUS
    uint ignored = 0u;
    InterlockedMax(state.maxScreenRadiusBits, asuint(max(radius, 0.0)), ignored);
#else
    float ignored = 0.0;
    InterlockedMax(state.maxScreenRadius, radius, ignored);
#endif
}

void atomicMaxDensificationScreenRadius(inout GaussianDensificationState state, float radius) {
#if TRAIN_USE_UINT_RADIUS
    uint ignored = 0u;
    InterlockedMax(state.maxScreenRadiusBits, asuint(max(radius, 0.0)), ignored);
#else
    float ignored = 0.0;
    InterlockedMax(state.maxScreenRadius, radius, ignored);
#endif
}

float loadDensificationMaxScreenRadius(GaussianDensificationState state) {
#if TRAIN_USE_UINT_RADIUS
    return asfloat(state.maxScreenRadiusBits);
#else
    return state.maxScreenRadius;
#endif
}
