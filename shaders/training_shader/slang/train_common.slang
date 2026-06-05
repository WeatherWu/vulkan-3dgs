struct TrainingPushConstants {
    uint gaussianCount;
    uint pixelCount;
    uint width;
    uint height;
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

    float4 result = gaussian.sh[0] * SH_C0;
    result += SH_C1 * (
        -viewDirection.y * gaussian.sh[1] +
         viewDirection.z * gaussian.sh[2] -
         viewDirection.x * gaussian.sh[3]
    );
    return result;
}

uint pixelIndex(uint2 pixel) {
    return pixel.y * params.width + pixel.x;
}
