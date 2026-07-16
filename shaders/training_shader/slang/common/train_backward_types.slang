#ifndef TRAIN_BACKWARD_TYPES_SLANG
#define TRAIN_BACKWARD_TYPES_SLANG

struct PixelGrad {
    float4 color;
};

struct SsimBackwardState {
    float4 base;
    float4 targetCoefficient;
    float4 renderedCoefficient;
};

struct ProjectedGaussianGrad {
    float4 centerRadius;
    float4 conicOpacity;
    float4 color;
};

ProjectedGaussianGrad makeZeroProjectedGaussianGrad() {
    ProjectedGaussianGrad grad;
    grad.centerRadius = float4(0.0);
    grad.conicOpacity = float4(0.0);
    grad.color = float4(0.0);
    return grad;
}

#endif
