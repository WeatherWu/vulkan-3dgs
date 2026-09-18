#include "../common/train_common.slang"

[[vk::binding(0, 0)]]
StructuredBuffer<GaussianTrainParam> gaussianParams;

[[vk::binding(19, 0)]]
StructuredBuffer<uint> densificationCounters;

[[vk::binding(30, 0)]]
RWStructuredBuffer<TrainingGaussianValidationPartial> gaussianValidationPartials;

groupshared uint validationNonFiniteGaussianCount[256];
groupshared uint validationNonFinitePositionCount[256];
groupshared uint validationNonFiniteOpacityCount[256];
groupshared uint validationNonFiniteRawScaleCount[256];
groupshared uint validationNonFiniteActivatedScaleCount[256];
groupshared uint validationNonFiniteRotationCount[256];
groupshared uint validationNonFiniteSHCount[256];
groupshared uint validationFirstNonFiniteGaussianIndex[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID,
          uint3 groupThreadID : SV_GroupThreadID,
          uint3 groupID : SV_GroupID) {
    uint index = dispatchThreadID.x;
    uint gaussianCount = params.validationUsesDensifiedGaussians != 0u
        ? min(densificationCounters[1], params.gaussianCount)
        : params.gaussianCount;
    bool active = index < gaussianCount;
    uint nonFiniteCount = 0u;
    uint nonFinitePositionCount = 0u;
    uint nonFiniteOpacityCount = 0u;
    uint nonFiniteRawScaleCount = 0u;
    uint nonFiniteActivatedScaleCount = 0u;
    uint nonFiniteRotationCount = 0u;
    uint nonFiniteSHCount = 0u;
    uint firstNonFiniteGaussianIndex = 0xffffffffu;

    if (active) {
        GaussianTrainParam gaussian = gaussianParams[index];
        bool finitePosition = all(isfinite(gaussian.positionOpacity.xyz));
        bool finiteOpacity = isfinite(gaussian.positionOpacity.w);
        bool finiteRawScale = all(isfinite(gaussian.scale.xyz));
        bool finiteActivatedScale = all(isfinite(exp(gaussian.scale.xyz)));
        bool finiteRotation = all(isfinite(gaussian.rotation));
        bool finiteSH = true;
        for (uint coefficient = 0u; coefficient < 12u; ++coefficient) {
            finiteSH = finiteSH && all(isfinite(gaussian.sh[coefficient]));
        }
        bool finite = finitePosition && finiteOpacity && finiteRawScale &&
                      finiteActivatedScale && finiteRotation && finiteSH;
        nonFiniteCount = finite ? 0u : 1u;
        nonFinitePositionCount = finitePosition ? 0u : 1u;
        nonFiniteOpacityCount = finiteOpacity ? 0u : 1u;
        nonFiniteRawScaleCount = finiteRawScale ? 0u : 1u;
        nonFiniteActivatedScaleCount = finiteActivatedScale ? 0u : 1u;
        nonFiniteRotationCount = finiteRotation ? 0u : 1u;
        nonFiniteSHCount = finiteSH ? 0u : 1u;
        firstNonFiniteGaussianIndex = finite ? 0xffffffffu : index;
    }

    uint localIndex = groupThreadID.x;
    validationNonFiniteGaussianCount[localIndex] = nonFiniteCount;
    validationNonFinitePositionCount[localIndex] = nonFinitePositionCount;
    validationNonFiniteOpacityCount[localIndex] = nonFiniteOpacityCount;
    validationNonFiniteRawScaleCount[localIndex] = nonFiniteRawScaleCount;
    validationNonFiniteActivatedScaleCount[localIndex] = nonFiniteActivatedScaleCount;
    validationNonFiniteRotationCount[localIndex] = nonFiniteRotationCount;
    validationNonFiniteSHCount[localIndex] = nonFiniteSHCount;
    validationFirstNonFiniteGaussianIndex[localIndex] = firstNonFiniteGaussianIndex;
    GroupMemoryBarrierWithGroupSync();

    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (localIndex < stride) {
            validationNonFiniteGaussianCount[localIndex] +=
                validationNonFiniteGaussianCount[localIndex + stride];
            validationNonFinitePositionCount[localIndex] +=
                validationNonFinitePositionCount[localIndex + stride];
            validationNonFiniteOpacityCount[localIndex] +=
                validationNonFiniteOpacityCount[localIndex + stride];
            validationNonFiniteRawScaleCount[localIndex] +=
                validationNonFiniteRawScaleCount[localIndex + stride];
            validationNonFiniteActivatedScaleCount[localIndex] +=
                validationNonFiniteActivatedScaleCount[localIndex + stride];
            validationNonFiniteRotationCount[localIndex] +=
                validationNonFiniteRotationCount[localIndex + stride];
            validationNonFiniteSHCount[localIndex] +=
                validationNonFiniteSHCount[localIndex + stride];
            validationFirstNonFiniteGaussianIndex[localIndex] = min(
                validationFirstNonFiniteGaussianIndex[localIndex],
                validationFirstNonFiniteGaussianIndex[localIndex + stride]);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (localIndex == 0u) {
        TrainingGaussianValidationPartial partial;
        partial.nonFiniteGaussianCount = validationNonFiniteGaussianCount[0];
        partial.nonFinitePositionCount = validationNonFinitePositionCount[0];
        partial.nonFiniteOpacityCount = validationNonFiniteOpacityCount[0];
        partial.nonFiniteRawScaleCount = validationNonFiniteRawScaleCount[0];
        partial.nonFiniteActivatedScaleCount = validationNonFiniteActivatedScaleCount[0];
        partial.nonFiniteRotationCount = validationNonFiniteRotationCount[0];
        partial.nonFiniteSHCount = validationNonFiniteSHCount[0];
        partial.firstNonFiniteGaussianIndex = validationFirstNonFiniteGaussianIndex[0];
        gaussianValidationPartials[groupID.x] = partial;
    }
}
