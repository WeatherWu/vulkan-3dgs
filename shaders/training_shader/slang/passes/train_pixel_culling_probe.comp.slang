#include "../common/train_forward_common.slang"
#include "../common/train_pixel_culling.slang"

// Explicit diagnostic only. Output each classification, not global atomics.
// Inputs: opacity, power, valid-pixel flag, inside-processed-prefix flag.
[[vk::binding(0, 0)]] StructuredBuffer<float4> inputs;
[[vk::binding(1, 0)]] RWStructuredBuffer<float4> outputs;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint4> decisions;
groupshared float thresholds[128];

[shader("compute")]
[numthreads(128, 1, 1)]
void main(uint3 id : SV_DispatchThreadID, uint3 local : SV_GroupThreadID) {
    bool valid = id.x < params.pixelCount;
    float4 input = valid ? inputs[id.x] : float4(0.0);
    thresholds[local.x] = trainPixelCullThreshold(input.x);
    GroupMemoryBarrierWithGroupSync();
    float threshold = thresholds[local.x];
    bool eligible = valid && input.z != 0.0 && input.w != 0.0;
    bool oldExp = eligible && isInsideGaussianSupport(input.y);
    bool rejected = oldExp && trainPixelCullReject(input.y, threshold);
    // All native lanes, including the tail, participate. Diagnostic VOTE only;
    // the actual EarlyCull Tensor shader does not require this capability.
    bool allSkipped = WaveActiveAnyTrue(oldExp) && WaveActiveAllTrue(!oldExp || rejected);
    bool skippedLeader = WaveIsFirstLane() && allSkipped;
    if (!valid) return;
    float coverage = oldExp ? evaluateGaussianCoverage(input.y) : 0.0;
    ProjectedGaussian gaussian;
    gaussian.centerRadius = float4(0.0);
    gaussian.conicOpacity = float4(0.0, 0.0, 0.0, input.x);
    gaussian.color = float4(0.0);
    float alpha = eligible ? evaluateGaussianAlpha(gaussian, coverage) : 0.0;
    float candidate = rejected ? 0.0 : alpha;
    outputs[id.x] = float4(threshold, skippedLeader ? 1.0 : 0.0, alpha, candidate);
    decisions[id.x] = uint4(eligible, oldExp, rejected, alpha >= TRAIN_ALPHA_MIN);
}
