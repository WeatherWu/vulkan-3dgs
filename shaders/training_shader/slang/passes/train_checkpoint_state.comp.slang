#include "../common/train_forward_common.slang"
#include "../common/train_checkpoint_layout.slang"
[[vk::binding(3, 0)]] StructuredBuffer<ProjectedGaussian> projectedGaussians;
[[vk::binding(4, 0)]] StructuredBuffer<uint> tileItems;
[[vk::binding(14, 0)]] StructuredBuffer<PixelBlendState> pixelBlendStates;
[[vk::binding(45, 0)]] RWStructuredBuffer<float4> bank0;
[[vk::binding(46, 0)]] RWStructuredBuffer<float4> bank1;
[[vk::binding(47, 0)]] RWStructuredBuffer<float4> bank2;
[[vk::binding(48, 0)]] RWStructuredBuffer<float4> bank3;
groupshared uint indices[256];
groupshared ProjectedGaussian gaussians[256];
groupshared uint writes[256];
groupshared uint initialError;
[shader("compute")]
[numthreads(16,16,1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    uint tile = gid.y * tileCountX() + gid.x;
    uint lane = tid.y * 16u + tid.x;
    if (lane == 0u) InterlockedOr(checkpointControl[1].z, 0u, initialError);
    GroupMemoryBarrierWithGroupSync();
    if (initialError != 0u) return;
    uint2 pixel = gid.xy * 16u + tid.xy;
    bool valid = pixel.x < params.width && pixel.y < params.height;
    CheckpointTile meta = checkpointTiles[tile];
    // The nearest bucket consumes B(32); no reader ever needs B(0).
    // Pixels whose entire prefix fits one bucket need no stored state.
    if (meta.activity.x <= 32u) return;
    uint p = 0u;
    float T = 1.0;
    if (valid) {
        PixelBlendState blend = pixelBlendStates[pixelIndex(pixel)];
        p = min(blend.processedCount, meta.range.y);
        T = blend.finalTransmittance;
    }
    float3 suffix = float3(0.0);
    uint written = 0u;
    for (uint remaining = meta.activity.x; remaining > 32u;) {
        uint count = min(remaining - 32u, 256u);
        uint start = remaining - count;
        if (lane < count) {
            uint id = tileItems[meta.range.x + start + lane];
            indices[lane] = id;
            if (id < params.gaussianCount) gaussians[lane] = projectedGaussians[id];
        }
        GroupMemoryBarrierWithGroupSync();
        for (uint j = count; j > 0u; --j) {
            uint offset = start + j - 1u;
            if (valid && offset < p) {
                if (indices[j - 1u] < params.gaussianCount) {
                    GaussianSample sample = evaluateGaussianSample(gaussians[j - 1u], pixelCenter(pixel));
                    if (contributesToPixel(sample)) {
                        float u = max(1.0 - sample.alpha, 1e-6);
                        T /= u;
                        suffix = sample.alpha * sample.color + u * suffix;
                    }
                }
                if (offset > 0u && offset % 32u == 0u) {
                    uint slot = meta.range.z + offset / 32u - 1u;
                    if (slot >= checkpointControl[0].x) { checkpointError(2u); }
                    else {
                        float4 state = float4(suffix, T);
                        if (!all(isfinite(state))) checkpointError(8u);
                        uint element = slot * 64u + (lane & 63u);
                        switch (lane >> 6u) {
                            case 0u: bank0[element] = state; break;
                            case 1u: bank1[element] = state; break;
                            case 2u: bank2[element] = state; break;
                            default: bank3[element] = state; break;
                        }
                        ++written;
                    }
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
        remaining = start;
    }
    writes[lane] = written;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (lane < stride) writes[lane] += writes[lane + stride];
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0u) { uint ignored; InterlockedAdd(checkpointControl[3].x, writes[0], ignored); }
}
