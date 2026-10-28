struct MortonPush {
    uint stage;
    uint count;
    uint groups;
    uint padding;
};
[[vk::push_constant]] ConstantBuffer<MortonPush> params;
[[vk::binding(0, 0)]] StructuredBuffer<uint4> sourceParams;
[[vk::binding(1, 0)]] StructuredBuffer<uint4> sourceAdam;
[[vk::binding(2, 0)]] StructuredBuffer<uint4> sourceStats;
[[vk::binding(3, 0)]] RWStructuredBuffer<float4> bounds;
[[vk::binding(4, 0)]] RWStructuredBuffer<uint2> codes;
[[vk::binding(5, 0)]] RWStructuredBuffer<uint> keys;
[[vk::binding(6, 0)]] RWStructuredBuffer<uint> indices;
[[vk::binding(7, 0)]] RWStructuredBuffer<uint4> targetParams;
[[vk::binding(8, 0)]] RWStructuredBuffer<uint4> targetAdam;
[[vk::binding(9, 0)]] RWStructuredBuffer<uint4> targetStats;

groupshared float4 groupMin[256];
groupshared float4 groupMax[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchId : SV_DispatchThreadID, uint3 groupId : SV_GroupID,
          uint lane : SV_GroupIndex) {
    uint index = dispatchId.x;
    if (params.stage == 0u) {
        float infinity = asfloat(0x7f800000u);
        float3 position = index < params.count
            ? asfloat(sourceParams[index * 15u].xyz) : float3(0.0);
        bool valid = all(isfinite(position));
        bool active = index < params.count && valid;
        groupMin[lane] = float4(active ? position : float3(infinity), 0.0);
        groupMax[lane] = float4(active ? position : float3(-infinity),
                                index < params.count && !valid ? 1.0 : 0.0);
        GroupMemoryBarrierWithGroupSync();
        for (uint stride = 128u; stride > 0u; stride >>= 1u) {
            if (lane < stride) {
                groupMin[lane].xyz = min(groupMin[lane].xyz, groupMin[lane + stride].xyz);
                groupMax[lane].xyz = max(groupMax[lane].xyz, groupMax[lane + stride].xyz);
                groupMax[lane].w = max(groupMax[lane].w, groupMax[lane + stride].w);
            }
            GroupMemoryBarrierWithGroupSync();
        }
        if (lane == 0u) {
            bounds[2u + 2u * groupId.x] = groupMin[0];
            bounds[3u + 2u * groupId.x] = groupMax[0];
        }
        return;
    }
    if (params.stage == 1u) {
        if (index != 0u) return;
        float3 minimum = bounds[2].xyz;
        float3 maximum = bounds[3].xyz;
        float invalid = bounds[3].w;
        for (uint group = 1u; group < params.groups; ++group) {
            minimum = min(minimum, bounds[2u + 2u * group].xyz);
            maximum = max(maximum, bounds[3u + 2u * group].xyz);
            invalid = max(invalid, bounds[3u + 2u * group].w);
        }
        float3 size = maximum - minimum;
        float side = max(size.x, max(size.y, size.z));
        bounds[0] = float4(minimum, side);
        bounds[1] = float4(maximum, max(invalid, !isfinite(side) ? 1.0 : 0.0));
        return;
    }
    if (index >= params.count) return;
    if (params.stage == 2u) {
        float3 position = asfloat(sourceParams[index * 15u].xyz);
        float side = bounds[0].w;
        precise float inverseSide = side > 0.0 ? 1.0 / side : 0.0;
        precise float3 normalized = (position - bounds[0].xyz) * inverseSide;
        // CUDA __saturatef maps NaN to zero, including 0 * an overflowing reciprocal.
        float3 saturated = float3(isnan(normalized.x) ? 0.0 : saturate(normalized.x),
                                  isnan(normalized.y) ? 0.0 : saturate(normalized.y),
                                  isnan(normalized.z) ? 0.0 : saturate(normalized.z));
        uint3 quantized = bounds[1].w == 0.0
            ? uint3(saturated * 2097151.0) : uint3(0u);
        uint2 code = uint2(0u);
        for (uint bit = 0u; bit < 21u; ++bit) {
            for (uint axis = 0u; axis < 3u; ++axis) {
                uint targetBit = 3u * bit + axis;
                uint value = (quantized[axis] >> bit) & 1u;
                if (targetBit < 32u) code.x |= value << targetBit;
                else code.y |= value << (targetBit - 32u);
            }
        }
        codes[index] = code;
        keys[index] = code.x;
        indices[index] = index;
    } else if (params.stage == 3u) {
        keys[index] = codes[indices[index]].y;
    } else if (params.stage == 4u && bounds[1].w == 0.0) {
        uint source = indices[index];
        for (uint word = 0u; word < 15u; ++word)
            targetParams[index * 15u + word] = sourceParams[source * 15u + word];
        for (uint word = 0u; word < 30u; ++word)
            targetAdam[index * 30u + word] = sourceAdam[source * 30u + word];
        targetStats[index] = sourceStats[source];
    }
}
