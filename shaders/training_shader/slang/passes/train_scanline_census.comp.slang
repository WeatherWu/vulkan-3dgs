#include "../common/train_forward_common.slang"
#include "../common/train_scanline_diagnostic.slang"
#ifndef TRAIN_SCANLINE_ROW_EMPTY_R0
#define TRAIN_SCANLINE_ROW_EMPTY_R0 0
#endif
#if TRAIN_SCANLINE_ROW_EMPTY_R0
#include "../common/train_row_empty.slang"
static const uint SCANLINE_COUNTER_COUNT = 40u;
#else
static const uint SCANLINE_COUNTER_COUNT = 32u;
#endif
#ifndef TRAIN_SCANLINE_COST_ONLY
#define TRAIN_SCANLINE_COST_ONLY 0
#endif

[[vk::binding(3, 0)]] StructuredBuffer<ProjectedGaussian> projected;
[[vk::binding(4, 0)]] StructuredBuffer<uint> items;
[[vk::binding(5, 0)]] StructuredBuffer<uint4> ranges;
[[vk::binding(14, 0)]] StructuredBuffer<PixelBlendState> blends;
struct ScanlineTileCounters { uint values[SCANLINE_COUNTER_COUNT]; };
[[vk::binding(51, 0)]] RWStructuredBuffer<ScanlineTileCounters> output;

groupshared uint prefixes[256];
groupshared uint flags[256];
groupshared uint maximumPrefix;
groupshared uint counters[SCANLINE_COUNTER_COUNT];
groupshared ProjectedGaussian gaussians[16];
groupshared uint ids[16];
groupshared float4 coefficients[16];
groupshared uint spans[256];
groupshared uint4 rowCounts[16];
groupshared uint rowExpRejected[16];
groupshared uint batchEligible, batchContributing, batchRejected;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 tid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint rank = tid.y * 16u + tid.x;
    uint2 origin = gid.xy * 16u;
    uint2 pixel = origin + tid.xy;
    uint tile = gid.y * ((params.width + 15u) / 16u) + gid.x;
    bool valid = pixel.x < params.width && pixel.y < params.height;
    uint4 rawRange = ranges[tile];
    uint start = min(rawRange.x, params.tileItemCount);
    uint count = min(rawRange.y, params.tileItemCount - start);
    uint rawPrefix = 0u;
    if (valid) rawPrefix = blends[pixel.y * params.width + pixel.x].processedCount;
    prefixes[rank] = min(rawPrefix, count);
    flags[rank] = rawPrefix > count ? 1u : 0u;
    if (rank < SCANLINE_COUNTER_COUNT) counters[rank] = 0u;
    GroupMemoryBarrierWithGroupSync();
    if (rank == 0u) {
        uint p = 0u, clamped = 0u;
        for (uint i = 0u; i < 256u; ++i) { p = max(p, prefixes[i]); clamped += flags[i]; }
        maximumPrefix = p;
        counters[28] = p; counters[29] = count - p; counters[31] = clamped;
        counters[26] = rawRange.x != start || rawRange.y != count ? 1u : 0u;
    }
    GroupMemoryBarrierWithGroupSync();
    uint remaining = maximumPrefix;
    uint checksum = 0u;
    while (remaining > 0u) {
        uint n = min(16u, remaining), base = remaining - n;
        if (rank < n) {
            uint id = items[start + base + rank]; ids[rank] = id;
            ProjectedGaussian g;
            g.centerRadius = g.conicOpacity = g.color = float4(0.0);
            if (id < params.gaussianCount) g = projected[id];
            gaussians[rank] = g;
#if TRAIN_SCANLINE_ROW_EMPTY_R0
            coefficients[rank] = float4(0.0, 0.0, 0.0, trainRowEmptyThreshold(g));
#else
            coefficients[rank] = trainScanlinePrepare(g);
#endif
        }
        if (rank == 0u) { batchEligible = 0u; batchContributing = 0u; batchRejected = 0u; }
        GroupMemoryBarrierWithGroupSync();
        uint gaussianRank = rank / 16u, row = rank % 16u;
        if (gaussianRank < n) {
#if TRAIN_SCANLINE_ROW_EMPTY_R0
            float lowerBound;
            uint decision = trainRowEmptyClassify(gaussians[gaussianRank], coefficients[gaussianRank].w, origin, row, lowerBound);
            uint width = min(16u, params.width - origin.x);
            uint span = trainScanlinePack(0u, decision == 0u ? 0u : width, decision == 2u);
#else
            uint span = trainScanlineSpan(gaussians[gaussianRank], coefficients[gaussianRank], origin, row);
#endif
            spans[rank] = span;
            checksum += span * (row + 1u) + ids[gaussianRank];
        }
        GroupMemoryBarrierWithGroupSync();
#if !TRAIN_SCANLINE_COST_ONLY
        for (uint reverse = 0u; reverse < n; ++reverse) {
            uint local = n - 1u - reverse;
            if (ids[local] >= params.gaussianCount) {
                if (rank == 0u) ++counters[25];
                continue;
            }
            uint span = spans[local * 16u + tid.y];
            bool eligible = valid && base + local < prefixes[rank];
            bool oldExp = false, contributor = false;
            if (eligible) {
                GaussianSample sample = evaluateGaussianSample(gaussians[local], pixelCenter(pixel));
                oldExp = isInsideGaussianSupport(sample.power);
                contributor = contributesToPixel(sample);
            }
            bool rejected = eligible && (tid.x < (span & 31u) || tid.x >= ((span >> 5u) & 31u));
#if TRAIN_SCANLINE_ROW_EMPTY_R0
            // Actual native lane membership, not assumed two-row warps.
            bool waveEligible = WaveActiveAnyTrue(eligible);
            bool waveEmpty = WaveActiveAllTrue(!eligible || rejected);
            if (WaveIsFirstLane() && waveEligible) {
                InterlockedAdd(counters[36], 1u);
                if (waveEmpty) InterlockedAdd(counters[37], 1u);
            }
#endif
            flags[rank] = (eligible ? 1u : 0u) | (oldExp ? 2u : 0u) |
                (contributor ? 4u : 0u) | (rejected ? 8u : 0u) |
                (oldExp && rejected ? 16u : 0u);
            GroupMemoryBarrierWithGroupSync();
            if (tid.x == 0u) {
                uint4 sum = uint4(0u); uint expRejected = 0u;
                for (uint x = 0u; x < 16u; ++x) {
                    uint f = flags[tid.y * 16u + x];
                    sum += uint4(f & 1u, (f >> 1u) & 1u, (f >> 2u) & 1u, (f >> 3u) & 1u);
                    expRejected += (f >> 4u) & 1u;
                    if ((f & 12u) == 12u) expRejected += 65536u; // Exact false-drop count, packed separately below.
                }
                rowCounts[tid.y] = sum; rowExpRejected[tid.y] = expRejected;
            }
            GroupMemoryBarrierWithGroupSync();
            if (rank == 0u) {
                uint4 total = uint4(0u);
                for (uint y = 0u; y < 16u; ++y) {
                    uint4 r = rowCounts[y]; total += r;
                    uint s = spans[local * 16u + y];
#if TRAIN_SCANLINE_ROW_EMPTY_R0
                    bool geometricEmpty = ((s >> 5u) & 31u) == (s & 31u);
                    if (origin.y + y < params.height) {
                        if (geometricEmpty) ++counters[32];
                        if ((s & 1024u) != 0u) ++counters[39];
                        if (geometricEmpty && r.x == 0u) ++counters[35];
                    }
                    if (geometricEmpty) counters[33] += r.x;
                    if (r.x > 0u && r.z == 0u && !geometricEmpty) ++counters[34];
                    if (geometricEmpty && r.x > 0u) ++counters[38];
#endif
                    counters[15] += rowExpRejected[y] & 65535u;
                    counters[24] += rowExpRejected[y] >> 16u;
                    if (origin.y + y < params.height) { ++counters[23]; counters[30] += ((s >> 5u) & 31u) - (s & 31u); }
                    if (r.x > 0u) {
                        ++counters[5];
                        if (r.z == 0u) { ++counters[6]; counters[7] += r.x; }
                        if (r.w == r.x) { ++counters[16]; counters[17] += r.x; }
                        if ((s & 1024u) != 0u) ++counters[22];
                    }
                }
                uint validPixels = min(16u, params.width - origin.x) * min(16u, params.height - origin.y);
                counters[0] += validPixels; counters[1] += total.x; counters[2] += validPixels - total.x;
                counters[3] += total.y; counters[4] += total.z; counters[14] += total.w;
                if (total.x > 0u) {
                    ++counters[8];
                    if (total.z == 0u) { ++counters[9]; counters[10] += total.x; }
                    if (total.w == total.x) { ++counters[18]; counters[19] += total.x; }
                }
                batchEligible += total.x; batchContributing += total.z; batchRejected += total.w;
            }
            GroupMemoryBarrierWithGroupSync();
        }
        if (rank == 0u && batchEligible > 0u) {
            ++counters[11];
            if (batchContributing == 0u) { ++counters[12]; counters[13] += batchEligible; }
            if (batchRejected == batchEligible) { ++counters[20]; counters[21] += batchEligible; }
        }
#endif
        remaining = base;
    }
    flags[rank] = checksum;
    GroupMemoryBarrierWithGroupSync();
    if (rank == 0u) {
        for (uint i = 0u; i < 256u; ++i) counters[27] += flags[i];
        ScanlineTileCounters result;
        for (uint i = 0u; i < SCANLINE_COUNTER_COUNT; ++i) result.values[i] = counters[i];
        output[tile] = result;
    }
}
