#pragma once
#include "training/core/training_scanline_capture.hpp"

#include <algorithm>
#include <cmath>

namespace scanline_reference {
using namespace vulkan3DGS;
inline bool contributes(const ProjectedGaussian& g, float x, float y, bool* exp = nullptr) {
    const float dx = x - g.centerRadius.x, dy = y - g.centerRadius.y;
    const auto q = g.conicOpacity;
    const float power = -0.5f * (q.x * dx * dx + 2.0f * q.y * dx * dy + q.z * dy * dy);
    const bool inside = power <= 0.0f;
    if (exp) *exp = inside;
    const float coverage = inside ? std::exp(power) : 0.0f;
    return std::clamp(q.w * coverage, 0.0f, 0.99f) >= 1.0f / 255.0f;
}

// Independent scalar census of the original predicate; no candidate root formula.
inline std::array<uint64_t, 32> oracle(const TrainingScanlineCapture& data) {
    validateTrainingScanlineCapture(data);
    std::array<uint64_t, 32> out{};
    const auto& h = data.header;
    const uint32_t tilesX = (h.width + 15u) / 16u;
    for (uint32_t tile = 0; tile < h.tileCount; ++tile) {
        const auto range = data.ranges[tile];
        const uint32_t ox = tile % tilesX * 16u, oy = tile / tilesX * 16u;
        std::array<uint32_t, 256> prefixes{};
        uint32_t p = 0u;
        for (uint32_t y = 0; y < 16u && oy + y < h.height; ++y)
            for (uint32_t x = 0; x < 16u && ox + x < h.width; ++x) {
                const auto raw = data.blends[(oy + y) * h.width + ox + x].processedCount;
                prefixes[y * 16u + x] = std::min(raw, range.y);
                p = std::max(p, prefixes[y * 16u + x]);
                out[31] += raw > range.y ? 1u : 0u;
            }
        out[28] += p;
        out[29] += range.y - p;
        while (p > 0u) {
            const uint32_t n = std::min(16u, p), base = p - n;
            uint32_t batchE = 0, batchC = 0;
            for (uint32_t reverse = 0; reverse < n; ++reverse) {
                const auto offset = p - 1u - reverse;
                const auto id = data.items[range.x + offset];
                if (id >= h.gaussianCount) {
                    ++out[25];
                    continue;
                }
                uint32_t gaussianE = 0, gaussianC = 0;
                for (uint32_t y = 0; y < 16u && oy + y < h.height; ++y) {
                    uint32_t rowE = 0, rowC = 0;
                    for (uint32_t x = 0; x < 16u && ox + x < h.width; ++x) {
                        ++out[0];
                        if (offset >= prefixes[y * 16u + x]) {
                            ++out[2];
                            continue;
                        }
                        ++rowE;
                        ++out[1];
                        bool exp = false;
                        const bool c =
                            contributes(data.projected[id], float(ox + x), float(oy + y), &exp);
                        out[3] += exp ? 1u : 0u;
                        out[4] += c ? 1u : 0u;
                        rowC += c ? 1u : 0u;
                    }
                    if (rowE) {
                        ++out[5];
                        if (!rowC) {
                            ++out[6];
                            out[7] += rowE;
                        }
                    }
                    gaussianE += rowE;
                    gaussianC += rowC;
                }
                if (gaussianE) {
                    ++out[8];
                    if (!gaussianC) {
                        ++out[9];
                        out[10] += gaussianE;
                    }
                }
                batchE += gaussianE;
                batchC += gaussianC;
            }
            if (batchE) {
                ++out[11];
                if (!batchC) {
                    ++out[12];
                    out[13] += batchE;
                }
            }
            p = base;
        }
    }
    return out;
}
} // namespace scanline_reference
