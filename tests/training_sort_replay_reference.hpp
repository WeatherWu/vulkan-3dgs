#pragma once

#include "training/core/training_scanline_capture.hpp"

#include <bit>
#include <algorithm>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>

namespace sort_replay {
using namespace vulkan3DGS;
inline void check(bool okay, const char* reason) {
    if (!okay) throw std::runtime_error(reason);
}
struct Header {
    std::array<char, 8> magic{'V', '3', 'D', 'S', 'R', 'T', '0', '1'};
    uint32_t version = 1, capacity = 0;
    TrainingScanlineHeader source;
};
static_assert(sizeof(Header) == 104);
struct Inputs {
    Header header;
    std::vector<ProjectedGaussian> projected;
    std::vector<glm::uvec2> gaussianRanges;
    std::vector<uint32_t> items, depthKeys, tileKeys, indices;
};
inline uint32_t depthKey(float depth) {
    const auto bits = std::bit_cast<uint32_t>(depth);
    return bits & 0x80000000u ? ~bits : bits ^ 0x80000000u;
}
inline void verify(const Inputs& raw, const TrainingScanlineCapture& source) {
    const auto& h = source.header;
    const auto& r = raw.header.source;
    check(raw.header.magic == Header{}.magic && raw.header.version == 1 &&
              std::memcmp(&h, &r, sizeof(h)) == 0 && raw.header.capacity >= h.itemCount,
          "Sort input header/capacity mismatch");
    check(raw.projected.size() == h.gaussianCount && raw.gaussianRanges.size() == h.gaussianCount &&
              raw.items.size() == h.itemCount && raw.depthKeys.size() == h.itemCount &&
              raw.tileKeys.size() == h.itemCount && raw.indices.size() == h.itemCount,
          "Sort input dimensions mismatch");
    check(raw.projected.empty() || std::memcmp(raw.projected.data(), source.projected.data(),
                      raw.projected.size() * sizeof(ProjectedGaussian)) == 0,
          "Projected inputs changed in replay");
    uint64_t cursor = 0;
    std::vector<uint32_t> order(h.itemCount);
    std::iota(order.begin(), order.end(), 0u);
    for (uint32_t g = 0; g < h.gaussianCount; ++g) {
        const auto range = raw.gaussianRanges[g];
        check(range.x == cursor && cursor + range.y <= h.itemCount,
              "Production Gaussian prefix/count mismatch");
        for (uint32_t i = 0; i < range.y; ++i) {
            const auto index = range.x + i;
            check(raw.items[index] == g && raw.indices[index] == index &&
                      raw.depthKeys[index] == depthKey(raw.projected[g].centerRadius.z) &&
                      raw.tileKeys[index] < h.tileCount,
                  "Production emit IDs/keys/index seeds mismatch");
        }
        cursor += range.y;
    }
    check(cursor == h.itemCount, "Production count differs from capture item count");
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        return raw.depthKeys[a] < raw.depthKeys[b];
    });
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        return raw.tileKeys[a] < raw.tileKeys[b];
    });
    std::vector<glm::uvec4> ranges(h.tileCount, glm::uvec4(0));
    for (uint32_t i = 0; i < h.itemCount; ++i) {
        check(raw.items[order[i]] == source.items[i], "CPU emit/sort oracle differs from capture");
        const auto tile = raw.tileKeys[order[i]];
        if (!ranges[tile].y) ranges[tile].x = i;
        ++ranges[tile].y;
    }
    check(std::memcmp(ranges.data(), source.ranges.data(), ranges.size() * sizeof(glm::uvec4)) == 0,
          "CPU emit/sort ranges differ from capture");
}
inline std::string serialize(const Inputs& raw) {
    static_assert(std::endian::native == std::endian::little);
    std::string bytes;
    auto append = [&](const auto* data, size_t count) {
        if (count) bytes.append(reinterpret_cast<const char*>(data), count * sizeof(*data));
    };
    append(&raw.header, 1);
    append(raw.projected.data(), raw.projected.size());
    append(raw.gaussianRanges.data(), raw.gaussianRanges.size());
    for (const auto* values : {&raw.items, &raw.depthKeys, &raw.tileKeys, &raw.indices})
        append(values->data(), values->size());
    return bytes;
}
} // namespace sort_replay
