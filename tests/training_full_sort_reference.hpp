#pragma once
#include "training_range_key_reference.hpp"
#include <limits>

namespace full_sort {
using namespace range_keys;
inline uint32_t tileBits(uint32_t tiles) {
    check(tiles > 0, "Empty tile grid");
    return tiles <= 65536u ? 16u : 32u;
}
inline std::vector<uint32_t> visibleOrder(const Inputs& input) {
    std::vector<uint32_t> ids;
    for (uint32_t g = 0; g < input.header.source.gaussianCount; ++g)
        if (input.gaussianRanges[g].y) ids.push_back(g);
    std::stable_sort(ids.begin(), ids.end(), [&](auto a, auto b) {
        return depthKey(input.projected[a].centerRadius.z) <
               depthKey(input.projected[b].centerRadius.z);
    });
    return ids;
}
// Geometry has integer centers and support contained in known tiles; wide splats cover both tiles.
inline std::pair<Inputs, TrainingScanlineCapture>
synthetic(uint32_t count, bool wide = false, uint32_t tiles = 2, bool allInactive = false) {
    Inputs input;
    auto& h = input.header.source;
    h.gaussianCount = count;
    h.width = tiles * 16;
    h.height = 16;
    h.tileCount = tiles;
    input.projected.resize(count);
    input.gaussianRanges.resize(count);
    const std::array<float, 11> depths{-0.f,
                                       0.f,
                                       1.f,
                                       1.f,
                                       -1.f,
                                       2.f,
                                       2.f,
                                       3.f,
                                       std::numeric_limits<float>::infinity(),
                                       -std::numeric_limits<float>::infinity(),
                                       std::numeric_limits<float>::quiet_NaN()};
    for (uint32_t g = 0; g < count; ++g) {
        const uint32_t tile = (g & 1u) ? tiles - 1 : 0;
        const bool active = !allInactive && (count < 3 || g % 5 != 4);
        auto& p = input.projected[g];
        p.centerRadius = {wide ? 16.f : float(tile * 16 + 8), 8.f, depths[g % depths.size()],
                          active ? (wide ? 40.f : 4.f) : 0.f};
        p.conicOpacity = {wide ? .005f : 1.f, 0.f, wide ? .005f : 1.f, .5f};
        const uint32_t n = active ? (wide ? 2u : 1u) : 0u;
        input.gaussianRanges[g] = {uint32_t(input.items.size()), n};
        for (uint32_t j = 0; j < n; ++j) {
            input.indices.push_back(uint32_t(input.items.size()));
            input.items.push_back(g);
            input.depthKeys.push_back(depthKey(p.centerRadius.z));
            input.tileKeys.push_back(wide ? j : tile);
        }
    }
    h.itemCount = uint32_t(input.items.size());
    input.header.capacity = h.itemCount + 17;
    const auto expected = oracle(input);
    TrainingScanlineCapture source;
    source.header = h;
    source.projected = input.projected;
    source.items = expected.items;
    source.ranges = expected.ranges;
    // CPU verification uses projected/list/ranges only; no image-sized dummy arrays are needed.
    return {std::move(input), std::move(source)};
}
inline void verifyChainOracle(const Inputs& input) {
    auto ids = visibleOrder(input);
    std::vector<uint32_t> emittedIDs, keys;
    for (auto g : ids) {
        const auto r = input.gaussianRanges[g];
        for (uint32_t j = 0; j < r.y; ++j) {
            emittedIDs.push_back(g);
            keys.push_back(input.tileKeys[r.x + j]);
        }
    }
    std::vector<uint32_t> order(keys.size());
    std::iota(order.begin(), order.end(), 0u);
    std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) { return keys[a] < keys[b]; });
    const auto expected = oracle(input);
    check(order.size() == expected.items.size(), "Full chain CPU item count mismatch");
    for (size_t i = 0; i < order.size(); ++i)
        check(emittedIDs[order[i]] == expected.items[i] && keys[order[i]] == expected.high[i],
              "Stable compact/depth/emit/tile chain differs from original oracle");
}
} // namespace full_sort
