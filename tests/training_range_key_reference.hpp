#pragma once

#include "training_sort_replay_reference.hpp"

#include <fstream>

namespace range_keys {
using namespace sort_replay;
inline Inputs readInputs(const std::filesystem::path& path) {
    static_assert(std::endian::native == std::endian::little);
    const auto bytes = std::filesystem::file_size(path);
    check(bytes >= sizeof(Header) && bytes <= (256ull << 20), "Range input exceeds file bounds");
    std::ifstream file(path, std::ios::binary);
    Inputs data;
    file.read(reinterpret_cast<char*>(&data.header), sizeof(Header));
    check(bool(file) && data.header.magic == Header{}.magic && data.header.version == 1,
          "Invalid range input header");
    const auto& h = data.header.source;
    (void)trainingScanlineCaptureBytes(h);
    check(h.gaussianCount <= 2000000 && h.itemCount <= (1u << 22) &&
              data.header.capacity >= std::max(h.itemCount, 1u) && data.header.capacity <= (1u << 24),
          "Range input workload/capacity exceeds diagnostic bounds");
    check(bytes == sizeof(Header) + uint64_t(h.gaussianCount) * 56 + uint64_t(h.itemCount) * 16,
          "Truncated/trailing range input data");
    auto read = [&](auto& values, size_t count) {
        values.resize(count);
        if (count) file.read(reinterpret_cast<char*>(values.data()),
                             static_cast<std::streamsize>(count * sizeof(values[0])));
        check(bool(file), "Truncated range input vector");
    };
    read(data.projected, h.gaussianCount);
    read(data.gaussianRanges, h.gaussianCount);
    for (auto* values : {&data.items, &data.depthKeys, &data.tileKeys, &data.indices}) read(*values, h.itemCount);
    check(file.peek() == std::char_traits<char>::eof(), "Range input changed size during read");
    return data;
}

struct Oracle {
    std::vector<uint32_t> low, order, high, items;
    std::vector<glm::uvec4> ranges;
};
inline Oracle oracle(const Inputs& data) {
    const auto& h = data.header.source;
    check(data.depthKeys.size() == h.itemCount && data.tileKeys.size() == h.itemCount &&
              data.items.size() == h.itemCount && data.indices.size() == h.itemCount,
          "Range oracle input dimensions mismatch");
    for (uint32_t i = 0; i < h.itemCount; ++i)
        check(data.indices[i] == i && data.tileKeys[i] < h.tileCount, "Invalid range input seed/key");
    Oracle out;
    out.order = data.indices;
    std::stable_sort(out.order.begin(), out.order.end(), [&](uint32_t a, uint32_t b) {
        return data.depthKeys[a] < data.depthKeys[b];
    });
    out.low = data.depthKeys;
    std::sort(out.low.begin(), out.low.end());
    std::stable_sort(out.order.begin(), out.order.end(), [&](uint32_t a, uint32_t b) {
        return data.tileKeys[a] < data.tileKeys[b];
    });
    out.ranges.resize(h.tileCount, glm::uvec4(0));
    for (uint32_t i = 0; i < h.itemCount; ++i) {
        const auto index = out.order[i], tile = data.tileKeys[index];
        out.high.push_back(tile);
        out.items.push_back(data.items[index]);
        if (!out.ranges[tile].y) out.ranges[tile].x = i;
        ++out.ranges[tile].y;
    }
    return out;
}

inline std::pair<Inputs, TrainingScanlineCapture> synthetic(uint32_t count, bool ties = false) {
    Inputs data;
    auto& h = data.header.source;
    h.gaussianCount = h.itemCount = count;
    h.width = 256; h.height = 16; h.tileCount = 16;
    data.header.capacity = count + 17;
    data.projected.resize(count);
    data.gaussianRanges.resize(count);
    data.items.resize(count); data.indices.resize(count);
    data.depthKeys.resize(count); data.tileKeys.resize(count);
    uint32_t state = 0x87654321;
    for (uint32_t i = 0; i < count; ++i) {
        state = state * 1664525u + 1013904223u;
        data.projected[i].centerRadius.z = ties ? 1.0f : float((state >> 8) & 1023u);
        data.gaussianRanges[i] = {i, 1};
        data.items[i] = data.indices[i] = i;
        data.depthKeys[i] = depthKey(data.projected[i].centerRadius.z);
        data.tileKeys[i] = ties ? 7u : state % h.tileCount;
    }
    const auto expected = oracle(data);
    TrainingScanlineCapture source;
    source.header = h; source.projected = data.projected;
    source.items = expected.items; source.ranges = expected.ranges;
    source.blends.resize(h.width * h.height); source.pixelGrads.resize(h.width * h.height);
    return {std::move(data), std::move(source)};
}
} // namespace range_keys
