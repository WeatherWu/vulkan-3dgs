#pragma once
#include "training_row_empty_gpu_report.hpp"
#include "training_row_empty_reference.hpp"

#include <random>

inline void runRowEmptyGpuChecks(vk::Device device, vk::PhysicalDevice physical, vk::Queue queue,
                                 uint32_t family) {
    using namespace vulkan3DGS;
    TrainingScanlineCapture data;
    auto& h = data.header;
    h.width = 509;
    h.height = 507;
    h.tileCount = 1024;
    h.gaussianCount = h.itemCount = h.tileCount;
    data.projected.resize(h.gaussianCount);
    data.items.resize(h.itemCount);
    data.ranges.resize(h.tileCount);
    data.blends.resize(size_t(h.width) * h.height);
    data.pixelGrads.resize(data.blends.size());
    std::mt19937 random(0x73530003u);
    std::uniform_real_distribution<float> unit(0.0f, 1.0f);
    uint64_t rejectedPairs = 0, emptyRows = 0, fallbackRows = 0, emptyGuarded = 0,
             emptyEligibleRows = 0;
    for (uint32_t tile = 0; tile < h.tileCount; ++tile) {
        const uint32_t ox = tile % 32u * 16u, oy = tile / 32u * 16u;
        data.items[tile] = tile;
        data.ranges[tile] = {tile, 1, 0, 0};
        auto& g = data.projected[tile];
        g.centerRadius = {float(ox) + unit(random) * 40 - 12, float(oy) + unit(random) * 40 - 12, 1,
                          3};
        g.conicOpacity = {0.01f + unit(random) * 0.2f, 0, 0.01f + unit(random) * 0.2f,
                          1.0f / 255.0f + unit(random) * (1 - 1.0f / 255.0f)};
        g.color = {64, 0.001f, 1, 0};
        if (tile % 4u == 0u)
            g.conicOpacity.y =
                (unit(random) * 2 - 1) * std::sqrt(g.conicOpacity.x * g.conicOpacity.z);
        if (tile % 19u == 0u) g.conicOpacity.w = std::nextafter(1.0f / 255.0f, 0.0f);
        if (tile % 23u == 0u) g.conicOpacity.y = 1; // Non-SPD fallback.
        if (tile % 29u == 0u) g.conicOpacity.x = std::numeric_limits<float>::denorm_min();
        if (tile % 31u == 0u) g.centerRadius.x = std::numeric_limits<float>::infinity();
        if (tile % 37u == 0u) g.conicOpacity.z = 65536.0f; // M>128 fallback.
        if (tile < 165u) {
            constexpr std::array<float, 5> opacity = {1.0f / 255.0f, 0.005f, 0.01f, 0.99f, 1.0f};
            g.centerRadius = {float(ox), float(oy), 1, 3};
            g.conicOpacity = {0.01f, 0, 1, opacity[tile / 33u]};
            const float limit = -2.0f * row_empty_reference::threshold(g);
            const float error = 1.0f / 32768.0f;
            float boundary = (limit + error * (0.01f * 225.0f + 1.0f)) / (1.0f - error);
            const int step = int(tile % 33u) - 16;
            for (int s = 0; s < std::abs(step); ++s)
                boundary = std::nextafter(boundary,
                                          step < 0 ? 0.0f : std::numeric_limits<float>::infinity());
            g.conicOpacity.z = boundary;
        }
        for (uint32_t y = 0; y < 16u && oy + y < h.height; ++y) {
            const auto d = row_empty_reference::classify(g, h.width, h.height, ox, oy, y);
            emptyRows += d.status == 0u;
            fallbackRows += d.status == 2u;
            uint32_t eligible = 0;
            for (uint32_t x = 0; x < 16u && ox + x < h.width; ++x) {
                const bool active = (x + y + tile) % 5u != 0u;
                data.blends[size_t(oy + y) * h.width + ox + x].processedCount = active ? 1u : 0u;
                eligible += active;
            }
            if (d.status == 0u) {
                rejectedPairs += eligible;
                emptyGuarded += eligible == 0u;
                emptyEligibleRows += eligible != 0u;
            }
        }
    }
    const auto result = runScanlineGpuDiagnostic(device, physical, queue, family, data, true);
    const auto oracle = scanline_reference::oracle(data);
    for (size_t i :
         {0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u, 13u, 25u, 28u, 29u, 31u})
        if (result.counts[i] != oracle[i])
            throw std::runtime_error("R0 stress census/original scalar oracle mismatch");
    if (result.counts[24] || result.counts[26] || result.counts[14] != rejectedPairs ||
        result.rowEmptyCounts[0] != emptyRows || result.rowEmptyCounts[7] != fallbackRows ||
        result.rowEmptyCounts[3] != emptyGuarded || result.rowEmptyCounts[6] != emptyEligibleRows ||
        result.rowEmptyCounts[1] != result.counts[14] || result.counts[14] != result.counts[17] ||
        result.rowEmptyCounts[5] > result.rowEmptyCounts[4] || rejectedPairs == 0u)
        throw std::runtime_error("R0 stress mask safety/classification/counter contract failed");
    std::cout << "S3-3 R0 GPU: 1024 conics, edge/guard/fallback stress; rejected=" << rejectedPairs
              << " empty-rows=" << emptyRows << " native-empty-waves=" << result.rowEmptyCounts[5]
              << '/' << result.rowEmptyCounts[4] << "; false-drops=0\n";
}
