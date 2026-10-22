#pragma once
#include "training_scanline_reference.hpp"

#include <limits>

namespace row_empty_reference {
// Mirrors only the candidate for CPU boundary tests. The original scalar
// sample oracle remains scanline_reference::contributes, independently.
inline constexpr float disabled = -81.0f;
inline float threshold(const vulkan3DGS::ProjectedGaussian& g) {
    const auto c = g.conicOpacity;
    constexpr float low = 1.0f / 65536.0f, high = 65536.0f;
    if (!std::isfinite(g.centerRadius.x) || !std::isfinite(g.centerRadius.y) ||
        std::abs(g.centerRadius.x) > 1048576.0f || std::abs(g.centerRadius.y) > 1048576.0f ||
        !std::isfinite(c.x) || !std::isfinite(c.y) || !std::isfinite(c.z) || !std::isfinite(c.w) ||
        c.x < low || c.x > high || c.z < low || c.z > high || std::abs(c.y) > high ||
        (c.y != 0.0f && std::abs(c.y) < low) || c.w < 1.0f / 255.0f || c.w > 1.0f)
        return disabled;
    const float ac = c.x * c.z, bb = c.y * c.y;
    if (!(ac - bb > ac / 1024.0f)) return disabled;
    return std::log((1.0f / 255.0f) / c.w) - 1.0f / 1024.0f;
}
struct Decision {
    uint32_t status = 2;
    float lower = 0, magnitude = 0;
};
inline Decision classify(const vulkan3DGS::ProjectedGaussian& g, uint32_t width, uint32_t height,
                         uint32_t ox, uint32_t oy, uint32_t row) {
    const float cutoff = threshold(g);
    if (width > 1048576u || height > 1048576u || ox >= width || oy >= height ||
        row >= height - oy || cutoff == disabled)
        return {};
    const uint32_t n = std::min(16u, width - ox);
    const float lo = float(ox) - g.centerRadius.x;
    const float hi = float(ox + n - 1u) - g.centerRadius.x;
    const float dy = float(oy + row) - g.centerRadius.y;
    const float dx = std::max(std::abs(lo), std::abs(hi));
    const auto c = g.conicOpacity;
    const float m = c.x * dx * dx + 2.0f * std::abs(c.y) * dx * std::abs(dy) + c.z * dy * dy + 1.0f;
    if (!std::isfinite(m) || m > 128.0f) return {};
    const float v = std::clamp((-c.y * dy) / c.x, lo, hi);
    const float q = c.x * v * v + 2.0f * c.y * v * dy + c.z * dy * dy;
    const float slope = 2.0f * (c.x * v + c.y * dy);
    const float tangent = q - std::abs(slope) * (hi - lo);
    const float lower = tangent - (1.0f / 32768.0f) * m;
    if (!std::isfinite(v) || !std::isfinite(lower)) return {};
    return {lower > -2.0f * cutoff ? 0u : 1u, lower, m};
}
} // namespace row_empty_reference
