#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace pixel_culling_reference {
constexpr float alphaMin = 1.0f / 255.0f;
constexpr float guard = 1.0f / 1024.0f;
constexpr float disabled = -81.0f;
using Input = std::array<float, 4>;

inline float threshold(float opacity) {
    if (!std::isfinite(opacity) || opacity < alphaMin || opacity > 1.0f) return disabled;
    return std::log(alphaMin / opacity) - guard;
}
inline bool reject(float power, float cutoff) {
    return power >= -80.0f && power < cutoff;
}
inline bool contributes(float opacity, float power) {
    const float coverage = power <= 0.0f ? std::exp(power) : 0.0f;
    return std::fmin(std::fmax(opacity * coverage, 0.0f), 0.99f) >= alphaMin;
}

inline std::vector<Input> cases() {
    std::vector<Input> result;
    const auto add = [&](float o, float p, bool pixel = true, bool prefix = true) {
        result.push_back({o, p, pixel ? 1.0f : 0.0f, prefix ? 1.0f : 0.0f});
    };
    const float inf = std::numeric_limits<float>::infinity();
    for (uint32_t i = 0; i <= 4096u; ++i) {
        const float o = alphaMin + (1.0f - alphaMin) * float(i) / 4096.0f;
        for (float center : {std::log(alphaMin / o), threshold(o), -80.0f, 0.0f}) {
            float down = center, up = center;
            add(o, center);
            for (uint32_t step = 0; step < 8u; ++step) {
                down = std::nextafter(down, -inf);
                up = std::nextafter(up, inf);
                add(o, down);
                add(o, up);
            }
        }
    }
    for (float o :
         {0.0f, -1.0f, alphaMin * 0.5f, std::nextafter(alphaMin, 0.0f), std::nextafter(1.0f, inf),
          64.0f, inf, -inf, std::numeric_limits<float>::quiet_NaN(),
          std::numeric_limits<float>::denorm_min()})
        for (float p : {-1000.0f, -81.0f, -80.0f, -6.0f, -0.001f, 0.0f, 1.0f, inf, -inf,
                        std::numeric_limits<float>::quiet_NaN()})
            add(o, p);
    uint32_t random = 0x718abc23u;
    const auto next = [&]() {
        random = random * 1664525u + 1013904223u;
        return float(random >> 8u) / 16777216.0f;
    };
    for (uint32_t i = 0; i < 100000u; ++i) {
        const float o = alphaMin + (1.0f - alphaMin) * next();
        const float p = -100.0f * next();
        add(o, p, i % 17u != 0u, i % 19u != 0u);
    }
    // Fully rejected native subgroups as well as mixed/tail groups.
    for (uint32_t i = 0; i < 257u; ++i)
        add(0.01f, -8.0f);
    return result;
}
} // namespace pixel_culling_reference
