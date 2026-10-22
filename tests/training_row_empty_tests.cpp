#include "training_row_empty_metrics.hpp"
#include "training_row_empty_reference.hpp"

#include <iostream>
#include <random>
#include <stdexcept>

using namespace vulkan3DGS;
namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void inspect(const ProjectedGaussian& g, uint32_t width, uint32_t height, uint32_t ox, uint32_t oy,
             uint32_t row, uint64_t& rejected) {
    const auto d = row_empty_reference::classify(g, width, height, ox, oy, row);
    if (d.status == 2u) return;
    const uint32_t n = std::min(16u, width - ox);
    const long double a = g.conicOpacity.x, b = g.conicOpacity.y, c = g.conicOpacity.z;
    // Analytic exact-real minimum at clamped vertex, NOT candidate tangent math.
    const long double lo = float(ox) - g.centerRadius.x;
    const long double hi = float(ox + n - 1u) - g.centerRadius.x;
    const long double dy = float(oy + row) - g.centerRadius.y;
    const auto vertex = std::clamp(-b * dy / a, lo, hi);
    const auto minimum = a * vertex * vertex + 2 * b * vertex * dy + c * dy * dy;
    require(d.lower <= minimum, "Row lower bound exceeded analytic minimum");
    const float allowance = 32.0f * std::numeric_limits<float>::epsilon() * d.magnitude;
    for (uint32_t x = 0; x < n; ++x) {
        const float dx = float(ox + x) - g.centerRadius.x;
        const float yf = float(oy + row) - g.centerRadius.y;
        const float q = g.conicOpacity.x * dx * dx + 2.0f * g.conicOpacity.y * dx * yf +
                        g.conicOpacity.z * yf * yf;
        require(d.lower <= q - allowance, "Lower bound lost worst-direction power envelope");
        if (d.status == 0u) {
            require(q >= 0.0f && q <= 160.0f, "Rejected row escaped audited power domain");
            require(!scanline_reference::contributes(g, float(ox + x), float(oy + row)),
                    "Rejected original CPU contributor");
            // Independent long-double alpha, with pessimistic existing exp/log budget.
            const long double exact = a * dx * dx + 2 * b * dx * yf + c * yf * yf;
            require(static_cast<long double>(g.conicOpacity.w) *
                            std::exp(-exact / 2 + 1.0L / 8192) <
                        static_cast<long double>(1.0f / 255.0f),
                    "Worst-direction alpha envelope failed");
        }
    }
    rejected += d.status == 0u;
}
} // namespace
int main() {
    try {
        std::vector<ProjectedGaussianGrad> zero(1);
        std::memset(zero.data(), 0, sizeof(zero[0]));
        require(compareRowGradients(zero, zero).passed(), "Identical gradients failed");
        auto changed = zero;
        changed[0].color.x = 1;
        require(!compareRowGradients(zero, changed).passed(), "Changed gradient passed");
        changed = zero;
        changed[0].centerRadius.z = 1e-9f;
        require(!compareRowGradients(zero, changed).passed(), "Nonzero padding passed");
        changed = zero;
        changed[0].color.x = std::numeric_limits<float>::quiet_NaN();
        require(!compareRowGradients(zero, changed).passed(), "Non-finite gradient passed");
        require(!compareRowGradients(zero, std::span<const ProjectedGaussianGrad>{}).passed(),
                "Missing Gaussian gradient passed");
        require(rowTimingVerdict({3.0}, {3.1}, {3.0}, false) == "STOP_NO_NET_GAIN",
                "Timing regression passed");
        require(rowTimingVerdict({3.0}, {2.9}, {3.1}, true) == "INCONCLUSIVE_BASELINE_DRIFT",
                "Timing drift passed");
        require(rowTimingVerdict({3.0}, {2.8}, {3.0}, false) == "SHORT_GAIN_ONLY_REQUIRES_200_1000",
                "Short result was promoted");
        require(rowTimingVerdict({3.0}, {2.94}, {3.0}, true) == "STOP_GAIN_BELOW_ENGINEERING_GATE",
                "Tiny gain passed");
        require(rowTimingVerdict({3.0}, {2.8}, {3.0}, true) ==
                    "TIMING_PASS_REQUIRES_RESOURCE_AND_MULTIVIEW_REVIEW",
                "Measured gain rejected");
        require(rowTimingVerdict({}, {}, {}, true) == "INCONCLUSIVE_NO_TIMESTAMPS",
                "Missing timing passed");
        ProjectedGaussian g{};
        g.centerRadius = {8, 8, 1, 3};
        g.conicOpacity = {0.2f, 0, 0.2f, 0.5f};
        require(row_empty_reference::classify(g, 16, 16, 0, 0, 0).status == 0u,
                "Golden empty row not recognized");
        require(row_empty_reference::classify(g, 16, 16, 0, 0, 8).status == 1u,
                "Center row must be kept");
        uint64_t rejected = 0;
        for (float opacity :
             {1.0f / 255.0f, std::nextafter(1.0f / 255.0f, 1.0f), 0.01f, 0.99f, 1.0f}) {
            g.centerRadius = {0, 0, 1, 3};
            g.conicOpacity = {0.01f, 0, 1, opacity};
            const float limit = -2.0f * row_empty_reference::threshold(g);
            const float boundary = (limit + 1.0f / 32768.0f) / (1.0f - 1.0f / 32768.0f);
            for (int step = -16; step <= 16; ++step) {
                float value = boundary;
                for (int s = 0; s < std::abs(step); ++s)
                    value = std::nextafter(
                        value, step < 0 ? 0.0f : std::numeric_limits<float>::infinity());
                g.conicOpacity.z = value;
                inspect(g, 1, 2, 0, 0, 1, rejected);
            }
        }
        std::mt19937 random(0x73530003u);
        std::uniform_real_distribution<float> unit(0.0f, 1.0f);
        for (uint32_t i = 0; i < 160000u; ++i) {
            const uint32_t ox = i % 4u == 0u ? 1048560u : (i % 3u) * 16u;
            const uint32_t oy = i % 4u == 0u ? 1048560u : (i % 5u) * 16u;
            const uint32_t width = ox + 1u + i % 16u, height = oy + 16u;
            g.centerRadius.x = float(ox) + unit(random) * 40.0f - 12.0f;
            g.centerRadius.y = float(oy) + unit(random) * 40.0f - 12.0f;
            g.conicOpacity.x = std::exp2(-16.0f + 32.0f * unit(random));
            g.conicOpacity.z = std::exp2(-16.0f + 32.0f * unit(random));
            // Well-conditioned, thin, near-tangent and unsupported conics.
            g.conicOpacity.y =
                (2.0f * unit(random) - 1.0f) * std::sqrt(g.conicOpacity.x * g.conicOpacity.z);
            g.conicOpacity.w = i % 7u ? 1.0f / 255.0f + unit(random) * (1.0f - 1.0f / 255.0f)
                                      : std::nextafter(1.0f / 255.0f, i % 2u ? 0.0f : 1.0f);
            if (i % 3u == 0u) {
                g.conicOpacity.x = 0.01f + unit(random) * 0.2f;
                g.conicOpacity.z = 0.01f + unit(random) * 0.2f;
                g.conicOpacity.y = 0.0f;
            }
            inspect(g, width, height, ox, oy, i % 16u, rejected);
        }
        require(rejected > 1000u, "Stress corpus did not exercise exclusions");
        g.centerRadius = {8, 8, 1, 3};
        g.conicOpacity = {0.2f, 0, 0.2f, 0.5f};
        for (float opacity : {-1.0f, 0.0f, 0.001f, 2.0f, std::numeric_limits<float>::infinity(),
                              std::numeric_limits<float>::quiet_NaN()}) {
            auto bad = g;
            bad.conicOpacity.w = opacity;
            require(row_empty_reference::classify(bad, 16, 16, 0, 0, 0).status == 2u,
                    "Invalid opacity did not fall back");
        }
        auto bad = g;
        bad.conicOpacity.y = 1;
        require(row_empty_reference::classify(bad, 16, 16, 0, 0, 0).status == 2u,
                "Non-SPD must fall back");
        bad = g;
        bad.centerRadius.x = std::numeric_limits<float>::quiet_NaN();
        require(row_empty_reference::classify(bad, 16, 16, 0, 0, 0).status == 2u,
                "Non-finite center must fall back");
        require(row_empty_reference::classify(g, 1048577, 16, 0, 0, 0).status == 2u,
                "Coordinate scope widened");
        require(row_empty_reference::classify(g, 16, 16, 16, 0, 0).status == 2u,
                "Invalid origin not guarded");
        require(row_empty_reference::classify(g, 16, 16, 0, 0, 16).status == 2u,
                "Invalid row not guarded");
        std::cout << "S3-3 R0: 160000 row cases; conservative exclusions=" << rejected
                  << "; no false drops\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
