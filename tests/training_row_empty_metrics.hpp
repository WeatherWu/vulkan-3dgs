#pragma once
#include "training/core/training_types.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <numeric>
#include <span>
#include <string>
#include <vector>

struct RowGradientComparison {
    uint64_t compared = 0, differing = 0, nonFinite = 0, paddingErrors = 0, missingExtra = 0;
    double relativeL2 = 0, cosine = 1, maximum = 0;
    size_t maximumGaussian = 0, maximumComponent = 0;
    bool passed() const {
        return !differing && !nonFinite && !paddingErrors && !missingExtra;
    }
};
inline RowGradientComparison
compareRowGradients(std::span<const vulkan3DGS::ProjectedGaussianGrad> reference,
                    std::span<const vulkan3DGS::ProjectedGaussianGrad> candidate) {
    RowGradientComparison out;
    out.missingExtra = reference.size() > candidate.size() ? reference.size() - candidate.size()
                                                           : candidate.size() - reference.size();
    double normA = 0, normB = 0, squareDifference = 0, dot = 0;
    for (size_t id = 0; id < std::min(reference.size(), candidate.size()); ++id) {
        std::array<float, 12> a{}, b{};
        static_assert(sizeof(a) == sizeof(reference[id]));
        std::memcpy(a.data(), &reference[id], sizeof(a));
        std::memcpy(b.data(), &candidate[id], sizeof(b));
        for (size_t component = 0; component < a.size(); ++component) {
            ++out.compared;
            if (!std::isfinite(a[component]) || !std::isfinite(b[component])) {
                ++out.nonFinite;
                continue;
            }
            const bool padding = component == 2u || component == 3u || component == 11u;
            if (padding && (a[component] != 0 || b[component] != 0)) ++out.paddingErrors;
            const double av = a[component], bv = b[component], difference = std::abs(av - bv);
            out.differing += difference > 1e-5 + 1e-3 * std::max(std::abs(av), std::abs(bv));
            if (difference > out.maximum) {
                out.maximum = difference;
                out.maximumGaussian = id;
                out.maximumComponent = component;
            }
            if (!padding) {
                normA += av * av;
                normB += bv * bv;
                dot += av * bv;
                squareDifference += difference * difference;
            }
        }
    }
    out.relativeL2 = std::sqrt(squareDifference / std::max(normA, 1e-300));
    out.cosine = normA && normB ? dot / std::sqrt(normA * normB) : (normA == normB ? 1 : 0);
    return out;
}
inline double rowMean(const std::vector<double>& values) {
    return values.empty()
               ? 0
               : std::accumulate(values.begin(), values.end(), 0.0) / double(values.size());
}
inline std::string rowTimingVerdict(const std::vector<double>& before,
                                    const std::vector<double>& candidate,
                                    const std::vector<double>& after, bool longRun) {
    if (before.empty() || candidate.empty() || after.empty()) return "INCONCLUSIVE_NO_TIMESTAMPS";
    const double a = rowMean(before), b = rowMean(candidate), c = rowMean(after);
    if (!(a > 0 && b > 0 && c > 0) || !std::isfinite(a + b + c))
        return "INCONCLUSIVE_INVALID_TIMING";
    const double baseline = (a + c) * 0.5;
    if (std::abs(a - c) > 0.02 * std::max(a, c)) return "INCONCLUSIVE_BASELINE_DRIFT";
    if (b >= baseline) return "STOP_NO_NET_GAIN";
    if (!longRun) return "SHORT_GAIN_ONLY_REQUIRES_200_1000";
    if (baseline - b < 0.05 * baseline || baseline - b <= 2 * std::abs(a - c))
        return "STOP_GAIN_BELOW_ENGINEERING_GATE";
    return "TIMING_PASS_REQUIRES_RESOURCE_AND_MULTIVIEW_REVIEW";
}
