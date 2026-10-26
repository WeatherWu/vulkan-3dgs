#pragma once
#include "training_full_sort_reference.hpp"
#include "training_row_empty_metrics.hpp"
#include <sstream>

namespace full_sort {
struct DownstreamSnapshot {
    std::vector<glm::vec4> color;
    std::vector<PixelBlendState> blends;
    std::vector<ProjectedGaussianGrad> gradients;
};
struct DownstreamComparison {
    uint64_t colorDifferent = 0, blendDifferent = 0;
    double maximumColorDifference = 0;
    RowGradientComparison gradients;
    bool passed() const {
        return !colorDifferent && !blendDifferent && gradients.passed();
    }
};
inline DownstreamComparison compareDownstream(const DownstreamSnapshot& a,
                                              const DownstreamSnapshot& b) {
    check(a.color.size() == b.color.size() && a.blends.size() == b.blends.size(),
          "Downstream image dimensions differ");
    DownstreamComparison out;
    for (size_t i = 0; i < a.color.size(); ++i) {
        out.colorDifferent += std::memcmp(&a.color[i], &b.color[i], 16) != 0;
        for (int j = 0; j < 4; ++j) {
            check(std::isfinite(a.color[i][j]) && std::isfinite(b.color[i][j]),
                  "Non-finite rendered color");
            out.maximumColorDifference =
                std::max(out.maximumColorDifference,
                         std::abs(double(a.color[i][j]) - double(b.color[i][j])));
        }
    }
    for (size_t i = 0; i < a.blends.size(); ++i) {
        for (const auto* blend : {&a.blends[i], &b.blends[i]})
            check(std::isfinite(blend->finalTransmittance) &&
                      std::isfinite(blend->accumulatedAlpha),
                  "Non-finite blend state");
        out.blendDifferent += std::memcmp(&a.blends[i], &b.blends[i], 16) != 0;
    }
    out.gradients = compareRowGradients(a.gradients, b.gradients);
    return out;
}
inline void downstreamComparisonJson(std::ostream& out, const DownstreamComparison& value) {
    const auto& g = value.gradients;
    out << "{\"passed\":" << std::boolalpha << value.passed()
        << ",\"color_differing_pixels\":" << value.colorDifferent
        << ",\"blend_differing_pixels\":" << value.blendDifferent
        << ",\"color_max_abs_difference\":" << value.maximumColorDifference
        << ",\"gradients\":{\"passed\":" << g.passed() << ",\"compared_components\":" << g.compared
        << ",\"differing_components\":" << g.differing << ",\"non_finite\":" << g.nonFinite
        << ",\"padding_errors\":" << g.paddingErrors
        << ",\"missing_extra_gaussians\":" << g.missingExtra << ",\"relative_L2\":" << g.relativeL2
        << ",\"cosine\":" << g.cosine << ",\"max_abs_difference\":" << g.maximum
        << ",\"max_gaussian\":" << g.maximumGaussian << ",\"max_component\":" << g.maximumComponent
        << "}}";
}
inline std::pair<Inputs, TrainingScanlineCapture> downstreamSynthetic(uint32_t count, bool wide) {
    auto pair = synthetic(count, wide);
    auto& [input, source] = pair;
    auto& h = input.header.source;
    h.width = 31;
    h.height = 15;
    for (uint32_t g = 0; g < count; ++g) {
        input.projected[g].centerRadius.z = float(g % 3);
        input.projected[g].color = {.1f + .1f * float(g % 5), .3f, .7f, 0.f};
        const auto range = input.gaussianRanges[g];
        for (uint32_t j = 0; j < range.y; ++j)
            input.depthKeys[range.x + j] = depthKey(input.projected[g].centerRadius.z);
    }
    const auto expected = oracle(input);
    source.header = h;
    source.projected = input.projected;
    source.items = expected.items;
    source.ranges = expected.ranges;
    source.pixelGrads.resize(uint64_t(h.width) * h.height);
    for (size_t i = 0; i < source.pixelGrads.size(); ++i)
        source.pixelGrads[i].color = {.1f * float(int(i % 7) - 3), -.2f, .3f, 0.f};
    // These fixtures have no historical blend reference; A/B/A colors and blends are GPU produced.
    return pair;
}
} // namespace full_sort
