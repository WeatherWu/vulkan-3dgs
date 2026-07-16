#include "gaussian_training/training_types.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

using vulkan3DGS::TrainingGaussianValidationPartial;
using vulkan3DGS::TrainingPixelValidationPartial;
using vulkan3DGS::TrainingValidationGpuResult;
using vulkan3DGS::kTrainingValidationWorkgroupSize;

struct PixelSample {
    float loss = 0.0f;
    std::array<float, 4> rendered{};
};

bool finiteRendered(const PixelSample& sample) {
    return std::all_of(sample.rendered.begin(), sample.rendered.end(), [](float value) {
        return std::isfinite(value);
    });
}

TrainingValidationGpuResult directReduction(const std::vector<PixelSample>& pixels,
                                            const std::vector<bool>& finiteGaussians) {
    TrainingValidationGpuResult result{};
    for (const auto& pixel : pixels) {
        if (std::isfinite(pixel.loss)) {
            result.lossSum += pixel.loss;
            result.maxLoss = std::max(result.maxLoss, pixel.loss);
        } else {
            ++result.invalidLossCount;
        }

        if (finiteRendered(pixel)) {
            result.alphaSum += pixel.rendered[3];
            result.luminanceSum +=
                (pixel.rendered[0] + pixel.rendered[1] + pixel.rendered[2]) / 3.0f;
            ++result.validRenderedPixelCount;
        } else {
            ++result.invalidRenderedPixelCount;
        }
    }
    for (bool finite : finiteGaussians) {
        result.nonFiniteGaussianCount += finite ? 0u : 1u;
    }
    return result;
}

TrainingValidationGpuResult groupedReduction(const std::vector<PixelSample>& pixels,
                                             const std::vector<bool>& finiteGaussians) {
    const size_t pixelPartialCount =
        (pixels.size() + kTrainingValidationWorkgroupSize - 1u) / kTrainingValidationWorkgroupSize;
    std::vector<TrainingPixelValidationPartial> pixelPartials(pixelPartialCount);
    for (size_t group = 0; group < pixelPartialCount; ++group) {
        auto& partial = pixelPartials[group];
        const size_t begin = group * kTrainingValidationWorkgroupSize;
        const size_t end = std::min(begin + kTrainingValidationWorkgroupSize, pixels.size());
        for (size_t index = begin; index < end; ++index) {
            const auto& pixel = pixels[index];
            if (std::isfinite(pixel.loss)) {
                partial.lossSum += pixel.loss;
                partial.maxLoss = std::max(partial.maxLoss, pixel.loss);
            } else {
                ++partial.invalidLossCount;
            }
            if (finiteRendered(pixel)) {
                partial.alphaSum += pixel.rendered[3];
                partial.luminanceSum +=
                    (pixel.rendered[0] + pixel.rendered[1] + pixel.rendered[2]) / 3.0f;
                ++partial.validRenderedPixelCount;
            } else {
                ++partial.invalidRenderedPixelCount;
            }
        }
    }

    const size_t gaussianPartialCount =
        (finiteGaussians.size() + kTrainingValidationWorkgroupSize - 1u) /
        kTrainingValidationWorkgroupSize;
    std::vector<TrainingGaussianValidationPartial> gaussianPartials(gaussianPartialCount);
    for (size_t group = 0; group < gaussianPartialCount; ++group) {
        const size_t begin = group * kTrainingValidationWorkgroupSize;
        const size_t end = std::min(begin + kTrainingValidationWorkgroupSize, finiteGaussians.size());
        for (size_t index = begin; index < end; ++index) {
            gaussianPartials[group].nonFiniteGaussianCount += finiteGaussians[index] ? 0u : 1u;
        }
    }

    TrainingValidationGpuResult result{};
    for (const auto& partial : pixelPartials) {
        result.lossSum += partial.lossSum;
        result.maxLoss = std::max(result.maxLoss, partial.maxLoss);
        result.alphaSum += partial.alphaSum;
        result.luminanceSum += partial.luminanceSum;
        result.invalidLossCount += partial.invalidLossCount;
        result.invalidRenderedPixelCount += partial.invalidRenderedPixelCount;
        result.validRenderedPixelCount += partial.validRenderedPixelCount;
    }
    for (const auto& partial : gaussianPartials) {
        result.nonFiniteGaussianCount += partial.nonFiniteGaussianCount;
    }
    return result;
}

bool approximatelyEqual(float lhs, float rhs) {
    const float scale = std::max({1.0f, std::abs(lhs), std::abs(rhs)});
    return std::abs(lhs - rhs) <= scale * 1e-4f;
}

} // namespace

int main() {
    constexpr size_t pixelCount = 70001;
    constexpr size_t gaussianCount = 65793;
    std::vector<PixelSample> pixels(pixelCount);
    for (size_t index = 0; index < pixels.size(); ++index) {
        const float base = static_cast<float>((index % 97u) + 1u) / 100000.0f;
        pixels[index].loss = base;
        pixels[index].rendered = {
            static_cast<float>(index % 11u) / 10.0f,
            static_cast<float>(index % 13u) / 12.0f,
            static_cast<float>(index % 17u) / 16.0f,
            static_cast<float>(index % 19u) / 18.0f,
        };
    }
    pixels[17].loss = std::numeric_limits<float>::quiet_NaN();
    pixels[65539].loss = std::numeric_limits<float>::infinity();
    pixels[255].rendered[0] = std::numeric_limits<float>::quiet_NaN();
    pixels[65536].rendered[3] = -std::numeric_limits<float>::infinity();

    std::vector<bool> finiteGaussians(gaussianCount, true);
    finiteGaussians[0] = false;
    finiteGaussians[256] = false;
    finiteGaussians.back() = false;

    const TrainingValidationGpuResult direct = directReduction(pixels, finiteGaussians);
    const TrainingValidationGpuResult grouped = groupedReduction(pixels, finiteGaussians);

    if (!approximatelyEqual(direct.lossSum, grouped.lossSum) ||
        direct.maxLoss != grouped.maxLoss ||
        !approximatelyEqual(direct.alphaSum, grouped.alphaSum) ||
        !approximatelyEqual(direct.luminanceSum, grouped.luminanceSum) ||
        direct.invalidLossCount != grouped.invalidLossCount ||
        direct.invalidRenderedPixelCount != grouped.invalidRenderedPixelCount ||
        direct.validRenderedPixelCount != grouped.validRenderedPixelCount ||
        direct.nonFiniteGaussianCount != grouped.nonFiniteGaussianCount ||
        grouped.invalidLossCount != 2u ||
        grouped.invalidRenderedPixelCount != 2u ||
        grouped.nonFiniteGaussianCount != 3u) {
        return 1;
    }
    return 0;
}
