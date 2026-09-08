#include "gaussian_training/training_types.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
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
    uint32_t processedCandidates = 0;
    uint32_t contributors = 0;
};

bool finiteRendered(const PixelSample& sample) {
    return std::all_of(sample.rendered.begin(), sample.rendered.end(), [](float value) {
        return std::isfinite(value);
    });
}

uint32_t processedCandidateBucket(uint32_t count) {
    if (count == 0u) return 0u;
    if (count <= 32u) return 1u;
    if (count <= 64u) return 2u;
    if (count <= 128u) return 3u;
    if (count <= 256u) return 4u;
    if (count <= 512u) return 5u;
    if (count <= 1024u) return 6u;
    return 7u;
}

void addProcessedCandidateBucket(TrainingValidationGpuResult& result, uint32_t count) {
    const uint32_t bucket = processedCandidateBucket(count);
    if (bucket < 4u) {
        ++result.processedCandidateBucketsLow[bucket];
    } else {
        ++result.processedCandidateBucketsHigh[bucket - 4u];
    }
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
        result.processedCandidateSum += static_cast<float>(pixel.processedCandidates);
        result.contributorSum += static_cast<float>(pixel.contributors);
        result.maxProcessedCandidates = std::max(result.maxProcessedCandidates,
                                                 pixel.processedCandidates);
        addProcessedCandidateBucket(result, pixel.processedCandidates);
    }
    for (size_t index = 0; index < finiteGaussians.size(); ++index) {
        if (!finiteGaussians[index]) {
            ++result.nonFiniteGaussianCount;
            ++result.nonFinitePositionCount;
            result.firstNonFiniteGaussianIndex = std::min(
                result.firstNonFiniteGaussianIndex,
                static_cast<uint32_t>(index));
        }
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
            partial.processedCandidateSum += static_cast<float>(pixel.processedCandidates);
            partial.contributorSum += static_cast<float>(pixel.contributors);
            partial.maxProcessedCandidates = std::max(partial.maxProcessedCandidates,
                                                      pixel.processedCandidates);
            const uint32_t bucket = processedCandidateBucket(pixel.processedCandidates);
            if (bucket < 4u) {
                ++partial.processedCandidateBucketsLow[bucket];
            } else {
                ++partial.processedCandidateBucketsHigh[bucket - 4u];
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
            if (!finiteGaussians[index]) {
                ++gaussianPartials[group].nonFiniteGaussianCount;
                ++gaussianPartials[group].nonFinitePositionCount;
                gaussianPartials[group].firstNonFiniteGaussianIndex = std::min(
                    gaussianPartials[group].firstNonFiniteGaussianIndex,
                    static_cast<uint32_t>(index));
            }
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
        result.processedCandidateSum += partial.processedCandidateSum;
        result.contributorSum += partial.contributorSum;
        result.maxProcessedCandidates = std::max(result.maxProcessedCandidates,
                                                 partial.maxProcessedCandidates);
        result.processedCandidateBucketsLow += partial.processedCandidateBucketsLow;
        result.processedCandidateBucketsHigh += partial.processedCandidateBucketsHigh;
    }
    for (const auto& partial : gaussianPartials) {
        result.nonFiniteGaussianCount += partial.nonFiniteGaussianCount;
        result.nonFinitePositionCount += partial.nonFinitePositionCount;
        result.nonFiniteOpacityCount += partial.nonFiniteOpacityCount;
        result.nonFiniteRawScaleCount += partial.nonFiniteRawScaleCount;
        result.nonFiniteActivatedScaleCount += partial.nonFiniteActivatedScaleCount;
        result.nonFiniteRotationCount += partial.nonFiniteRotationCount;
        result.nonFiniteSHCount += partial.nonFiniteSHCount;
        result.firstNonFiniteGaussianIndex = std::min(result.firstNonFiniteGaussianIndex,
                                                      partial.firstNonFiniteGaussianIndex);
    }
    return result;
}

bool approximatelyEqual(float lhs, float rhs) {
    const float scale = std::max({1.0f, std::abs(lhs), std::abs(rhs)});
    return std::abs(lhs - rhs) <= scale * 1e-4f;
}

bool approximatelyEqualProfileCount(float lhs, float rhs) {
    const float scale = std::max({1.0f, std::abs(lhs), std::abs(rhs)});
    return std::abs(lhs - rhs) <= scale * 1e-3f;
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
        pixels[index].processedCandidates = static_cast<uint32_t>(index % 1024u);
        pixels[index].contributors = static_cast<uint32_t>(index % 47u);
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
        direct.nonFinitePositionCount != grouped.nonFinitePositionCount ||
        !approximatelyEqualProfileCount(direct.processedCandidateSum, grouped.processedCandidateSum) ||
        !approximatelyEqualProfileCount(direct.contributorSum, grouped.contributorSum) ||
        direct.maxProcessedCandidates != grouped.maxProcessedCandidates ||
        direct.processedCandidateBucketsLow != grouped.processedCandidateBucketsLow ||
        direct.processedCandidateBucketsHigh != grouped.processedCandidateBucketsHigh ||
        direct.firstNonFiniteGaussianIndex != grouped.firstNonFiniteGaussianIndex ||
        grouped.invalidLossCount != 2u ||
        grouped.invalidRenderedPixelCount != 2u ||
        grouped.nonFiniteGaussianCount != 3u ||
        grouped.nonFinitePositionCount != 3u ||
        grouped.firstNonFiniteGaussianIndex != 0u) {
        std::cerr << "validation reduction mismatch: candidates "
                  << direct.processedCandidateSum << " / " << grouped.processedCandidateSum
                  << ", contributors " << direct.contributorSum << " / " << grouped.contributorSum
                  << ", max " << direct.maxProcessedCandidates << " / " << grouped.maxProcessedCandidates
                  << '\n';
        return 1;
    }
    const uint32_t histogramTotal = grouped.processedCandidateBucketsLow.x +
                                    grouped.processedCandidateBucketsLow.y +
                                    grouped.processedCandidateBucketsLow.z +
                                    grouped.processedCandidateBucketsLow.w +
                                    grouped.processedCandidateBucketsHigh.x +
                                    grouped.processedCandidateBucketsHigh.y +
                                    grouped.processedCandidateBucketsHigh.z +
                                    grouped.processedCandidateBucketsHigh.w;
    if (histogramTotal != pixelCount) {
        std::cerr << "validation processed-candidate histogram count mismatch\n";
        return 1;
    }
    return 0;
}
