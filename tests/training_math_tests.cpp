#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <utility>
#include <vector>

namespace {

bool approximatelyEqual(double lhs, double rhs, double tolerance = 1e-9) {
    const double scale = std::max({1.0, std::abs(lhs), std::abs(rhs)});
    return std::abs(lhs - rhs) <= tolerance * scale;
}

bool testConicOffDiagonalConvention() {
    constexpr double dx = 1.7;
    constexpr double dy = -0.6;
    constexpr double dLossDpower = 0.8;
    const double packedGradient = dLossDpower * (-0.5 * dx * dy);
    const double directPackedDerivative = dLossDpower * (-dx * dy);
    return approximatelyEqual(2.0 * packedGradient, directPackedDerivative);
}

double ndcToPixel(double ndc, uint32_t extent) {
    return ((ndc + 1.0) * static_cast<double>(extent) - 1.0) * 0.5;
}

std::array<int, 2> tileBounds(double center, int radius, int tileCount) {
    constexpr int tileSize = 16;
    int minimum = static_cast<int>((center - static_cast<double>(radius)) / tileSize);
    int maximumExclusive = static_cast<int>(
        (center + static_cast<double>(radius) + tileSize - 1.0) / tileSize);
    minimum = std::clamp(minimum, 0, tileCount);
    maximumExclusive = std::clamp(maximumExclusive, 0, tileCount);
    return {minimum, maximumExclusive};
}

bool testProjectionAndTileBounds() {
    if (!approximatelyEqual(ndcToPixel(-1.0, 64), -0.5) ||
        !approximatelyEqual(ndcToPixel(0.0, 64), 31.5) ||
        !approximatelyEqual(ndcToPixel(1.0, 64), 63.5)) {
        return false;
    }

    const auto bounds = tileBounds(16.0, 1, 4);
    return bounds[0] == 0 && bounds[1] == 2;
}

struct ConicGaussian2D {
    double centerX = 0.0;
    double centerY = 0.0;
    double a = 1.0;
    double b = 0.0;
    double c = 1.0;
    double opacity = 1.0;
    double radius = 1.0;
};

double conicQuadratic(const ConicGaussian2D& gaussian, double dx, double dy) {
    return gaussian.a * dx * dx + 2.0 * gaussian.b * dx * dy + gaussian.c * dy * dy;
}

double minimumConicQuadraticOnRectangle(const ConicGaussian2D& gaussian,
                                        double minimumX,
                                        double minimumY,
                                        double maximumX,
                                        double maximumY) {
    auto evaluate = [&](double x, double y) {
        return conicQuadratic(gaussian, x, y);
    };

    double minimum = evaluate(std::clamp(0.0, minimumX, maximumX),
                              std::clamp(0.0, minimumY, maximumY));
    const double yAtMinimumX = std::clamp(-gaussian.b * minimumX / gaussian.c,
                                         minimumY, maximumY);
    const double yAtMaximumX = std::clamp(-gaussian.b * maximumX / gaussian.c,
                                         minimumY, maximumY);
    minimum = std::min(minimum, evaluate(minimumX, yAtMinimumX));
    minimum = std::min(minimum, evaluate(maximumX, yAtMaximumX));

    const double xAtMinimumY = std::clamp(-gaussian.b * minimumY / gaussian.a,
                                         minimumX, maximumX);
    const double xAtMaximumY = std::clamp(-gaussian.b * maximumY / gaussian.a,
                                         minimumX, maximumX);
    minimum = std::min(minimum, evaluate(xAtMinimumY, minimumY));
    minimum = std::min(minimum, evaluate(xAtMaximumY, maximumY));
    return minimum;
}

bool gaussianMayContributeToTile(const ConicGaussian2D& gaussian,
                                 uint32_t tileX,
                                 uint32_t tileY,
                                 uint32_t width,
                                 uint32_t height) {
    constexpr uint32_t tileSize = 16;
    constexpr double alphaMinimum = 1.0 / 255.0;
    if (!(gaussian.opacity >= alphaMinimum)) {
        return false;
    }

    const double determinant = gaussian.a * gaussian.c - gaussian.b * gaussian.b;
    if (!std::isfinite(gaussian.centerX) || !std::isfinite(gaussian.centerY) ||
        !std::isfinite(gaussian.a) || !std::isfinite(gaussian.b) ||
        !std::isfinite(gaussian.c) || !std::isfinite(gaussian.opacity) ||
        !(gaussian.a > 0.0) || !(gaussian.c > 0.0) || !(determinant > 0.0)) {
        return true;
    }

    const uint32_t minimumPixelX = tileX * tileSize;
    const uint32_t minimumPixelY = tileY * tileSize;
    const uint32_t maximumPixelX = std::min(minimumPixelX + tileSize, width);
    const uint32_t maximumPixelY = std::min(minimumPixelY + tileSize, height);
    if (maximumPixelX <= minimumPixelX || maximumPixelY <= minimumPixelY) {
        return false;
    }

    const double minimumQuadratic = minimumConicQuadraticOnRectangle(
        gaussian,
        static_cast<double>(minimumPixelX) - gaussian.centerX,
        static_cast<double>(minimumPixelY) - gaussian.centerY,
        static_cast<double>(maximumPixelX - 1u) - gaussian.centerX,
        static_cast<double>(maximumPixelY - 1u) - gaussian.centerY);
    const double supportQuadratic = 2.0 * std::log(gaussian.opacity / alphaMinimum);
    const double tolerance = 1e-5 * std::max(1.0, std::abs(supportQuadratic));
    return minimumQuadratic <= supportQuadratic + tolerance;
}

struct TileBounds2D {
    int minimumX = 0;
    int minimumY = 0;
    int maximumXExclusive = 0;
    int maximumYExclusive = 0;

    bool valid() const {
        return maximumXExclusive > minimumX && maximumYExclusive > minimumY;
    }
};

TileBounds2D coarseTileBounds(const ConicGaussian2D& gaussian,
                              uint32_t width,
                              uint32_t height) {
    constexpr double tileSize = 16.0;
    const int tileCountX = static_cast<int>((width + 15u) / 16u);
    const int tileCountY = static_cast<int>((height + 15u) / 16u);
    const int radius = static_cast<int>(gaussian.radius);
    TileBounds2D bounds{};
    bounds.minimumX = std::clamp(
        static_cast<int>((gaussian.centerX - static_cast<double>(radius)) / tileSize),
        0, tileCountX);
    bounds.minimumY = std::clamp(
        static_cast<int>((gaussian.centerY - static_cast<double>(radius)) / tileSize),
        0, tileCountY);
    bounds.maximumXExclusive = std::clamp(
        static_cast<int>((gaussian.centerX + static_cast<double>(radius) + 15.0) / tileSize),
        0, tileCountX);
    bounds.maximumYExclusive = std::clamp(
        static_cast<int>((gaussian.centerY + static_cast<double>(radius) + 15.0) / tileSize),
        0, tileCountY);
    return bounds;
}

struct GaussianTileCullingData {
    TileBounds2D bounds{};
    double centerX = 0.0;
    double centerY = 0.0;
    double a = 0.0;
    double b = 0.0;
    double c = 0.0;
    double negativeBOverA = 0.0;
    double negativeBOverC = 0.0;
    double supportQuadratic = 0.0;
    double tolerance = 0.0;
    bool active = false;
    bool useExactEllipse = false;
};

GaussianTileCullingData prepareGaussianTileCulling(const ConicGaussian2D& gaussian,
                                                   uint32_t width,
                                                   uint32_t height) {
    constexpr double alphaMinimum = 1.0 / 255.0;
    constexpr double tileSize = 16.0;
    constexpr double lastPixelOffset = tileSize - 1.0;
    GaussianTileCullingData culling{};
    if (!(gaussian.opacity >= alphaMinimum) || !(gaussian.radius > 0.0) ||
        !std::isfinite(gaussian.centerX) || !std::isfinite(gaussian.centerY) ||
        !std::isfinite(gaussian.radius)) {
        return culling;
    }

    culling.bounds = coarseTileBounds(gaussian, width, height);
    if (!culling.bounds.valid()) {
        return culling;
    }
    culling.active = true;
    culling.centerX = gaussian.centerX;
    culling.centerY = gaussian.centerY;
    culling.a = gaussian.a;
    culling.b = gaussian.b;
    culling.c = gaussian.c;

    const double determinant = gaussian.a * gaussian.c - gaussian.b * gaussian.b;
    if (!std::isfinite(gaussian.a) || !std::isfinite(gaussian.b) ||
        !std::isfinite(gaussian.c) || !std::isfinite(gaussian.opacity) ||
        !(gaussian.a > 0.0) || !(gaussian.c > 0.0) || !(determinant > 0.0)) {
        return culling;
    }

    culling.useExactEllipse = true;
    culling.negativeBOverA = -gaussian.b / gaussian.a;
    culling.negativeBOverC = -gaussian.b / gaussian.c;
    culling.supportQuadratic = 2.0 * std::log(gaussian.opacity / alphaMinimum);
    culling.tolerance = 1e-5 * std::max(1.0, std::abs(culling.supportQuadratic));

    const double acceptedQuadratic = std::max(
        0.0, culling.supportQuadratic + culling.tolerance);
    const double extentX = std::sqrt(acceptedQuadratic * gaussian.c / determinant);
    const double extentY = std::sqrt(acceptedQuadratic * gaussian.a / determinant);
    if (!std::isfinite(extentX) || !std::isfinite(extentY)) {
        return culling;
    }

    const int tileCountX = static_cast<int>((width + 15u) / 16u);
    const int tileCountY = static_cast<int>((height + 15u) / 16u);
    TileBounds2D tightBounds{};
    tightBounds.minimumX = std::clamp(static_cast<int>(std::ceil(
        (gaussian.centerX - extentX - lastPixelOffset) / tileSize)), 0, tileCountX);
    tightBounds.minimumY = std::clamp(static_cast<int>(std::ceil(
        (gaussian.centerY - extentY - lastPixelOffset) / tileSize)), 0, tileCountY);
    tightBounds.maximumXExclusive = std::clamp(static_cast<int>(std::floor(
        (gaussian.centerX + extentX) / tileSize)) + 1, 0, tileCountX);
    tightBounds.maximumYExclusive = std::clamp(static_cast<int>(std::floor(
        (gaussian.centerY + extentY) / tileSize)) + 1, 0, tileCountY);

    culling.bounds.minimumX = std::max(culling.bounds.minimumX, tightBounds.minimumX);
    culling.bounds.minimumY = std::max(culling.bounds.minimumY, tightBounds.minimumY);
    culling.bounds.maximumXExclusive = std::min(
        culling.bounds.maximumXExclusive, tightBounds.maximumXExclusive);
    culling.bounds.maximumYExclusive = std::min(
        culling.bounds.maximumYExclusive, tightBounds.maximumYExclusive);
    return culling;
}

double optimizedMinimumConicQuadraticOnRectangle(const GaussianTileCullingData& culling,
                                                 double minimumX,
                                                 double minimumY,
                                                 double maximumX,
                                                 double maximumY) {
    auto evaluate = [&](double x, double y) {
        return culling.a * x * x + 2.0 * culling.b * x * y + culling.c * y * y;
    };

    const double yAtMinimumX = std::clamp(
        culling.negativeBOverC * minimumX, minimumY, maximumY);
    const double yAtMaximumX = std::clamp(
        culling.negativeBOverC * maximumX, minimumY, maximumY);
    const double xAtMinimumY = std::clamp(
        culling.negativeBOverA * minimumY, minimumX, maximumX);
    const double xAtMaximumY = std::clamp(
        culling.negativeBOverA * maximumY, minimumX, maximumX);
    return std::min({
        evaluate(minimumX, yAtMinimumX),
        evaluate(maximumX, yAtMaximumX),
        evaluate(xAtMinimumY, minimumY),
        evaluate(xAtMaximumY, maximumY),
    });
}

bool optimizedGaussianMayContributeToTile(const GaussianTileCullingData& culling,
                                          uint32_t tileX,
                                          uint32_t tileY,
                                          uint32_t width,
                                          uint32_t height) {
    if (!culling.useExactEllipse) {
        return true;
    }

    constexpr uint32_t tileSize = 16u;
    const uint32_t minimumPixelX = tileX * tileSize;
    const uint32_t minimumPixelY = tileY * tileSize;
    const uint32_t maximumPixelX = std::min(minimumPixelX + tileSize, width);
    const uint32_t maximumPixelY = std::min(minimumPixelY + tileSize, height);
    if (maximumPixelX <= minimumPixelX || maximumPixelY <= minimumPixelY) {
        return false;
    }

    const double minimumX = static_cast<double>(minimumPixelX) - culling.centerX;
    const double minimumY = static_cast<double>(minimumPixelY) - culling.centerY;
    const double maximumX = static_cast<double>(maximumPixelX - 1u) - culling.centerX;
    const double maximumY = static_cast<double>(maximumPixelY - 1u) - culling.centerY;
    if (minimumX <= 0.0 && maximumX >= 0.0 &&
        minimumY <= 0.0 && maximumY >= 0.0) {
        return true;
    }

    const double minimumQuadratic = optimizedMinimumConicQuadraticOnRectangle(
        culling, minimumX, minimumY, maximumX, maximumY);
    return !std::isfinite(minimumQuadratic) ||
           minimumQuadratic <= culling.supportQuadratic + culling.tolerance;
}

std::vector<uint32_t> acceptedTilesLegacy(const ConicGaussian2D& gaussian,
                                          uint32_t width,
                                          uint32_t height) {
    std::vector<uint32_t> accepted;
    if (!(gaussian.opacity >= 1.0 / 255.0) || !(gaussian.radius > 0.0) ||
        !std::isfinite(gaussian.centerX) || !std::isfinite(gaussian.centerY) ||
        !std::isfinite(gaussian.radius)) {
        return accepted;
    }
    const TileBounds2D bounds = coarseTileBounds(gaussian, width, height);
    const uint32_t tileCountX = (width + 15u) / 16u;
    for (int tileY = bounds.minimumY; tileY < bounds.maximumYExclusive; ++tileY) {
        for (int tileX = bounds.minimumX; tileX < bounds.maximumXExclusive; ++tileX) {
            if (gaussianMayContributeToTile(
                    gaussian, static_cast<uint32_t>(tileX), static_cast<uint32_t>(tileY),
                    width, height)) {
                accepted.push_back(static_cast<uint32_t>(tileY) * tileCountX +
                                   static_cast<uint32_t>(tileX));
            }
        }
    }
    return accepted;
}

std::vector<uint32_t> acceptedTilesOptimized(const ConicGaussian2D& gaussian,
                                             uint32_t width,
                                             uint32_t height) {
    std::vector<uint32_t> accepted;
    const GaussianTileCullingData culling = prepareGaussianTileCulling(gaussian, width, height);
    if (!culling.active || !culling.bounds.valid()) {
        return accepted;
    }
    const uint32_t tileCountX = (width + 15u) / 16u;
    for (int tileY = culling.bounds.minimumY;
         tileY < culling.bounds.maximumYExclusive; ++tileY) {
        for (int tileX = culling.bounds.minimumX;
             tileX < culling.bounds.maximumXExclusive; ++tileX) {
            if (optimizedGaussianMayContributeToTile(
                    culling, static_cast<uint32_t>(tileX), static_cast<uint32_t>(tileY),
                    width, height)) {
                accepted.push_back(static_cast<uint32_t>(tileY) * tileCountX +
                                   static_cast<uint32_t>(tileX));
            }
        }
    }
    return accepted;
}

std::vector<uint32_t> hierarchicalExclusivePrefix(const std::vector<uint32_t>& values) {
    constexpr size_t blockSize = 256u;
    std::vector<uint32_t> result(values.size(), 0u);
    std::vector<uint32_t> blockSums((values.size() + blockSize - 1u) / blockSize, 0u);
    for (size_t block = 0u; block < blockSums.size(); ++block) {
        uint32_t running = 0u;
        const size_t begin = block * blockSize;
        const size_t end = std::min(begin + blockSize, values.size());
        for (size_t index = begin; index < end; ++index) {
            result[index] = running;
            running += values[index];
        }
        blockSums[block] = running;
    }
    if (blockSums.size() > 1u) {
        const std::vector<uint32_t> blockOffsets = hierarchicalExclusivePrefix(blockSums);
        for (size_t index = 0u; index < result.size(); ++index) {
            result[index] += blockOffsets[index / blockSize];
        }
    }
    return result;
}

bool testHierarchicalExclusivePrefix() {
    for (size_t count : {0u, 1u, 255u, 256u, 257u, 65536u, 65537u, 100000u}) {
        std::vector<uint32_t> values(count);
        for (size_t index = 0u; index < count; ++index) {
            values[index] = static_cast<uint32_t>((index * 17u + 3u) % 11u);
        }
        const std::vector<uint32_t> prefix = hierarchicalExclusivePrefix(values);
        uint32_t running = 0u;
        for (size_t index = 0u; index < count; ++index) {
            if (prefix[index] != running) {
                return false;
            }
            running += values[index];
        }
    }
    return true;
}

std::vector<std::pair<uint32_t, uint32_t>> rebuildTileRanges(
    const std::vector<uint32_t>& sortedTileKeys,
    uint32_t tileCount) {
    std::vector<std::pair<uint32_t, uint32_t>> ranges(tileCount, {0u, 0u});
    for (uint32_t index = 0u; index <= sortedTileKeys.size(); ++index) {
        const uint32_t currentTile = index == sortedTileKeys.size()
            ? tileCount
            : sortedTileKeys[index];
        const uint32_t previousTile = index == 0u
            ? tileCount
            : sortedTileKeys[index - 1u];
        if (index < sortedTileKeys.size() && currentTile < tileCount &&
            (index == 0u || currentTile != previousTile)) {
            ranges[currentTile].first = index;
        }
        if (index > 0u && previousTile < tileCount &&
            (index == sortedTileKeys.size() || currentTile != previousTile)) {
            ranges[previousTile].second = index - ranges[previousTile].first;
        }
    }
    return ranges;
}

bool testSortedTileRangeRebuild() {
    const std::vector<std::pair<uint32_t, uint32_t>> empty = rebuildTileRanges({}, 4u);
    if (empty != std::vector<std::pair<uint32_t, uint32_t>>(4u, {0u, 0u})) {
        return false;
    }

    const auto single = rebuildTileRanges({2u}, 4u);
    if (single[0] != std::pair<uint32_t, uint32_t>{0u, 0u} ||
        single[1] != std::pair<uint32_t, uint32_t>{0u, 0u} ||
        single[2] != std::pair<uint32_t, uint32_t>{0u, 1u} ||
        single[3] != std::pair<uint32_t, uint32_t>{0u, 0u}) {
        return false;
    }

    const auto multiple = rebuildTileRanges({0u, 0u, 2u, 2u, 2u, 3u}, 4u);
    return multiple[0] == std::pair<uint32_t, uint32_t>{0u, 2u} &&
           multiple[1] == std::pair<uint32_t, uint32_t>{0u, 0u} &&
           multiple[2] == std::pair<uint32_t, uint32_t>{2u, 3u} &&
           multiple[3] == std::pair<uint32_t, uint32_t>{5u, 1u};
}

bool testConservativeEllipseTileCulling() {
    constexpr uint32_t width = 63;
    constexpr uint32_t height = 47;
    constexpr double alphaMinimum = 1.0 / 255.0;
    std::mt19937 random(7u);
    std::uniform_real_distribution<double> centerDistribution(-32.0, 96.0);
    std::uniform_real_distribution<double> coefficientDistribution(0.01, 2.0);
    std::uniform_real_distribution<double> angleDistribution(0.0, 6.283185307179586);
    std::uniform_real_distribution<double> opacityDistribution(0.001, 1.0);
    uint32_t culledTileCount = 0;

    for (uint32_t sample = 0; sample < 500u; ++sample) {
        const double firstEigenvalue = coefficientDistribution(random);
        const double secondEigenvalue = coefficientDistribution(random);
        const double angle = angleDistribution(random);
        const double cosine = std::cos(angle);
        const double sine = std::sin(angle);
        ConicGaussian2D gaussian{};
        gaussian.centerX = centerDistribution(random);
        gaussian.centerY = centerDistribution(random);
        gaussian.a = firstEigenvalue * cosine * cosine + secondEigenvalue * sine * sine;
        gaussian.b = (firstEigenvalue - secondEigenvalue) * cosine * sine;
        gaussian.c = firstEigenvalue * sine * sine + secondEigenvalue * cosine * cosine;
        gaussian.opacity = opacityDistribution(random);

        for (uint32_t tileY = 0; tileY < (height + 15u) / 16u; ++tileY) {
            for (uint32_t tileX = 0; tileX < (width + 15u) / 16u; ++tileX) {
                if (gaussianMayContributeToTile(gaussian, tileX, tileY, width, height)) {
                    continue;
                }
                ++culledTileCount;

                const uint32_t minimumX = tileX * 16u;
                const uint32_t minimumY = tileY * 16u;
                const uint32_t maximumX = std::min(minimumX + 16u, width);
                const uint32_t maximumY = std::min(minimumY + 16u, height);
                for (uint32_t y = minimumY; y < maximumY; ++y) {
                    for (uint32_t x = minimumX; x < maximumX; ++x) {
                        const double quadratic = conicQuadratic(
                            gaussian,
                            static_cast<double>(x) - gaussian.centerX,
                            static_cast<double>(y) - gaussian.centerY);
                        const double alpha = gaussian.opacity * std::exp(-0.5 * quadratic);
                        if (alpha >= alphaMinimum) {
                            return false;
                        }
                    }
                }
            }
        }
    }
    return culledTileCount > 0u;
}

bool testOptimizedEllipseTileCullingMatchesLegacy() {
    constexpr uint32_t width = 63u;
    constexpr uint32_t height = 47u;
    constexpr double alphaMinimum = 1.0 / 255.0;
    std::vector<ConicGaussian2D> edgeCases = {
        {31.5, 23.5, 1.0, 0.0, 1.0, 1.0, 8.0},
        {16.0, 16.0, 0.04, 0.0, 2.0, 0.8, 15.0},
        {32.0, 15.0, 0.8, 0.77, 0.8, 0.6, 14.0},
        {61.5, 45.5, 0.2, -0.12, 1.4, 0.9, 12.0},
        {20.0, 20.0, 1.0, 0.0, 1.0, alphaMinimum, 4.0},
        {20.0, 20.0, 1.0, 0.0, 1.0, alphaMinimum * 0.5, 4.0},
        {24.0, 24.0, -1.0, 0.0, 1.0, 0.8, 8.0},
        {24.0, 24.0, 1.0, 1.0, 1.0, 0.8, 8.0},
        {24.0, 24.0, 1.0, std::numeric_limits<double>::quiet_NaN(), 1.0, 0.8, 8.0},
        {24.0, 24.0, std::numeric_limits<double>::infinity(), 0.0, 1.0, 0.8, 8.0},
        {24.0, 24.0, 1.0, 0.0, 1.0, std::numeric_limits<double>::infinity(), 8.0},
    };

    uint64_t coarseCandidateCount = 0u;
    uint64_t optimizedCandidateCount = 0u;
    auto compare = [&](const ConicGaussian2D& gaussian) {
        const std::vector<uint32_t> legacy = acceptedTilesLegacy(gaussian, width, height);
        const std::vector<uint32_t> optimized = acceptedTilesOptimized(gaussian, width, height);
        if (legacy != optimized) {
            return false;
        }
        if (gaussian.opacity >= alphaMinimum && gaussian.radius > 0.0 &&
            std::isfinite(gaussian.centerX) && std::isfinite(gaussian.centerY) &&
            std::isfinite(gaussian.radius)) {
            const TileBounds2D coarse = coarseTileBounds(gaussian, width, height);
            const GaussianTileCullingData culling = prepareGaussianTileCulling(
                gaussian, width, height);
            if (coarse.valid()) {
                coarseCandidateCount += static_cast<uint64_t>(
                    coarse.maximumXExclusive - coarse.minimumX) *
                    static_cast<uint64_t>(coarse.maximumYExclusive - coarse.minimumY);
            }
            if (culling.active && culling.bounds.valid()) {
                optimizedCandidateCount += static_cast<uint64_t>(
                    culling.bounds.maximumXExclusive - culling.bounds.minimumX) *
                    static_cast<uint64_t>(
                        culling.bounds.maximumYExclusive - culling.bounds.minimumY);
            }
        }
        return true;
    };

    for (const ConicGaussian2D& gaussian : edgeCases) {
        if (!compare(gaussian)) {
            return false;
        }
    }

    std::mt19937 random(29u);
    std::uniform_real_distribution<double> centerXDistribution(-24.0, 88.0);
    std::uniform_real_distribution<double> centerYDistribution(-24.0, 72.0);
    std::uniform_real_distribution<double> eigenvalueDistribution(0.01, 2.0);
    std::uniform_real_distribution<double> angleDistribution(0.0, 6.283185307179586);
    std::uniform_real_distribution<double> opacityDistribution(0.001, 1.0);
    for (uint32_t sample = 0u; sample < 2000u; ++sample) {
        const double firstEigenvalue = eigenvalueDistribution(random);
        const double secondEigenvalue = eigenvalueDistribution(random);
        const double angle = angleDistribution(random);
        const double cosine = std::cos(angle);
        const double sine = std::sin(angle);
        ConicGaussian2D gaussian{};
        gaussian.centerX = centerXDistribution(random);
        gaussian.centerY = centerYDistribution(random);
        gaussian.a = firstEigenvalue * cosine * cosine + secondEigenvalue * sine * sine;
        gaussian.b = (firstEigenvalue - secondEigenvalue) * cosine * sine;
        gaussian.c = firstEigenvalue * sine * sine + secondEigenvalue * cosine * cosine;
        gaussian.opacity = opacityDistribution(random);
        gaussian.radius = std::ceil(3.0 / std::sqrt(
            std::min(firstEigenvalue, secondEigenvalue)));
        if (!compare(gaussian)) {
            return false;
        }
    }

    return optimizedCandidateCount < coarseCandidateCount;
}

bool useDirectPixelTo2DGS(const std::vector<uint32_t>& processedCounts,
                          uint32_t laneCount,
                          double minimumSubgroupUtilization) {
    uint64_t sum = 0u;
    uint32_t maximum = 0u;
    for (uint32_t count : processedCounts) {
        sum += count;
        maximum = std::max(maximum, count);
    }
    if (maximum <= laneCount) {
        return true;
    }
    const double capacity = static_cast<double>(processedCounts.size()) * maximum;
    const double utilization = capacity > 0.0 ? static_cast<double>(sum) / capacity : 1.0;
    return utilization < minimumSubgroupUtilization;
}

bool testAdaptivePixelTo2DGSSelection() {
    constexpr uint32_t laneCount = 32u;
    if (!useDirectPixelTo2DGS(std::vector<uint32_t>(laneCount, 16u), laneCount, 0.5)) {
        return false;
    }
    if (useDirectPixelTo2DGS(std::vector<uint32_t>(laneCount, 128u), laneCount, 0.5)) {
        return false;
    }
    if (useDirectPixelTo2DGS(std::vector<uint32_t>(8u, 128u), laneCount, 0.5)) {
        return false;
    }

    std::vector<uint32_t> sparseLongTail(laneCount, 16u);
    sparseLongTail[0] = 1024u;
    if (!useDirectPixelTo2DGS(sparseLongTail, laneCount, 0.5)) {
        return false;
    }
    return !useDirectPixelTo2DGS(sparseLongTail, laneCount, 0.0);
}

bool testPackedConicOffDiagonalGradient() {
    const double a = 0.7;
    const double b = -0.23;
    const double c = 1.1;
    const double dx = 2.4;
    const double dy = -1.7;
    const double upstream = 0.83;
    const double epsilon = 1e-6;
    auto loss = [&](double conicB) {
        const double power = -0.5 *
            (a * dx * dx + 2.0 * conicB * dx * dy + c * dy * dy);
        return upstream * power;
    };

    const double finiteDifference = (loss(b + epsilon) - loss(b - epsilon)) / (2.0 * epsilon);
    const double packedAnalytical = upstream * (-0.5 * dx * dy);
    return std::abs(finiteDifference - 2.0 * packedAnalytical) < 1e-8;
}

double positionLearningRate(double iteration,
                            double initial,
                            double final,
                            double maxSteps) {
    const double t = std::clamp(iteration / maxSteps, 0.0, 1.0);
    return std::exp((1.0 - t) * std::log(initial) + t * std::log(final));
}

double adamUpdate(double& firstMoment,
                  double& secondMoment,
                  double gradient,
                  uint32_t optimizerStep,
                  double learningRate) {
    constexpr double beta1 = 0.9;
    constexpr double beta2 = 0.999;
    constexpr double epsilon = 1e-15;
    firstMoment = beta1 * firstMoment + (1.0 - beta1) * gradient;
    secondMoment = beta2 * secondMoment + (1.0 - beta2) * gradient * gradient;
    const double correctedFirst = firstMoment / (1.0 - std::pow(beta1, optimizerStep));
    const double correctedSecond = secondMoment / (1.0 - std::pow(beta2, optimizerStep));
    return learningRate * correctedFirst / (std::sqrt(correctedSecond) + epsilon);
}

bool testLearningRateAndAdamSteps() {
    const double firstIterationLr = positionLearningRate(1.0, 1.6e-4, 1.6e-6, 30000.0);
    const double finalIterationLr = positionLearningRate(30000.0, 1.6e-4, 1.6e-6, 30000.0);
    if (!(firstIterationLr < 1.6e-4 && firstIterationLr > finalIterationLr) ||
        !approximatelyEqual(finalIterationLr, 1.6e-6, 1e-12)) {
        return false;
    }

    uint32_t optimizerStep = 0;
    ++optimizerStep;
    const uint32_t stepBeforeDensification = optimizerStep;
    const bool optimizerEnabledOnDensification = false;
    if (optimizerEnabledOnDensification) {
        ++optimizerStep;
    }
    ++optimizerStep;
    if (stepBeforeDensification != 1u || optimizerStep != 2u) {
        return false;
    }

    double firstMoment = 0.0;
    double secondMoment = 0.0;
    const double firstUpdate = adamUpdate(firstMoment, secondMoment, 2.0, 1u, 0.1);
    return approximatelyEqual(firstUpdate, 0.1, 1e-12);
}

bool testDensificationDenominator() {
    constexpr std::array<double, 3> visibleGradientMagnitudes = {2.0, 0.0, 6.0};
    double sum = 0.0;
    uint32_t count = 0;
    for (double magnitude : visibleGradientMagnitudes) {
        sum += magnitude;
        ++count;
    }
    return count == 3u && approximatelyEqual(sum / count, 8.0 / 3.0);
}

using Quaternion = std::array<double, 4>;

double dot(const Quaternion& lhs, const Quaternion& rhs) {
    double result = 0.0;
    for (size_t i = 0; i < lhs.size(); ++i) {
        result += lhs[i] * rhs[i];
    }
    return result;
}

Quaternion normalize(const Quaternion& value) {
    const double length = std::sqrt(dot(value, value));
    Quaternion result{};
    for (size_t i = 0; i < value.size(); ++i) {
        result[i] = value[i] / std::max(length, 1e-12);
    }
    return result;
}

Quaternion backwardNormalize(const Quaternion& value, const Quaternion& upstream) {
    const double length = std::sqrt(dot(value, value));
    if (length <= 1e-12) {
        Quaternion result{};
        for (size_t i = 0; i < value.size(); ++i) {
            result[i] = upstream[i] / 1e-12;
        }
        return result;
    }

    const Quaternion normalized = normalize(value);
    const double projectedGradient = dot(normalized, upstream);
    Quaternion result{};
    for (size_t i = 0; i < value.size(); ++i) {
        result[i] = (upstream[i] - normalized[i] * projectedGradient) / length;
    }
    return result;
}

bool testQuaternionNormalizationBackward() {
    constexpr Quaternion value = {0.2, -0.3, 0.5, 0.7};
    constexpr Quaternion upstream = {-0.4, 0.8, 0.1, -0.2};
    constexpr double epsilon = 1e-6;
    const Quaternion analytic = backwardNormalize(value, upstream);
    for (size_t component = 0; component < value.size(); ++component) {
        Quaternion plus = value;
        Quaternion minus = value;
        plus[component] += epsilon;
        minus[component] -= epsilon;
        const double finiteDifference =
            (dot(normalize(plus), upstream) - dot(normalize(minus), upstream)) /
            (2.0 * epsilon);
        if (!approximatelyEqual(analytic[component], finiteDifference, 1e-7)) {
            return false;
        }
    }
    return true;
}

} // namespace

int main() {
    if (!testConicOffDiagonalConvention()) {
        std::cerr << "conic off-diagonal convention test failed\n";
        return 1;
    }
    if (!testProjectionAndTileBounds()) {
        std::cerr << "projection/tile bounds test failed\n";
        return 1;
    }
    if (!testConservativeEllipseTileCulling()) {
        std::cerr << "conservative ellipse/tile culling test failed\n";
        return 1;
    }
    if (!testOptimizedEllipseTileCullingMatchesLegacy()) {
        std::cerr << "optimized ellipse/tile culling equivalence test failed\n";
        return 1;
    }
    if (!testAdaptivePixelTo2DGSSelection()) {
        std::cerr << "adaptive pixel-to-2DGS selection test failed\n";
        return 1;
    }
    if (!testHierarchicalExclusivePrefix()) {
        std::cerr << "hierarchical exclusive-prefix test failed\n";
        return 1;
    }
    if (!testSortedTileRangeRebuild()) {
        std::cerr << "sorted tile-range rebuild test failed\n";
        return 1;
    }
    if (!testPackedConicOffDiagonalGradient()) {
        std::cerr << "packed conic off-diagonal gradient test failed\n";
        return 1;
    }
    if (!testLearningRateAndAdamSteps()) {
        std::cerr << "learning-rate/Adam test failed\n";
        return 1;
    }
    if (!testDensificationDenominator()) {
        std::cerr << "densification denominator test failed\n";
        return 1;
    }
    if (!testQuaternionNormalizationBackward()) {
        std::cerr << "quaternion normalization backward test failed\n";
        return 1;
    }

    std::cout << "training math tests passed\n";
    return 0;
}
