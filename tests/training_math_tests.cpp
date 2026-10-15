#include "training/core/training_tile_layout.hpp"

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

using namespace vulkan3DGS;

uint32_t activeTileSize = 16u;

uint32_t tileCountForExtent(uint32_t extent) {
    return (extent + activeTileSize - 1u) / activeTileSize;
}

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
    const int tileSize = static_cast<int>(activeTileSize);
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

    const int count = static_cast<int>(tileCountForExtent(64u));
    const auto bounds = tileBounds(16.0, 1, count);
    return activeTileSize == 8u
        ? bounds[0] == 1 && bounds[1] == 3
        : bounds[0] == 0 && bounds[1] == 2;
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
    const uint32_t tileSize = activeTileSize;
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
    const double tileSize = static_cast<double>(activeTileSize);
    const int tileCountX = static_cast<int>(tileCountForExtent(width));
    const int tileCountY = static_cast<int>(tileCountForExtent(height));
    const int radius = static_cast<int>(gaussian.radius);
    TileBounds2D bounds{};
    bounds.minimumX = std::clamp(
        static_cast<int>((gaussian.centerX - static_cast<double>(radius)) / tileSize),
        0, tileCountX);
    bounds.minimumY = std::clamp(
        static_cast<int>((gaussian.centerY - static_cast<double>(radius)) / tileSize),
        0, tileCountY);
    bounds.maximumXExclusive = std::clamp(
        static_cast<int>((gaussian.centerX + static_cast<double>(radius) + tileSize - 1.0) / tileSize),
        0, tileCountX);
    bounds.maximumYExclusive = std::clamp(
        static_cast<int>((gaussian.centerY + static_cast<double>(radius) + tileSize - 1.0) / tileSize),
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
    const double tileSize = static_cast<double>(activeTileSize);
    const double lastPixelOffset = tileSize - 1.0;
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

    const int tileCountX = static_cast<int>(tileCountForExtent(width));
    const int tileCountY = static_cast<int>(tileCountForExtent(height));
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

    const uint32_t tileSize = activeTileSize;
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
    const uint32_t tileCountX = tileCountForExtent(width);
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
    const uint32_t tileCountX = tileCountForExtent(width);
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

std::pair<int, int> scanlineTileRange(const GaussianTileCullingData& culling,
                                      int stripTile,
                                      bool scanRows,
                                      uint32_t width,
                                      uint32_t height) {
    const int tileSize = static_cast<int>(activeTileSize);
    const int boundMinimum = scanRows
        ? culling.bounds.minimumX
        : culling.bounds.minimumY;
    const int boundMaximum = scanRows
        ? culling.bounds.maximumXExclusive
        : culling.bounds.maximumYExclusive;
    if (!culling.useExactEllipse) {
        return {boundMinimum, boundMaximum};
    }

    const uint32_t imageExtent = scanRows ? height : width;
    const uint32_t minimumPixel = static_cast<uint32_t>(stripTile) * tileSize;
    const uint32_t maximumPixelExclusive = std::min(
        minimumPixel + tileSize, imageExtent);
    const double stripCenter = scanRows ? culling.centerY : culling.centerX;
    const double outputCenter = scanRows ? culling.centerX : culling.centerY;
    const double stripMinimum = static_cast<double>(minimumPixel) - stripCenter;
    const double stripMaximum = static_cast<double>(maximumPixelExclusive - 1u) - stripCenter;

    const double a = scanRows ? culling.a : culling.c;
    const double b = culling.b;
    const double c = scanRows ? culling.c : culling.a;
    const double determinant = a * c - b * b;
    const double acceptedQuadratic = culling.supportQuadratic + culling.tolerance;
    const double stripExtent = std::sqrt(acceptedQuadratic * a / determinant);
    if (stripMaximum < -stripExtent || stripMinimum > stripExtent) {
        return {boundMinimum, boundMinimum};
    }
    const double stripExtremum = (-b / c) *
        std::sqrt(c * acceptedQuadratic / determinant);
    const double minimumAt = std::clamp(
        -stripExtremum, stripMinimum, stripMaximum);
    const double maximumAt = std::clamp(
        stripExtremum, stripMinimum, stripMaximum);
    const double minimumRoot = std::sqrt(std::max(
        0.0, a * acceptedQuadratic - determinant * minimumAt * minimumAt));
    const double maximumRoot = std::sqrt(std::max(
        0.0, a * acceptedQuadratic - determinant * maximumAt * maximumAt));
    const double rangeMinimum = (-b * minimumAt - minimumRoot) / a;
    const double rangeMaximum = (-b * maximumAt + maximumRoot) / a;

    int minimumTile = static_cast<int>(std::ceil(
        (rangeMinimum + outputCenter - (tileSize - 1.0)) / tileSize));
    int maximumTileExclusive = static_cast<int>(std::floor(
        (rangeMaximum + outputCenter) / tileSize)) + 1;
    minimumTile = std::clamp(minimumTile, boundMinimum, boundMaximum);
    const uint32_t outputImageExtent = scanRows ? width : height;
    if (minimumTile < boundMaximum) {
        const uint32_t lastPixel = std::min(
            (static_cast<uint32_t>(minimumTile) + 1u) * tileSize,
            outputImageExtent) - 1u;
        if (static_cast<double>(lastPixel) < rangeMinimum + outputCenter) {
            ++minimumTile;
        }
    }
    maximumTileExclusive = std::clamp(
        maximumTileExclusive, minimumTile, boundMaximum);
    return {minimumTile, maximumTileExclusive};
}

std::vector<uint32_t> acceptedTilesScanline(const ConicGaussian2D& gaussian,
                                            uint32_t width,
                                            uint32_t height) {
    std::vector<uint32_t> accepted;
    const GaussianTileCullingData culling = prepareGaussianTileCulling(
        gaussian, width, height);
    if (!culling.active || !culling.bounds.valid()) {
        return accepted;
    }

    const int boundWidth = culling.bounds.maximumXExclusive - culling.bounds.minimumX;
    const int boundHeight = culling.bounds.maximumYExclusive - culling.bounds.minimumY;
    const bool scanRows = boundHeight <= boundWidth;
    const int stripMinimum = scanRows
        ? culling.bounds.minimumY
        : culling.bounds.minimumX;
    const int stripMaximum = scanRows
        ? culling.bounds.maximumYExclusive
        : culling.bounds.maximumXExclusive;
    const uint32_t tileCountX = tileCountForExtent(width);
    for (int stripTile = stripMinimum; stripTile < stripMaximum; ++stripTile) {
        const auto range = scanlineTileRange(
            culling, stripTile, scanRows, width, height);
        for (int rangeTile = range.first; rangeTile < range.second; ++rangeTile) {
            const uint32_t tileX = static_cast<uint32_t>(
                scanRows ? rangeTile : stripTile);
            const uint32_t tileY = static_cast<uint32_t>(
                scanRows ? stripTile : rangeTile);
            accepted.push_back(tileY * tileCountX + tileX);
        }
    }
    std::sort(accepted.begin(), accepted.end());
    return accepted;
}

using CompactProjectedGrad = std::array<double, 9>;

CompactProjectedGrad addCompactProjectedGrad(CompactProjectedGrad lhs,
                                             const CompactProjectedGrad& rhs) {
    for (size_t component = 0u; component < lhs.size(); ++component) {
        lhs[component] += rhs[component];
    }
    return lhs;
}

bool testTileLocalSubgroupReduction() {
    const size_t pixelCount = static_cast<size_t>(activeTileSize) * activeTileSize;
    constexpr size_t gaussianCount = 7u;
    std::array<std::vector<CompactProjectedGrad>, gaussianCount> contributions{};
    for (auto& gaussianContributions : contributions) {
        gaussianContributions.resize(pixelCount);
    }
    std::array<CompactProjectedGrad, gaussianCount> direct{};
    for (size_t gaussian = 0u; gaussian < gaussianCount; ++gaussian) {
        for (size_t pixel = 0u; pixel < pixelCount; ++pixel) {
            if ((pixel + gaussian * 3u) % 5u == 0u) {
                continue;
            }
            for (size_t component = 0u; component < 9u; ++component) {
                contributions[gaussian][pixel][component] = static_cast<double>(
                    ((gaussian + 1u) * (pixel + 3u) * (component + 2u)) % 17u);
            }
            direct[gaussian] = addCompactProjectedGrad(
                direct[gaussian], contributions[gaussian][pixel]);
        }
    }

    for (size_t subgroupSize : {16u, 32u, 64u}) {
        if (subgroupSize > pixelCount) {
            continue;
        }
        std::array<CompactProjectedGrad, gaussianCount> tileReduced{};
        for (size_t gaussian = 0u; gaussian < gaussianCount; ++gaussian) {
            for (size_t subgroupStart = 0u;
                 subgroupStart < pixelCount;
                 subgroupStart += subgroupSize) {
                CompactProjectedGrad subgroup{};
                for (size_t lane = 0u; lane < subgroupSize; ++lane) {
                    subgroup = addCompactProjectedGrad(
                        subgroup, contributions[gaussian][subgroupStart + lane]);
                }
                tileReduced[gaussian] = addCompactProjectedGrad(
                    tileReduced[gaussian], subgroup);
            }
        }
        if (tileReduced != direct) {
            return false;
        }
    }
    return true;
}

bool testTensorChildBatchReductionLayout() {
    constexpr uint32_t pixels = 64u;
    for (uint32_t batch : {8u, 16u, 32u}) {
        const uint32_t lanes = pixels / batch;
        for (uint32_t active : {1u, batch - 1u, batch}) {
            std::array<CompactProjectedGrad, pixels> threadGrad{};
            for (uint32_t thread = 0u; thread < pixels; ++thread) {
                const uint32_t gaussian = thread / lanes;
                if (gaussian >= active) continue;
                for (uint32_t pixel = thread % lanes; pixel < pixels; pixel += lanes) {
                    for (uint32_t component = 0u; component < 9u; ++component) {
                        threadGrad[thread][component] +=
                            static_cast<double>(((gaussian + 3u) * (pixel + 1u) * (component + 2u)) % 17u);
                    }
                }
            }
            for (uint32_t gaussian = 0u; gaussian < active; ++gaussian) {
                CompactProjectedGrad actual{}, expected{};
                for (uint32_t lane = 0u; lane < lanes; ++lane) {
                    actual = addCompactProjectedGrad(actual, threadGrad[gaussian * lanes + lane]);
                }
                for (uint32_t pixel = 0u; pixel < pixels; ++pixel) {
                    for (uint32_t component = 0u; component < 9u; ++component) {
                        expected[component] +=
                            static_cast<double>(((gaussian + 3u) * (pixel + 1u) * (component + 2u)) % 17u);
                    }
                }
                if (actual != expected) return false;
            }
        }
    }
    return normalizeAdaptiveTensorChildBatch(12u) == 8u &&
           normalizeAdaptiveTensorChildBatch(16u) == 16u &&
           normalizeAdaptiveTensorChildBatch(32u) == 32u;
}

bool testVkSplatTraversalOrder() {
    constexpr uint32_t gaussianCount = 11u;
    constexpr uint32_t pixelCount = 9u;
    constexpr uint32_t perSplatBatchSize = 4u;
    constexpr uint32_t tensorBatchSize = 16u;
    const std::array<uint32_t, pixelCount> processedCounts = {
        0u, 1u, 2u, 3u, 5u, 7u, 9u, 10u, 11u};
    std::array<std::vector<uint32_t>, pixelCount> perSplatOrder;
    std::array<std::vector<uint32_t>, pixelCount> tensorOrder;

    for (uint32_t batchBase = 0u; batchBase < gaussianCount;
         batchBase += perSplatBatchSize) {
        const uint32_t batchCount = std::min(
            perSplatBatchSize, gaussianCount - batchBase);
        for (uint32_t threadRank = 0u; threadRank < batchCount; ++threadRank) {
            const uint32_t itemOffset = gaussianCount - 1u -
                                        (batchBase + threadRank);
            for (uint32_t step = 0u; step < batchCount + pixelCount - 1u; ++step) {
                const int pixelRank = static_cast<int>(step) -
                                      static_cast<int>(threadRank);
                if (pixelRank >= 0 && pixelRank < static_cast<int>(pixelCount) &&
                    itemOffset < processedCounts[static_cast<size_t>(pixelRank)]) {
                    perSplatOrder[static_cast<size_t>(pixelRank)].push_back(itemOffset);
                }
            }
        }
    }

    uint32_t remaining = gaussianCount;
    while (remaining > 0u) {
        const uint32_t batchCount = std::min(tensorBatchSize, remaining);
        const uint32_t batchStart = remaining - batchCount;
        for (uint32_t reverseIndex = 0u; reverseIndex < batchCount; ++reverseIndex) {
            const uint32_t itemOffset = batchStart + batchCount - 1u - reverseIndex;
            for (uint32_t pixel = 0u; pixel < pixelCount; ++pixel) {
                if (itemOffset < processedCounts[pixel]) {
                    tensorOrder[pixel].push_back(itemOffset);
                }
            }
        }
        remaining = batchStart;
    }

    for (uint32_t pixel = 0u; pixel < pixelCount; ++pixel) {
        std::vector<uint32_t> expected;
        for (uint32_t item = processedCounts[pixel]; item > 0u; --item) {
            expected.push_back(item - 1u);
        }
        if (perSplatOrder[pixel] != expected || tensorOrder[pixel] != expected) {
            return false;
        }
    }
    return true;
}

bool testTensorProcessedPrefixTrimming() {
    constexpr uint32_t rangeCount = 257u;
    for (uint32_t batchSize : {8u, 16u, 32u}) {
        for (uint32_t maximum : {0u, 1u, 7u, 8u, 9u, 16u, 17u, 31u, 129u, rangeCount}) {
            // The final lane represents an out-of-image edge lane. Its stale
            // count must never affect the shared maximum or traversal.
            const std::array<uint32_t, 5> stored = {
                0u, maximum / 2u, maximum,
                maximum == rangeCount ? rangeCount + 100u : maximum, rangeCount + 100u};
            const std::array<bool, 5> valid = {true, true, true, true, false};
            std::array<uint32_t, 5> processed{};
            uint32_t prefix = 0u;
            for (size_t pixel = 0u; pixel < processed.size(); ++pixel) {
                processed[pixel] = valid[pixel] ? std::min(stored[pixel], rangeCount) : 0u;
                prefix = std::max(prefix, processed[pixel]);
            }
            if (prefix != maximum) return false;
            const auto traversal = [&](uint32_t remaining) {
                std::array<std::vector<uint32_t>, 5> order;
                while (remaining != 0u) {
                    const uint32_t count = std::min(remaining, batchSize);
                    const uint32_t start = remaining - count;
                    for (uint32_t reverse = 0u; reverse < count; ++reverse) {
                        const uint32_t item = start + count - 1u - reverse;
                        for (size_t pixel = 0u; pixel < processed.size(); ++pixel) {
                            if (item < processed[pixel]) order[pixel].push_back(item);
                        }
                    }
                    remaining = start;
                }
                return order;
            };
            if (traversal(rangeCount) != traversal(prefix)) return false;
            for (size_t pixel = 0u; pixel < processed.size(); ++pixel) {
                std::vector<uint32_t> expected;
                for (uint32_t item = processed[pixel]; item > 0u; --item) {
                    expected.push_back(item - 1u);
                }
                if (traversal(prefix)[pixel] != expected) return false;
            }
        }
    }
    return true;
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

        for (uint32_t tileY = 0; tileY < tileCountForExtent(height); ++tileY) {
            for (uint32_t tileX = 0; tileX < tileCountForExtent(width); ++tileX) {
                if (gaussianMayContributeToTile(gaussian, tileX, tileY, width, height)) {
                    continue;
                }
                ++culledTileCount;

                const uint32_t minimumX = tileX * activeTileSize;
                const uint32_t minimumY = tileY * activeTileSize;
                const uint32_t maximumX = std::min(minimumX + activeTileSize, width);
                const uint32_t maximumY = std::min(minimumY + activeTileSize, height);
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

bool testScanlineEllipseTileCullingMatchesRectanglePredicate() {
    constexpr uint32_t width = 63u;
    constexpr uint32_t height = 47u;
    constexpr double alphaMinimum = 1.0 / 255.0;
    std::vector<ConicGaussian2D> cases = {
        {31.5, 23.5, 1.0, 0.0, 1.0, 1.0, 8.0},
        {16.0, 16.0, 0.04, 0.0, 2.0, 0.8, 15.0},
        {32.0, 15.0, 0.8, 0.77, 0.8, 0.6, 14.0},
        {61.5, 45.5, 0.2, -0.12, 1.4, 0.9, 12.0},
        {20.0, 20.0, 1.0, 0.0, 1.0, alphaMinimum, 4.0},
        {20.0, 20.0, 1.0, 0.0, 1.0, alphaMinimum * 0.5, 4.0},
        {24.0, 24.0, -1.0, 0.0, 1.0, 0.8, 8.0},
        {24.0, 24.0, 1.0, 1.0, 1.0, 0.8, 8.0},
    };

    std::mt19937 random(41u);
    std::uniform_real_distribution<double> centerXDistribution(-24.0, 88.0);
    std::uniform_real_distribution<double> centerYDistribution(-24.0, 72.0);
    std::uniform_real_distribution<double> eigenvalueDistribution(0.01, 2.0);
    std::uniform_real_distribution<double> angleDistribution(0.0, 6.283185307179586);
    std::uniform_real_distribution<double> opacityDistribution(alphaMinimum, 1.0);
    for (uint32_t sample = 0u; sample < 4000u; ++sample) {
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
        cases.push_back(gaussian);
    }

    for (const ConicGaussian2D& gaussian : cases) {
        const auto scanline = acceptedTilesScanline(gaussian, width, height);
        const auto rectangle = acceptedTilesOptimized(gaussian, width, height);
        if (scanline != rectangle) {
            std::cerr << "scanline mismatch center=(" << gaussian.centerX << ", "
                      << gaussian.centerY << ") conic=(" << gaussian.a << ", "
                      << gaussian.b << ", " << gaussian.c << ") opacity="
                      << gaussian.opacity << " radius=" << gaussian.radius << "\n";
            return false;
        }
    }
    return true;
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

bool testAdaptiveSpatialTileIndexing() {
    constexpr TrainingExtent extent{53u, 37u};
    const TrainingTileLayout parents = makeTrainingTileLayout(extent, kTrainingTileSize16);
    const uint32_t leafCapacity = trainingAdaptiveRangeCapacity(extent);
    std::vector<uint8_t> assigned(leafCapacity, 0u);

    for (uint32_t parentY = 0u; parentY < parents.tileCountY; ++parentY) {
        for (uint32_t parentX = 0u; parentX < parents.tileCountX; ++parentX) {
            for (uint32_t childY = 0u; childY < 2u; ++childY) {
                for (uint32_t childX = 0u; childX < 2u; ++childX) {
                    const uint32_t leafIndex = trainingAdaptiveLeafIndex(
                        parentX, parentY, childX, childY, parents.tileCountX);
                    if (leafIndex >= leafCapacity || assigned[leafIndex] != 0u) {
                        return false;
                    }
                    assigned[leafIndex] = 1u;

                    const uint32_t pixelX = parentX * 16u + childX * 8u;
                    const uint32_t pixelY = parentY * 16u + childY * 8u;
                    if (trainingAdaptiveLeafIndexForPixel(pixelX, pixelY, parents.tileCountX,
                                                          true) != leafIndex) {
                        return false;
                    }
                }
            }

            const uint32_t coarseLeaf = trainingAdaptiveLeafIndexForPixel(
                parentX * 16u + 15u, parentY * 16u + 15u, parents.tileCountX, false);
            if (coarseLeaf !=
                trainingAdaptiveLeafIndex(parentX, parentY, 0u, 0u, parents.tileCountX)) {
                return false;
            }
        }
    }

    const TrainingTileLayout leaves = makeTrainingTileLayout(extent, kTrainingTileSize8);
    return leafCapacity == parents.tileCount * kAdaptiveTrainingTileLeafFactor &&
           leafCapacity > leaves.tileCount &&
           trainingAdaptiveLeafIndex(parents.tileCountX - 1u, parents.tileCountY - 1u, 1u, 1u,
                                     parents.tileCountX) == leafCapacity - 1u &&
           trainingAdaptiveRangeCapacity(TrainingExtent{32u, 32u}) ==
               makeTrainingTileLayout(TrainingExtent{32u, 32u}, kTrainingTileSize8).tileCount &&
           trainingAdaptiveParentIndex(2u, 1u, parents.tileCountX) < parents.tileCount;
}

bool testCompactAdaptiveChildCoverage() {
    constexpr TrainingExtent extent{53u, 37u};
    const auto parents = makeTrainingTileLayout(extent, kTrainingTileSize16);
    // Deliberately unsorted, including the partial bottom-right parent.
    for (const std::vector<uint32_t>& activeParents :
         {std::vector<uint32_t>{}, std::vector<uint32_t>{11u, 0u, 2u, 7u}}) {
        std::vector<uint8_t> selected(parents.tileCount, 0u);
        std::vector<uint8_t> coverage(extent.width * extent.height, 0u);
        for (uint32_t parent : activeParents) selected[parent] = 1u;
        for (uint32_t work = 0u; work < activeParents.size() * 4u; ++work) {
            const uint32_t parent = activeParents[work / 4u];
            const uint32_t originX = (parent % parents.tileCountX) * 16u + (work % 2u) * 8u;
            const uint32_t originY = (parent / parents.tileCountX) * 16u + ((work % 4u) / 2u) * 8u;
            for (uint32_t y = originY; y < std::min(originY + 8u, extent.height); ++y) {
                for (uint32_t x = originX; x < std::min(originX + 8u, extent.width); ++x) {
                    if (trainingAdaptiveLeafIndexForPixel(x, y, parents.tileCountX, true) !=
                        parent * 4u + work % 4u) return false;
                    ++coverage[y * extent.width + x];
                }
            }
        }
        for (uint32_t y = 0u; y < extent.height; ++y) {
            for (uint32_t x = 0u; x < extent.width; ++x) {
                const uint32_t parent = (y / 16u) * parents.tileCountX + x / 16u;
                if (coverage[y * extent.width + x] != selected[parent]) return false;
            }
        }
    }
    return true;
}

bool testCompactAdaptiveDispatchRows() {
    constexpr uint32_t widthLimit = kAdaptiveTrainingChildDispatchWidth;
    for (uint32_t parents : {0u, 1u, 16383u, 16384u, 20000u}) {
        const uint32_t children = parents * 4u;
        const uint32_t width = std::min(children, widthLimit);
        const uint32_t height = std::max(children / widthLimit +
                                             static_cast<uint32_t>(children % widthLimit != 0u), 1u);
        uint32_t validGroups = 0u;
        for (uint32_t y = 0u; y < height; ++y) {
            for (uint32_t x = 0u; x < width; ++x) {
                const uint32_t child = y * widthLimit + x;
                if (child / 4u >= parents) continue;
                if (child != validGroups++) return false;
            }
        }
        if (validGroups != children) return false;
    }
    return true;
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
    if (!approximatelyEqual(firstUpdate, 0.1, 1e-12)) {
        return false;
    }

    // The fused path must still run Adam with a zero gradient for an
    // invisible Gaussian so that existing moments decay exactly as in the
    // staged clear -> projection-backward -> optimizer path.
    double stagedFirstMoment = 0.4;
    double stagedSecondMoment = 0.2;
    double fusedFirstMoment = stagedFirstMoment;
    double fusedSecondMoment = stagedSecondMoment;
    const double stagedUpdate = adamUpdate(
        stagedFirstMoment, stagedSecondMoment, 0.0, 17u, 0.01);
    const double fusedUpdate = adamUpdate(
        fusedFirstMoment, fusedSecondMoment, 0.0, 17u, 0.01);
    return approximatelyEqual(stagedUpdate, fusedUpdate, 1e-12) &&
           approximatelyEqual(stagedFirstMoment, fusedFirstMoment, 1e-12) &&
           approximatelyEqual(stagedSecondMoment, fusedSecondMoment, 1e-12) &&
           std::abs(fusedUpdate) > 0.0;
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

bool testTileDependentMath(uint32_t tileSize) {
    activeTileSize = tileSize;
    return testProjectionAndTileBounds() &&
           testConservativeEllipseTileCulling() &&
           testOptimizedEllipseTileCullingMatchesLegacy() &&
           testScanlineEllipseTileCullingMatchesRectanglePredicate() &&
           testTileLocalSubgroupReduction();
}

} // namespace

int main() {
    if (!testConicOffDiagonalConvention()) {
        std::cerr << "conic off-diagonal convention test failed\n";
        return 1;
    }
    for (uint32_t tileSize : {8u, 16u}) {
        if (!testTileDependentMath(tileSize)) {
            std::cerr << "tile-dependent math test failed for " << tileSize << "x"
                      << tileSize << "\n";
            return 1;
        }
    }
    if (!testVkSplatTraversalOrder()) {
        std::cerr << "VkSplat traversal-order test failed\n";
        return 1;
    }
    if (!testTensorProcessedPrefixTrimming()) {
        std::cerr << "Tensor processed-prefix trimming test failed\n";
        return 1;
    }
    if (!testTensorChildBatchReductionLayout()) {
        std::cerr << "Tensor child batch/reduction layout test failed\n";
        return 1;
    }
    if (!testAdaptivePixelTo2DGSSelection()) {
        std::cerr << "adaptive pixel-to-2DGS selection test failed\n";
        return 1;
    }
    if (!testAdaptiveSpatialTileIndexing()) {
        std::cerr << "adaptive spatial tile indexing test failed\n";
        return 1;
    }
    if (!testCompactAdaptiveChildCoverage() || !testCompactAdaptiveDispatchRows()) {
        std::cerr << "compact adaptive child dispatch test failed\n";
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
