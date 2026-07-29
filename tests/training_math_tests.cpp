#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>

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
    if (!(gaussian.a > 0.0) || !(gaussian.c > 0.0) || !(determinant > 0.0)) {
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
