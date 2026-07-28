#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>

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
