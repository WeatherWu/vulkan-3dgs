#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

constexpr int kWidth = 7;
constexpr int kHeight = 5;
constexpr int kRadius = 5;
constexpr double kSigma = 1.5;
constexpr double kC1 = 0.0001;
constexpr double kC2 = 0.0009;

struct State {
    double base = 0.0;
    double targetCoefficient = 0.0;
    double renderedCoefficient = 0.0;
};

double weight(int dx, int dy) {
    return std::exp(-static_cast<double>(dx * dx + dy * dy) / (2.0 * kSigma * kSigma));
}

double weightSum() {
    double result = 0.0;
    for (int dy = -kRadius; dy <= kRadius; ++dy) {
        for (int dx = -kRadius; dx <= kRadius; ++dx) {
            result += weight(dx, dy);
        }
    }
    return result;
}

size_t index(int x, int y) {
    return static_cast<size_t>(y * kWidth + x);
}

double imageValue(const std::vector<double>& image, int x, int y) {
    if (x < 0 || y < 0 || x >= kWidth || y >= kHeight) {
        return 0.0;
    }
    return image[index(x, y)];
}

double centerLossAndState(const std::vector<double>& rendered,
                          const std::vector<double>& target,
                          int centerX,
                          int centerY,
                          State* state) {
    const double inverseWeightSum = 1.0 / weightSum();
    double renderedMean = 0.0;
    double targetMean = 0.0;
    double renderedSqMean = 0.0;
    double targetSqMean = 0.0;
    double renderedTargetMean = 0.0;
    for (int dy = -kRadius; dy <= kRadius; ++dy) {
        for (int dx = -kRadius; dx <= kRadius; ++dx) {
            const double normalizedWeight = weight(dx, dy) * inverseWeightSum;
            const double x = imageValue(rendered, centerX + dx, centerY + dy);
            const double y = imageValue(target, centerX + dx, centerY + dy);
            renderedMean += x * normalizedWeight;
            targetMean += y * normalizedWeight;
            renderedSqMean += x * x * normalizedWeight;
            targetSqMean += y * y * normalizedWeight;
            renderedTargetMean += x * y * normalizedWeight;
        }
    }

    const double renderedVariance = std::max(renderedSqMean - renderedMean * renderedMean, 0.0);
    const double targetVariance = std::max(targetSqMean - targetMean * targetMean, 0.0);
    const double covariance = renderedTargetMean - renderedMean * targetMean;
    const double a = 2.0 * renderedMean * targetMean + kC1;
    const double b = 2.0 * covariance + kC2;
    const double c = renderedMean * renderedMean + targetMean * targetMean + kC1;
    const double d = renderedVariance + targetVariance + kC2;
    const double rawSsim = (a * b) / std::max(c * d, 1e-8);
    const double clampedSsim = std::clamp(rawSsim, 0.0, 1.0);

    if (state) {
        const double mask = rawSsim > 0.0 && rawSsim < 1.0 ? 1.0 : 0.0;
        const double scale = -mask * rawSsim / 3.0;
        const double safeA = std::max(a, 1e-8);
        const double safeB = std::max(b, 1e-8);
        const double safeC = std::max(c, 1e-8);
        const double safeD = std::max(d, 1e-8);
        state->base = scale * (2.0 * targetMean / safeA -
                               2.0 * targetMean / safeB -
                               2.0 * renderedMean / safeC +
                               2.0 * renderedMean / safeD);
        state->targetCoefficient = scale * (2.0 / safeB);
        state->renderedCoefficient = scale * (-2.0 / safeD);
    }
    return (1.0 - clampedSsim) / 3.0;
}

double totalLoss(const std::vector<double>& rendered, const std::vector<double>& target) {
    double result = 0.0;
    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            result += centerLossAndState(rendered, target, x, y, nullptr);
        }
    }
    return result / static_cast<double>(kWidth * kHeight);
}

double analyticGradient(const std::vector<double>& rendered,
                        const std::vector<double>& target,
                        int sampleX,
                        int sampleY) {
    std::vector<State> states(static_cast<size_t>(kWidth * kHeight));
    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            centerLossAndState(rendered, target, x, y, &states[index(x, y)]);
        }
    }

    const double inverseWeightSum = 1.0 / weightSum();
    const double renderedSample = rendered[index(sampleX, sampleY)];
    const double targetSample = target[index(sampleX, sampleY)];
    double gradient = 0.0;
    for (int centerY = std::max(0, sampleY - kRadius);
         centerY <= std::min(kHeight - 1, sampleY + kRadius);
         ++centerY) {
        for (int centerX = std::max(0, sampleX - kRadius);
             centerX <= std::min(kWidth - 1, sampleX + kRadius);
             ++centerX) {
            const State& state = states[index(centerX, centerY)];
            const double sampleWeight = weight(sampleX - centerX, sampleY - centerY) * inverseWeightSum;
            gradient += sampleWeight * (state.base +
                                        state.targetCoefficient * targetSample +
                                        state.renderedCoefficient * renderedSample);
        }
    }
    return gradient / static_cast<double>(kWidth * kHeight);
}

} // namespace

int main() {
    std::vector<double> rendered(static_cast<size_t>(kWidth * kHeight));
    std::vector<double> target(static_cast<size_t>(kWidth * kHeight));
    for (size_t i = 0; i < rendered.size(); ++i) {
        rendered[i] = 0.2 + 0.6 * static_cast<double>((i * 17u) % 31u) / 30.0;
        target[i] = 0.15 + 0.7 * static_cast<double>((i * 11u + 3u) % 29u) / 28.0;
    }

    constexpr int sampleX = 3;
    constexpr int sampleY = 2;
    constexpr double epsilon = 1e-5;
    std::vector<double> plus = rendered;
    std::vector<double> minus = rendered;
    plus[index(sampleX, sampleY)] += epsilon;
    minus[index(sampleX, sampleY)] -= epsilon;
    const double finiteDifference = (totalLoss(plus, target) - totalLoss(minus, target)) / (2.0 * epsilon);
    const double analytic = analyticGradient(rendered, target, sampleX, sampleY);
    if (std::abs(finiteDifference - analytic) > 2e-5) {
        std::cerr << "SSIM gradient mismatch: analytic=" << analytic
                  << " finiteDifference=" << finiteDifference << '\n';
        return 1;
    }

    std::cout << "training SSIM gradient test passed\n";
    return 0;
}
