#include "training/core/training_ssim_mode.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using namespace vulkan3DGS;

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

double centerLossAndState(const std::vector<double>& rendered, const std::vector<double>& target,
                          int centerX, int centerY, State* state) {
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

    const double renderedVariance = renderedSqMean - renderedMean * renderedMean;
    const double targetVariance = targetSqMean - targetMean * targetMean;
    const double covariance = renderedTargetMean - renderedMean * targetMean;
    const double a = 2.0 * renderedMean * targetMean + kC1;
    const double b = 2.0 * covariance + kC2;
    const double c = renderedMean * renderedMean + targetMean * targetMean + kC1;
    const double d = renderedVariance + targetVariance + kC2;
    const double numerator = a * b;
    const double denominator = c * d;
    const double rawSsim = numerator / denominator;

    if (state) {
        const double denominatorSquared = denominator * denominator;
        constexpr double lossScale = -1.0 / 3.0;
        state->base =
            lossScale *
            (2.0 * targetMean * (b - a) * denominator - 2.0 * numerator * renderedMean * (d - c)) /
            denominatorSquared;
        state->targetCoefficient = lossScale * 2.0 * a * denominator / denominatorSquared;
        state->renderedCoefficient = lossScale * -2.0 * numerator * c / denominatorSquared;
    }
    return (1.0 - rawSsim) / 3.0;
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

double analyticGradient(const std::vector<double>& rendered, const std::vector<double>& target,
                        int sampleX, int sampleY) {
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
         centerY <= std::min(kHeight - 1, sampleY + kRadius); ++centerY) {
        for (int centerX = std::max(0, sampleX - kRadius);
             centerX <= std::min(kWidth - 1, sampleX + kRadius); ++centerX) {
            const State& state = states[index(centerX, centerY)];
            const double sampleWeight =
                weight(sampleX - centerX, sampleY - centerY) * inverseWeightSum;
            gradient += sampleWeight * (state.base + state.targetCoefficient * targetSample +
                                        state.renderedCoefficient * renderedSample);
        }
    }
    return gradient / static_cast<double>(kWidth * kHeight);
}

bool verifyGradient(const std::vector<double>& rendered, const std::vector<double>& target,
                    const char* label) {
    constexpr int sampleX = 3;
    constexpr int sampleY = 2;
    constexpr double epsilon = 1e-5;
    std::vector<double> plus = rendered;
    std::vector<double> minus = rendered;
    plus[index(sampleX, sampleY)] += epsilon;
    minus[index(sampleX, sampleY)] -= epsilon;
    const double finiteDifference =
        (totalLoss(plus, target) - totalLoss(minus, target)) / (2.0 * epsilon);
    const double analytic = analyticGradient(rendered, target, sampleX, sampleY);
    if (std::abs(finiteDifference - analytic) > 2e-5) {
        std::cerr << label << " SSIM gradient mismatch: analytic=" << analytic
                  << " finiteDifference=" << finiteDifference << '\n';
        return false;
    }
    return true;
}

// Mirror the shader's FP32 constants, fixed normalization, and zero padding.
// Scalar channels are independent; exercising them separately also covers RGB.
template <typename T> T shaderWeight(int offset) {
    constexpr std::array<double, 6> weights = {1.0,          0.8007374029, 0.4111122905,
                                               0.1353352832, 0.0285655008, 0.0038659201};
    return static_cast<T>(weights[static_cast<size_t>(std::abs(offset))]);
}

template <typename T, size_t N> using Fields = std::vector<std::array<T, N>>;

template <typename T, size_t N>
Fields<T, N> convolve(const Fields<T, N>& source, int width, int height, bool separable,
                      bool normalize = true) {
    constexpr T inverseWeight = static_cast<T>(0.0707622378);
    Fields<T, N> result(source.size());
    Fields<T, N> horizontal(source.size());
    if (separable) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                auto& sum = horizontal[static_cast<size_t>(y * width + x)];
                for (int dx = -kRadius; dx <= kRadius; ++dx) {
                    if (x + dx < 0 || x + dx >= width) continue;
                    const auto& sample = source[static_cast<size_t>(y * width + x + dx)];
                    for (size_t c = 0; c < N; ++c)
                        sum[c] += sample[c] * shaderWeight<T>(dx);
                }
            }
        }
    }
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            auto& sum = result[static_cast<size_t>(y * width + x)];
            for (int dy = -kRadius; dy <= kRadius; ++dy) {
                if (y + dy < 0 || y + dy >= height) continue;
                if (separable) {
                    const auto& sample = horizontal[static_cast<size_t>((y + dy) * width + x)];
                    for (size_t c = 0; c < N; ++c)
                        sum[c] += sample[c] * shaderWeight<T>(dy);
                } else {
                    for (int dx = -kRadius; dx <= kRadius; ++dx) {
                        if (x + dx < 0 || x + dx >= width) continue;
                        const T w = shaderWeight<T>(dx) * shaderWeight<T>(dy);
                        const auto& sample = source[static_cast<size_t>((y + dy) * width + x + dx)];
                        for (size_t c = 0; c < N; ++c)
                            sum[c] += sample[c] * w;
                    }
                }
            }
            if (normalize) {
                for (auto& value : sum)
                    value *= inverseWeight;
            }
        }
    }
    return result;
}

template <typename T> struct SsimResult {
    Fields<T, 5> moments;
    Fields<T, 3> coefficients;
    std::vector<T> loss;
    std::vector<T> gradient;
};

template <typename T>
SsimResult<T> evaluate(const std::vector<T>& rendered, const std::vector<T>& target, int width,
                       int height, TrainingSsimMode mode, T dssimWeight) {
    const auto count = rendered.size();
    Fields<T, 5> products(count);
    for (size_t i = 0; i < count; ++i) {
        const T r = rendered[i], t = target[i];
        products[i] = {r, t, r * r, t * t, r * t};
    }
    SsimResult<T> output{convolve(products, width, height, false),
                         Fields<T, 3>(count), std::vector<T>(count), std::vector<T>(count)};
    for (size_t i = 0; i < count; ++i) {
        const auto& m = output.moments[i];
        const T a = T(2) * m[0] * m[1] + T(kC1);
        const T b = T(2) * (m[4] - m[0] * m[1]) + T(kC2);
        const T c = m[0] * m[0] + m[1] * m[1] + T(kC1);
        const T renderedVariance = m[2] - m[0] * m[0];
        const T targetVariance = m[3] - m[1] * m[1];
        const T d = renderedVariance + targetVariance + T(kC2);
        const T numerator = a * b, denominator = c * d;
        const T denominatorSquared = denominator * denominator;
        constexpr T scale = T(-1) / T(3);
        output.coefficients[i] = {
            scale * (T(2) * m[1] * (b - a) * denominator - T(2) * numerator * m[0] * (d - c)) /
                denominatorSquared,
            scale * (T(2) * a * denominator) / denominatorSquared,
            scale * (-T(2) * numerator * c) / denominatorSquared};
        output.loss[i] = ((T(1) - dssimWeight) * std::abs(rendered[i] - target[i]) / T(3) +
                          dssimWeight * (T(1) - numerator / denominator) / T(3)) /
                         T(count);
    }
    // Direct mimics the old per-neighbor evaluation, rather than just comparing
    // two convolutions with the same factored derivative expression.
    Fields<T, 3> filtered;
    const bool separable = ssimUsesSeparableBackward(mode);
    if (separable) filtered = convolve(output.coefficients, width, height, true, false);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const size_t i = static_cast<size_t>(y * width + x);
            T derivative = T(0);
            if (separable) {
                derivative =
                    filtered[i][0] + target[i] * filtered[i][1] + rendered[i] * filtered[i][2];
                derivative *= T(0.0707622378);
            } else {
                for (int dy = -kRadius; dy <= kRadius; ++dy) {
                    if (y + dy < 0 || y + dy >= height) continue;
                    for (int dx = -kRadius; dx <= kRadius; ++dx) {
                        if (x + dx < 0 || x + dx >= width) continue;
                        const auto& s =
                            output.coefficients[static_cast<size_t>((y + dy) * width + x + dx)];
                        const T w = shaderWeight<T>(dx) * shaderWeight<T>(dy) * T(0.0707622378);
                        derivative += w * (s[0] + target[i] * s[1] + rendered[i] * s[2]);
                    }
                }
            }
            const T l1 = rendered[i] > target[i]   ? T(1) / T(3)
                         : rendered[i] < target[i] ? -T(1) / T(3)
                                                   : T(0);
            output.gradient[i] = ((T(1) - dssimWeight) * l1 + dssimWeight * derivative) / T(count);
        }
    }
    return output;
}

template <typename T> bool close(T a, T b, const char* field, int width, int height, size_t pixel) {
    const double error = std::abs(double(a) - double(b));
    constexpr bool fp32 = sizeof(T) == sizeof(float);
    const double tolerance =
        fp32 ? 1e-5 + 1e-3 * std::max(std::abs(double(a)), std::abs(double(b)))
             : 1e-9 + 1e-10 * std::max(std::abs(double(a)), std::abs(double(b)));
    if (!std::isfinite(a) || !std::isfinite(b) || error > tolerance) {
        std::cerr << field << " mismatch at " << width << 'x' << height << " pixel " << pixel
                  << ": 2D=" << a << " separable=" << b << '\n';
        return false;
    }
    return true;
}

template <typename T> bool verifySeparable() {
    double maxCoefficientDifference = 0.0;
    double maxGradientDifference = 0.0;
    int maxGradientWidth = 0, maxGradientHeight = 0;
    constexpr std::array<std::array<int, 2>, 6> extents = {
        {{1, 1}, {1, 19}, {19, 1}, {7, 5}, {17, 19}, {32, 16}}};
    for (const auto& extent : extents) {
        const int width = extent[0], height = extent[1];
        for (int pattern = 0; pattern < 6; ++pattern) {
            std::vector<T> rendered(static_cast<size_t>(width * height));
            std::vector<T> target(rendered.size());
            for (size_t i = 0; i < rendered.size(); ++i) {
                const T r = T((i * 17u + 1u) % 31u) / T(30);
                rendered[i] = pattern == 0   ? T(0.4)
                              : pattern == 1 ? T(i % 2u)
                              : pattern == 2 ? r
                                             : r - T(0.7);
                target[i] = pattern == 0   ? T(0.6)
                            : pattern == 1 ? T(1u - i % 2u)
                            : pattern == 2 ? T(1) - r
                                           : T((i * 11u + 3u) % 29u) / T(28);
                if (pattern >= 4) {
                    rendered[i] = target[i] = pattern == 4 ? T(0) : T(1);
                }
            }
            for (T blend : {T(0), T(0.2), T(1)}) {
                const auto direct = evaluate(rendered, target, width, height,
                                             TrainingSsimMode::TwoDimensional, blend);
                const auto separated = evaluate(rendered, target, width, height,
                                                TrainingSsimMode::BackwardSeparable, blend);
                const auto hybrid = evaluate(rendered, target, width, height,
                                             TrainingSsimMode::BackwardSeparable, blend);
                if (hybrid.moments != direct.moments ||
                    hybrid.coefficients != direct.coefficients || hybrid.loss != direct.loss) {
                    std::cerr << "Hybrid SSIM must preserve the exact 2D forward loss/state\n";
                    return false;
                }
                for (size_t i = 0; i < rendered.size(); ++i) {
                    for (size_t j = 0; j < 5; ++j) {
                        if (!close(direct.moments[i][j], separated.moments[i][j], "moment", width,
                                   height, i))
                            return false;
                    }
                    for (size_t j = 0; j < 3; ++j) {
                        const T a = direct.coefficients[i][j], b = separated.coefficients[i][j];
                        if (!std::isfinite(a) || !std::isfinite(b)) return false;
                        maxCoefficientDifference =
                            std::max(maxCoefficientDifference, std::abs(double(a) - double(b)));
                        // Coefficients precede pixel-count scaling and can cancel strongly
                        // (white-on-white base is analytically zero). Prove their equivalence
                        // in double; report FP32 intermediate error instead of applying a
                        // final-output absolute tolerance to this ill-conditioned quantity.
                        if constexpr (sizeof(T) == sizeof(double)) {
                            if (!close(a, b, "coefficient", width, height, i)) return false;
                        }
                    }
                    const double gradientDifference =
                        std::abs(double(direct.gradient[i]) - double(separated.gradient[i]));
                    if (gradientDifference > maxGradientDifference) {
                        maxGradientDifference = gradientDifference;
                        maxGradientWidth = width;
                        maxGradientHeight = height;
                    }
                    if (!close(direct.gradient[i], hybrid.gradient[i], "hybrid gradient", width,
                               height, i) ||
                        !close(direct.loss[i], separated.loss[i], "loss", width, height, i) ||
                        !close(direct.gradient[i], separated.gradient[i], "gradient", width, height,
                               i))
                        return false;
                }
                if constexpr (sizeof(T) == sizeof(double)) {
                    // Corners, image/tile edges and interior; no relaxed finite-difference
                    // threshold.
                    const std::array<size_t, 5> probes = {
                        0u, rendered.size() - 1u, rendered.size() / 2u,
                        static_cast<size_t>(std::min(15, width - 1)),
                        static_cast<size_t>(std::min(16, width - 1))};
                    for (size_t i : probes) {
                        auto plus = rendered, minus = rendered;
                        constexpr T epsilon = T(1e-5);
                        plus[i] += epsilon;
                        minus[i] -= epsilon;
                        const auto hi = evaluate(plus, target, width, height,
                                                 TrainingSsimMode::BackwardSeparable, blend);
                        const auto lo = evaluate(minus, target, width, height,
                                                 TrainingSsimMode::BackwardSeparable, blend);
                        T difference = T(0);
                        for (size_t j = 0; j < rendered.size(); ++j)
                            difference += hi.loss[j] - lo.loss[j];
                        const T finiteDifference = difference / (T(2) * epsilon);
                        if (std::abs(finiteDifference - separated.gradient[i]) > T(2e-5)) {
                            std::cerr << "Separable finite-difference mismatch at " << width << 'x'
                                      << height << " pixel " << i << ": " << finiteDifference
                                      << " != " << separated.gradient[i] << '\n';
                            return false;
                        }
                        const auto hybridHi = evaluate(plus, target, width, height,
                                                       TrainingSsimMode::BackwardSeparable, blend);
                        const auto hybridLo = evaluate(minus, target, width, height,
                                                       TrainingSsimMode::BackwardSeparable, blend);
                        T hybridDifference = T(0);
                        for (size_t j = 0; j < rendered.size(); ++j)
                            hybridDifference += hybridHi.loss[j] - hybridLo.loss[j];
                        if (std::abs(hybridDifference / (T(2) * epsilon) - hybrid.gradient[i]) >
                            T(2e-5)) {
                            std::cerr << "Hybrid finite-difference mismatch\n";
                            return false;
                        }
                    }
                }
            }
        }
    }
    std::cout << (sizeof(T) == sizeof(float) ? "FP32" : "Double") << " max coefficient difference "
              << maxCoefficientDifference << ", max pixel gradient difference "
              << maxGradientDifference << " at " << maxGradientWidth << 'x' << maxGradientHeight
              << '\n';
    return true;
}

} // namespace

int main() {
    std::vector<double> rendered(static_cast<size_t>(kWidth * kHeight));
    std::vector<double> target(static_cast<size_t>(kWidth * kHeight));
    for (size_t i = 0; i < rendered.size(); ++i) {
        rendered[i] = 0.2 + 0.6 * static_cast<double>((i * 17u) % 31u) / 30.0;
        target[i] = 0.15 + 0.7 * static_cast<double>((i * 11u + 3u) % 29u) / 28.0;
    }

    if (!verifyGradient(rendered, target, "regular")) {
        return 1;
    }

    for (size_t i = 0; i < rendered.size(); ++i) {
        rendered[i] = 0.1 + 0.8 * static_cast<double>((i * 13u) % 37u) / 36.0;
        target[i] = 1.0 - rendered[i];
    }
    if (centerLossAndState(rendered, target, 3, 2, nullptr) <= 1.0 / 3.0) {
        std::cerr << "negative SSIM test input did not produce negative SSIM\n";
        return 1;
    }
    if (!verifyGradient(rendered, target, "negative")) {
        return 1;
    }

    if (!verifySeparable<double>() || !verifySeparable<float>()) return 1;
    std::cout << "training SSIM 2D/hybrid moments, coefficients, loss and gradient tests passed\n";
    return 0;
}
