#include <algorithm>
#include <cmath>
#include <iostream>

namespace {

bool near(float lhs, float rhs, float tolerance = 1e-6f) {
    return std::abs(lhs - rhs) <= tolerance *
        std::max({1.0f, std::abs(lhs), std::abs(rhs)});
}

float compatibleKernel(float squaredRadius) {
    constexpr float exp4 = 0.01831563888873418f;
    constexpr float inverseOneMinusExp4 = 1.018657360363774f;
    return (std::exp(-4.0f * squaredRadius) - exp4) *
        inverseOneMinusExp4;
}

bool testFiniteNormalizedKernel() {
    return near(compatibleKernel(0.0f), 1.0f) &&
           near(compatibleKernel(1.0f), 0.0f) &&
           compatibleKernel(0.5f) > 0.0f &&
           compatibleKernel(0.5f) < 1.0f;
}

bool testPremultipliedSourceOver() {
    constexpr float sourceColor = 0.8f;
    constexpr float destinationColor = 0.2f;
    constexpr float sourceAlpha = 0.35f;
    constexpr float destinationAlpha = 0.6f;

    const float straightRgb = sourceColor * sourceAlpha +
        destinationColor * (1.0f - sourceAlpha);
    const float premultipliedRgb = sourceColor * sourceAlpha +
        destinationColor * (1.0f - sourceAlpha);
    const float accumulatedAlpha = sourceAlpha +
        destinationAlpha * (1.0f - sourceAlpha);
    const float legacyAlpha = sourceAlpha * sourceAlpha +
        destinationAlpha * (1.0f - sourceAlpha);

    return near(straightRgb, premultipliedRgb) &&
           accumulatedAlpha > legacyAlpha &&
           near(accumulatedAlpha, 0.74f);
}

} // namespace

int main() {
    if (!testFiniteNormalizedKernel()) {
        std::cerr << "Compatible Gaussian kernel is not normalized at its support\n";
        return 1;
    }
    if (!testPremultipliedSourceOver()) {
        std::cerr << "Premultiplied source-over blend math is invalid\n";
        return 1;
    }
    return 0;
}
