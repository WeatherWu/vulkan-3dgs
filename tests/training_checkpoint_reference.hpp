#pragma once

// Test-only scalar references for the shader contracts. In particular, reverse
// follows the existing Tensor shader, including its opacity/conic conventions;
// prefixCandidate is the proposed P/T formula, not an approved GPU algorithm.
#include "training/core/training_checkpoint_layout.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

namespace checkpoint_reference {

using namespace vulkan3DGS;
template <typename T> using Color = std::array<T, 3>;
template <typename T> using Gradient = std::array<T, 9>;

template <typename T> struct Gaussian {
    T centerX = T(0), centerY = T(0);
    T a = T(0.8), b = T(0.07), c = T(0.6);
    T opacity = T(0.1);
    Color<T> color{};
};

template <typename T> struct Scene {
    std::vector<Gaussian<T>> gaussians;
    std::vector<uint32_t> items;
};

template <typename T> struct State {
    Color<T> prefix{};
    T transmittance = T(1);
};

template <typename T> struct ReverseState {
    Color<T> suffix{};
    T transmittance = T(1);
};

template <typename T> struct Forward {
    State<T> finalState{};
    uint32_t processedCount = 0u;
    uint32_t contributionCount = 0u;
    std::vector<std::optional<State<T>>> checkpoints;
};

template <typename T> struct ReverseCheckpoints {
    std::vector<std::optional<ReverseState<T>>> boundaries;
};

template <typename T> struct Sample {
    T dx, dy, coverage, alpha;
};

template <typename T> Sample<T> sample(const Gaussian<T>& gaussian, T x, T y) {
    const T dx = x - gaussian.centerX, dy = y - gaussian.centerY;
    const T power =
        -T(0.5) * (gaussian.a * dx * dx + T(2) * gaussian.b * dx * dy + gaussian.c * dy * dy);
    const T coverage = power <= T(0) ? std::exp(power) : T(0);
    return {dx, dy, coverage, std::clamp(gaussian.opacity * coverage, T(0), T(0.99))};
}

template <typename T> bool contributes(const Sample<T>& value) {
    return value.alpha >= T(1) / T(255);
}

template <typename T> Forward<T> forward(const Scene<T>& scene, T x, T y, bool validPixel = true) {
    Forward<T> result{};
    result.checkpoints.resize(checkpointStoredCount(static_cast<uint32_t>(scene.items.size())));
    if (!validPixel) return result;
    for (uint32_t offset = 0u; offset < scene.items.size(); ++offset) {
        // Capture before index/contribution guards, at raw sorted offsets.
        if (offset != 0u && offset % kCheckpointBucketSize == 0u) {
            result.checkpoints[offset / kCheckpointBucketSize - 1u] = result.finalState;
        }
        const uint32_t id = scene.items[offset];
        if (id >= scene.gaussians.size()) continue;
        const auto& gaussian = scene.gaussians[id];
        const auto value = sample(gaussian, x, y);
        if (!contributes(value)) continue;
        const T next = result.finalState.transmittance * (T(1) - value.alpha);
        if (next < T(1e-4)) break; // Reject this Gaussian, do not advance processedCount.
        for (size_t channel = 0; channel < 3; ++channel) {
            result.finalState.prefix[channel] +=
                result.finalState.transmittance * value.alpha * gaussian.color[channel];
        }
        result.finalState.transmittance *= T(1) - value.alpha;
        result.processedCount = offset + 1u;
        ++result.contributionCount;
    }
    return result;
}

template <typename T>
void addPair(Gradient<T>& gradient, const Gaussian<T>& gaussian, const Sample<T>& value, T dPower,
             const Color<T>& dColor) {
    // Exactly the nine active components written by backwardTensorPair.
    gradient[0] += dPower * (gaussian.a * value.dx + gaussian.b * value.dy);
    gradient[1] += dPower * (gaussian.b * value.dx + gaussian.c * value.dy);
    gradient[2] += dPower * (-T(0.5) * value.dx * value.dx);
    gradient[3] += dPower * (-T(0.5) * value.dx * value.dy);
    gradient[4] += dPower * (-T(0.5) * value.dy * value.dy);
    gradient[5] += dPower / std::max(gaussian.opacity, T(1e-20));
    for (size_t channel = 0; channel < 3; ++channel)
        gradient[6 + channel] += dColor[channel];
}

template <typename T>
std::vector<Gradient<T>> reverse(const Scene<T>& scene, const Forward<T>& state, T x, T y,
                                 const Color<T>& pixelGradient) {
    std::vector<Gradient<T>> result(scene.gaussians.size());
    T transmittance = state.finalState.transmittance;
    Color<T> suffix{};
    const uint32_t processed =
        std::min(state.processedCount, static_cast<uint32_t>(scene.items.size()));
    for (uint32_t remaining = processed; remaining != 0u; --remaining) {
        const uint32_t id = scene.items[remaining - 1u];
        if (id >= scene.gaussians.size()) continue;
        const auto& gaussian = scene.gaussians[id];
        const auto value = sample(gaussian, x, y);
        if (!contributes(value)) continue;
        const T u = std::max(T(1) - value.alpha, T(1e-6));
        transmittance /= u;
        T dAlpha = T(0);
        Color<T> dColor{};
        for (size_t channel = 0; channel < 3; ++channel) {
            dAlpha += pixelGradient[channel] *
                      (transmittance * (gaussian.color[channel] - suffix[channel]));
            dColor[channel] = pixelGradient[channel] * (transmittance * value.alpha);
        }
        const T dPower = dAlpha * gaussian.opacity * value.coverage;
        addPair(result[id], gaussian, value, dPower, dColor);
        for (size_t channel = 0; channel < 3; ++channel) {
            suffix[channel] = value.alpha * gaussian.color[channel] + u * suffix[channel];
        }
    }
    return result;
}

template <typename T>
ReverseCheckpoints<T> buildReverseCheckpoints(const Scene<T>& scene, const Forward<T>& forwardState,
                                              T x, T y, bool validPixel = true) {
    ReverseCheckpoints<T> result{};
    result.boundaries.resize(checkpointStoredCount(static_cast<uint32_t>(scene.items.size())));
    if (!validPixel) return result;

    ReverseState<T> state{};
    state.transmittance = forwardState.finalState.transmittance;
    const uint32_t processed =
        std::min(forwardState.processedCount, static_cast<uint32_t>(scene.items.size()));
    for (uint32_t remaining = processed; remaining != 0u; --remaining) {
        const uint32_t offset = remaining - 1u;
        const uint32_t id = scene.items[offset];
        if (id < scene.gaussians.size()) {
            const auto& gaussian = scene.gaussians[id];
            const auto value = sample(gaussian, x, y);
            if (contributes(value)) {
                const T oneMinusAlpha = std::max(T(1) - value.alpha, T(1e-6));
                state.transmittance /= oneMinusAlpha;
                for (size_t channel = 0; channel < 3; ++channel) {
                    state.suffix[channel] = value.alpha * gaussian.color[channel] +
                                            oneMinusAlpha * state.suffix[channel];
                }
            }
        }

        // Offset e is the first raw item in the far-side suffix B(e). Save
        // after consuming it, including when it was invalid or noncontributing.
        if (offset != 0u && reverseCheckpointMustBeWritten(processed, offset)) {
            const uint32_t slot = offset / kCheckpointBucketSize - 1u;
            if (slot >= result.boundaries.size())
                throw std::runtime_error("Reverse checkpoint boundary exceeds its reserved slots");
            result.boundaries[slot] = state;
        }
    }
    return result;
}

template <typename T>
std::vector<Gradient<T>>
reverseBoundaryCandidate(const Scene<T>& scene, const Forward<T>& forwardState,
                         const ReverseCheckpoints<T>& checkpoints, T x, T y,
                         const Color<T>& pixelGradient, bool validPixel = true,
                         bool descendingBuckets = true) {
    std::vector<Gradient<T>> result(scene.gaussians.size());
    if (!validPixel) return result;

    const uint32_t rangeCount = static_cast<uint32_t>(scene.items.size());
    const uint32_t processedCount = std::min(forwardState.processedCount, rangeCount);
    const uint32_t bucketCount = checkpointBucketCount(processedCount);
    // Descending bucket order preserves the reference per-Gaussian accumulation
    // order, including repeated item IDs across bucket boundaries.
    for (uint32_t taskIndex = 0u; taskIndex < bucketCount; ++taskIndex) {
        const uint32_t bucket = descendingBuckets ? bucketCount - 1u - taskIndex : taskIndex;
        const uint32_t begin = bucket * kCheckpointBucketSize;
        const uint32_t boundary = reverseCheckpointBoundary(rangeCount, bucket);
        if (processedCount <= begin) continue; // Guard before any checkpoint read.

        ReverseState<T> state{};
        state.transmittance = forwardState.finalState.transmittance;
        if (reverseCheckpointReadRequired(processedCount, begin, boundary)) {
            if (boundary == 0u || boundary % kCheckpointBucketSize != 0u)
                throw std::runtime_error("Reverse reader requested a non-bucket checkpoint");
            const uint32_t slot = boundary / kCheckpointBucketSize - 1u;
            if (slot >= checkpoints.boundaries.size() || !checkpoints.boundaries[slot])
                throw std::runtime_error("Active reverse bucket reads an unwritten checkpoint");
            state = *checkpoints.boundaries[slot];
        }

        const uint32_t end = std::min(boundary, processedCount);
        for (uint32_t offset = end; offset > begin; --offset) {
            const uint32_t id = scene.items[offset - 1u];
            if (id >= scene.gaussians.size()) continue;
            const auto& gaussian = scene.gaussians[id];
            const auto value = sample(gaussian, x, y);
            if (!contributes(value)) continue;
            const T oneMinusAlpha = std::max(T(1) - value.alpha, T(1e-6));
            state.transmittance /= oneMinusAlpha;
            Color<T> dColor{};
            T dAlpha = T(0);
            for (size_t channel = 0; channel < 3; ++channel) {
                dAlpha += pixelGradient[channel] *
                          (state.transmittance * (gaussian.color[channel] - state.suffix[channel]));
                dColor[channel] = pixelGradient[channel] * (state.transmittance * value.alpha);
            }
            const T dPower = dAlpha * gaussian.opacity * value.coverage;
            addPair(result[id], gaussian, value, dPower, dColor);
            for (size_t channel = 0; channel < 3; ++channel) {
                state.suffix[channel] =
                    value.alpha * gaussian.color[channel] + oneMinusAlpha * state.suffix[channel];
            }
        }
    }
    return result;
}

template <typename T>
std::vector<Gradient<T>> prefixCandidate(const Scene<T>& scene, const Forward<T>& forwardState, T x,
                                         T y, const Color<T>& pixelGradient,
                                         bool validPixel = true) {
    std::vector<Gradient<T>> result(scene.gaussians.size());
    if (!validPixel) return result; // Must precede any checkpoint read.
    const uint32_t buckets = checkpointBucketCount(forwardState.processedCount);
    for (uint32_t bucket = 0; bucket < buckets; ++bucket) {
        State<T> state{};
        if (bucket != 0u) {
            const auto& captured = forwardState.checkpoints.at(bucket - 1u);
            if (!captured) throw std::runtime_error("Active bucket reads an unwritten checkpoint");
            state = *captured;
        }
        const uint32_t begin = bucket * kCheckpointBucketSize;
        const uint32_t end = std::min(forwardState.processedCount, begin + kCheckpointBucketSize);
        for (uint32_t offset = begin; offset < end; ++offset) {
            const uint32_t id = scene.items[offset];
            if (id >= scene.gaussians.size()) continue;
            const auto& gaussian = scene.gaussians[id];
            const auto value = sample(gaussian, x, y);
            if (!contributes(value)) continue;
            const T u = std::max(T(1) - value.alpha, T(1e-6));
            T dAlpha = T(0);
            Color<T> dColor{};
            for (size_t channel = 0; channel < 3; ++channel) {
                const T residual = forwardState.finalState.prefix[channel] - state.prefix[channel];
                dAlpha += pixelGradient[channel] *
                          ((state.transmittance * gaussian.color[channel] - residual) / u);
                dColor[channel] = pixelGradient[channel] * (state.transmittance * value.alpha);
            }
            addPair(result[id], gaussian, value, dAlpha * gaussian.opacity * value.coverage,
                    dColor);
            for (size_t channel = 0; channel < 3; ++channel) {
                state.prefix[channel] +=
                    state.transmittance * value.alpha * gaussian.color[channel];
            }
            state.transmittance *= T(1) - value.alpha;
        }
    }
    return result;
}

} // namespace checkpoint_reference
