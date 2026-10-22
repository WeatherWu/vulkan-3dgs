#include "training/core/training_checkpoint_layout.hpp"
#include "training_checkpoint_reference.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace vulkan3DGS;
namespace reference = checkpoint_reference;
constexpr double kAtol = 1e-5;
constexpr double kRtol = 1e-3;
constexpr std::array<const char*, 9> kComponentNames = {"center.x", "center.y", "conic.x",
                                                        "conic.y",  "conic.z",  "opacity",
                                                        "color.r",  "color.g",  "color.b"};

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

bool near(double a, double b, double atol = kAtol, double rtol = kRtol) {
    return std::isfinite(a) && std::isfinite(b) &&
           std::abs(a - b) <= atol + rtol * std::max(std::abs(a), std::abs(b));
}

void testLayout() {
    for (uint32_t a = 0; a <= 65; ++a) {
        for (uint32_t b = 0; b <= 65; ++b) {
            for (uint32_t c = 0; c <= 65; ++c) {
                const uint32_t m = a + b + c;
                require(checkpointStoredCount(a) + checkpointStoredCount(b) +
                                checkpointStoredCount(c) <=
                            m / 32u,
                        "Checkpoint slot bound failed");
                require(checkpointBucketCount(a) + checkpointBucketCount(b) +
                                checkpointBucketCount(c) <=
                            checkpointBucketCount(m) + 3u,
                        "Task capacity bound failed");
            }
        }
    }
    require(checkpointStoredCount(0) == 0 && checkpointStoredCount(32) == 0 &&
                checkpointStoredCount(33) == 1 && checkpointStoredCount(64) == 1 &&
                checkpointStoredCount(65) == 2,
            "Internal checkpoint reservation changed");
    require(static_cast<uint32_t>(TrainingCheckpointLayoutVersion::ForwardPrefix) == 1u &&
                static_cast<uint32_t>(TrainingCheckpointLayoutVersion::ReverseBoundary) == 2u,
            "Checkpoint layout versions overlap");
    for (uint32_t range = 0; range <= 257; ++range) {
        const uint32_t stored = checkpointStoredCount(range);
        for (uint32_t bucket = 0; bucket < checkpointBucketCount(range); ++bucket) {
            const uint32_t boundary = reverseCheckpointBoundary(range, bucket);
            const bool tail = bucket + 1u == checkpointBucketCount(range);
            const auto task = makeReverseCheckpointTask(0u, range, 5u, bucket, 5u + stored);
            require(task && task->itemCount > 0u && task->itemCount <= kCheckpointBucketSize,
                    "Reverse-boundary task construction failed");
            require(task->checkpointSlot == (tail ? kCheckpointImplicitSlot : 5u + bucket),
                    "Reverse task does not point at its far boundary");
            require(boundary <= range, "Reverse boundary exceeds the raw list");
            require(reverseCheckpointMustBeWritten(range, boundary) ==
                        (!tail && boundary == (bucket + 1u) * kCheckpointBucketSize),
                    "Reverse boundary writer predicate changed");
            for (uint32_t processed = 0; processed <= range; ++processed) {
                const uint32_t begin = bucket * kCheckpointBucketSize;
                const bool active = processed > begin;
                require(reverseCheckpointReadRequired(processed, begin, boundary) ==
                            (active && processed > boundary),
                        "Pixel-level reverse checkpoint read guard changed");
            }
        }
    }
    require(checkpointBucketCount(0xffffffffu) == 134217728u, "Bucket ceil overflowed");
    require(checkpointScratchElements(0u) == 0u && checkpointScratchElements(513u) == 3u &&
                checkpointScratchElements(65537u) == 259u,
            "Checkpoint hierarchical scratch layout changed");
    const auto capacity = makeTrainingCheckpointCapacity(1610569u, 1559u, 1039u);
    require(capacity.tileCount == 6370u && capacity.slotCapacity == 50331u &&
                capacity.taskCapacity == 56701u && capacity.bankBytes == 51538944u,
            "Real-scene capacity budget changed");
    require(capacity.fitsDescriptorRange(128u * 1024u * 1024u),
            "Four-bank descriptor budget failed");
    require(!capacity.fitsDescriptorRange(capacity.bankBytes - 1u),
            "Descriptor range guard failed");
    require(!makeTrainingCheckpointCapacity(1, 0, 1).shaderIndexSafe, "Zero extent accepted");
    require(!makeTrainingCheckpointCapacity(0xffffffffu, 1559, 1039).shaderIndexSafe,
            "Overflowing bank indexing accepted");
    require(!makeTrainingCheckpointCapacity(1, 0xffffffffu, 0xffffffffu).shaderIndexSafe,
            "Overflowing extent accepted");
    std::vector<bool> seen(4u * 4u * 64u);
    for (uint32_t slot = 0; slot < 4; ++slot) {
        for (uint32_t pixel = 0; pixel < 256; ++pixel) {
            const auto address = trainingCheckpointAddress(slot, pixel, 4);
            require(address && address->bank < 4 && address->element < 256,
                    "Bank address out of range");
            const auto flattened = address->bank * 256u + address->element;
            require(!seen[flattened], "Bank addresses alias");
            seen[flattened] = true;
        }
    }
    require(!trainingCheckpointAddress(kCheckpointImplicitSlot, 0, 4),
            "Implicit state read a bank");
    require(!trainingCheckpointAddress(4, 0, 4) && !trainingCheckpointAddress(0, 256, 4),
            "Out-of-range state accepted");
    require(!trainingCheckpointAddress(0xfffffffeu, 0, 0xffffffffu),
            "Bank multiply overflow accepted");
}

void testPixelMapping() {
    for (const auto extent : {std::array<uint32_t, 2>{1, 1}, {17, 19}, {1559, 1039}}) {
        uint64_t validCount = 0;
        const uint32_t tilesX = (extent[0] + 15u) / 16u;
        const uint32_t tilesY = (extent[1] + 15u) / 16u;
        for (uint32_t tileY = 0; tileY < tilesY; ++tileY) {
            for (uint32_t tileX = 0; tileX < tilesX; ++tileX) {
                for (uint32_t rank = 0; rank < 256; ++rank) {
                    const uint32_t x = tileX * 16u + rank % 16u;
                    const uint32_t y = tileY * 16u + rank / 16u;
                    if (x < extent[0] && y < extent[1]) ++validCount;
                    const auto address = trainingCheckpointAddress(0, rank, 1);
                    require(address && address->bank == rank / 64u &&
                                address->element == rank % 64u,
                            "Edge-tile bank mapping failed");
                }
            }
        }
        require(validCount == static_cast<uint64_t>(extent[0]) * extent[1],
                "Edge pixels counted twice");
    }
}

void testMultiTilePrefixAndTasks() {
    std::vector<uint32_t> counts(513);
    uint32_t totalItems = 0u;
    for (size_t i = 0; i < counts.size(); ++i) {
        counts[i] = static_cast<uint32_t>(i * 17u % 97u);
        totalItems += counts[i];
    }
    const uint64_t capacity = checkpointBucketCount(totalItems);
    uint32_t slotBase = 0u, reverseSlotBase = 0u, taskCount = 0u;
    std::vector<bool> reverseSlots(static_cast<size_t>(capacity));
    for (uint32_t tile = 0u; tile < counts.size(); ++tile) {
        const uint32_t range = counts[tile];
        for (uint32_t bucket = 0; bucket < checkpointBucketCount(range); ++bucket) {
            const auto task = makeTrainingCheckpointTask(tile, range, slotBase, bucket, capacity);
            require(task && task->itemCount > 0 && task->itemCount <= 32u,
                    "Multi-tile task construction failed");
            require(task->checkpointSlot ==
                        (bucket == 0u ? kCheckpointImplicitSlot : slotBase + bucket - 1u),
                    "Multi-tile checkpoint slot overlaps or skips");
            const auto reverseTask =
                makeReverseCheckpointTask(tile, range, reverseSlotBase, bucket, capacity);
            require(reverseTask && reverseTask->itemCount == task->itemCount,
                    "Multi-tile reverse task construction failed");
            const bool reverseTail = bucket + 1u == checkpointBucketCount(range);
            require(reverseTask->checkpointSlot ==
                        (reverseTail ? kCheckpointImplicitSlot : reverseSlotBase + bucket),
                    "Multi-tile reverse boundary slot overlaps or skips");
            if (!reverseTail) {
                require(reverseTask->checkpointSlot < reverseSlots.size() &&
                            !reverseSlots[reverseTask->checkpointSlot],
                        "Multi-tile reverse boundary slots alias");
                reverseSlots[reverseTask->checkpointSlot] = true;
            }
            ++taskCount;
        }
        require(!makeTrainingCheckpointTask(tile, range, slotBase, checkpointBucketCount(range),
                                            capacity),
                "Out-of-range tail bucket accepted");
        slotBase += checkpointStoredCount(range);
        reverseSlotBase += checkpointStoredCount(range);
    }
    require(slotBase <= totalItems / 32u && taskCount <= capacity + counts.size(),
            "Multi-block prefix exceeded capacity budget");
    require(!makeTrainingCheckpointTask(0, 33, 0, 1, 0), "Capacity overflow not rejected");
    require(!makeTrainingCheckpointTask(0, 65, 0xffffffffu, 1, 0xffffffffu),
            "Implicit sentinel reused as a physical slot");
    require(!makeReverseCheckpointTask(0, 65, 0, 1, 0),
            "Reverse physical boundary ignored capacity overflow");
    require(!makeReverseCheckpointTask(0, 65, 0xffffffffu, 1, 0xffffffffu),
            "Reverse boundary reused the implicit sentinel");
}

template <typename T> reference::Scene<T> makeScene(uint32_t length, uint32_t pattern) {
    reference::Scene<T> scene{};
    for (uint32_t i = 0; i < length; ++i) {
        reference::Gaussian<T> gaussian{};
        gaussian.centerX = T(0.12) + T((i * 7u) % 11u) / T(100);
        gaussian.centerY = -T(0.16) + T((i * 3u) % 7u) / T(100);
        gaussian.opacity = T(0.05) + T((i * 13u + 3u) % 97u) / T(194);
        gaussian.color = {T((i * 7u + 1u) % 31u) / T(30), T((i * 11u + 4u) % 29u) / T(28),
                          T((i * 17u + 2u) % 37u) / T(36)};
        if (pattern == 1u) gaussian.opacity = i % 5u == 1u ? T(0.2) : T(0.001);
        if (pattern == 2u || pattern == 4u || pattern == 5u || pattern == 6u) {
            gaussian.opacity = T(0.01);
        }
        if (pattern == 3u) {
            gaussian.opacity = i < 32u ? T(0.005) : T(0.98);
            gaussian.centerX = gaussian.centerY = T(0);
        }
        if (pattern == 4u) {
            for (auto& channel : gaussian.color)
                channel *= T(64);
        }
        if (pattern == 5u) {
            for (auto& channel : gaussian.color)
                channel = channel * T(72) - T(8);
        }
        if (pattern == 6u) gaussian.color = {T(64), T(32), T(16)};
        if (pattern == 7u || pattern == 8u || pattern == 9u) {
            // Valid nonnegative projected RGB, opaque foreground then a dim
            // accepted splat. The latter can disappear from FP32 final RGB.
            gaussian.centerX = gaussian.centerY = T(0);
            gaussian.opacity = i < 32u ? T(0.005) : i == 32u ? T(0.99) : T(0.98);
            gaussian.color = i <= 32u
                                 ? reference::Color<T>{pattern == 8u ? T(1) : T(64), T(0), T(0)}
                                 : reference::Color<T>{T(0.0002), T(0), T(0)};
            if (pattern == 9u && i >= 33u) {
                gaussian.opacity = i == 33u ? T(0.985) : T(0.01);
                if (i > 33u) gaussian.color = {T(0), T(0), T(0)};
            }
        }
        scene.gaussians.push_back(gaussian);
        scene.items.push_back(pattern == 1u && i % 11u == 3u ? 0xffffffffu : i);
    }
    // Repeat real Gaussian indices rather than inventing different values for one index.
    if (pattern == 1u && length > 2u) scene.items[length - 1u] = 1u;
    return scene;
}

void testCaptureAndStopContracts() {
    auto scene = makeScene<double>(65, 2);
    scene.gaussians[31].opacity = 0.001;
    scene.items[32] = 0xffffffffu;
    const auto full = reference::forward(scene, 0.0, 0.0);
    require(full.checkpoints[0].has_value(), "Non-contributing boundary was not captured");
    auto prefix = scene;
    prefix.items.resize(32);
    const auto expected = reference::forward(prefix, 0.0, 0.0);
    require(full.checkpoints[0]->prefix == expected.finalState.prefix &&
                full.checkpoints[0]->transmittance == expected.finalState.transmittance,
            "Checkpoint did not capture the raw-offset prefix");
    require(full.processedCount == 65 && full.contributionCount == 63,
            "Processed/contributor contract changed");
    auto missing = full;
    missing.checkpoints[1].reset();
    bool rejected = false;
    try {
        (void)reference::prefixCandidate(scene, missing, 0.0, 0.0,
                                         reference::Color<double>{0.3, -0.2, 0.1});
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, "Active reader accepted an unwritten checkpoint");
    const auto invalid = reference::prefixCandidate(
        scene, missing, 0.0, 0.0, reference::Color<double>{0.3, -0.2, 0.1}, false);
    for (const auto& gradient : invalid)
        for (double value : gradient)
            require(value == 0, "Invalid pixel produced a gradient");

    auto stopped = makeScene<double>(65, 0);
    for (auto& gaussian : stopped.gaussians) {
        gaussian.centerX = gaussian.centerY = 0.0;
        gaussian.opacity = 0.99;
    }
    stopped.gaussians[0].opacity = 0.9;
    const auto early = reference::forward(stopped, 0.0, 0.0);
    require(early.processedCount == 2 && early.contributionCount == 2,
            "Rejected early-stop Gaussian was blended");
    stopped.items.resize(2);
    const auto shortened = reference::forward(stopped, 0.0, 0.0);
    require(early.finalState.prefix == shortened.finalState.prefix &&
                early.finalState.transmittance == shortened.finalState.transmittance,
            "Early-stop state differs from the retained prefix");
    require(!early.checkpoints[0], "Finished pixel wrote an unreachable checkpoint");
}

void testFiniteDifferences() {
    auto scene = makeScene<double>(65, 2);
    const reference::Color<double> gradient{0.3, -0.2, 0.1};
    const auto state = reference::forward(scene, 0.0, 0.0);
    require(state.processedCount == 65,
            "Multi-bucket finite-difference case did not reach its expected tail");
    const auto analytic = reference::reverse(scene, state, 0.0, 0.0, gradient);
    const auto candidate = reference::prefixCandidate(scene, state, 0.0, 0.0, gradient);
    const auto reverseBoundaries = reference::buildReverseCheckpoints(scene, state, 0.0, 0.0);
    const auto boundaryCandidate =
        reference::reverseBoundaryCandidate(scene, state, reverseBoundaries, 0.0, 0.0, gradient);
    constexpr double epsilon = 1e-6;
    for (size_t id = 0; id < scene.gaussians.size(); ++id) {
        for (size_t component = 0; component < 9; ++component) {
            auto plus = scene, minus = scene;
            const auto perturb = [component](auto& gaussian, double delta) {
                switch (component) {
                    case 0:
                        gaussian.centerX += delta;
                        break;
                    case 1:
                        gaussian.centerY += delta;
                        break;
                    case 2:
                        gaussian.a += delta;
                        break;
                    case 3:
                        gaussian.b += delta;
                        break;
                    case 4:
                        gaussian.c += delta;
                        break;
                    case 5:
                        gaussian.opacity += delta;
                        break;
                    default:
                        gaussian.color[component - 6] += delta;
                        break;
                }
            };
            perturb(plus.gaussians[id], epsilon);
            perturb(minus.gaussians[id], -epsilon);
            const auto hi = reference::forward(plus, 0.0, 0.0).finalState.prefix;
            const auto lo = reference::forward(minus, 0.0, 0.0).finalState.prefix;
            double finiteDifference = 0;
            for (size_t channel = 0; channel < 3; ++channel)
                finiteDifference += gradient[channel] * (hi[channel] - lo[channel]) / (2 * epsilon);
            // Existing projected conic.y carries the half off-diagonal convention.
            const double scale = component == 3 ? 2.0 : 1.0;
            require(near(scale * analytic[id][component], finiteDifference),
                    "Tensor reference finite difference failed");
            require(near(scale * candidate[id][component], finiteDifference),
                    "Prefix formula finite difference failed");
            require(near(scale * boundaryCandidate[id][component], finiteDifference),
                    "Reverse-boundary formula finite difference failed");
        }
    }
}

void testReverseBoundaryCaptureContracts() {
    auto scene = makeScene<float>(65u, 2u);
    scene.items[32] = 0xffffffffu; // Boundary item is invalid, but its state must still be saved.
    const auto state = reference::forward(scene, 0.0f, 0.0f);
    require(state.processedCount == 65u, "Reverse boundary case lost its raw tail");
    const auto checkpoints = reference::buildReverseCheckpoints(scene, state, 0.0f, 0.0f);
    require(checkpoints.boundaries.size() == 2u && checkpoints.boundaries[0].has_value() &&
                checkpoints.boundaries[1].has_value(),
            "Reverse builder wrote a missing or unreachable boundary");
    auto oversized = state;
    oversized.processedCount = 0xffffffffu;
    const auto safeCheckpoints = reference::buildReverseCheckpoints(scene, oversized, 0.0f, 0.0f);
    const auto boundedGradient = reference::reverseBoundaryCandidate(
        scene, oversized, safeCheckpoints, 0.0f, 0.0f, reference::Color<float>{0.3f, -0.2f, 0.1f});
    require(boundedGradient == reference::reverse(scene, state, 0.0f, 0.0f,
                                                  reference::Color<float>{0.3f, -0.2f, 0.1f}),
            "Reverse CPU contract did not clamp processedCount to the raw range");

    reference::ReverseState<float> directBoundary{};
    directBoundary.transmittance = state.finalState.transmittance;
    for (uint32_t offset = state.processedCount; offset > 32u; --offset) {
        const uint32_t id = scene.items[offset - 1u];
        if (id >= scene.gaussians.size()) continue;
        const auto value = reference::sample(scene.gaussians[id], 0.0f, 0.0f);
        if (!reference::contributes(value)) continue;
        const float oneMinusAlpha = std::max(1.0f - value.alpha, 1e-6f);
        directBoundary.transmittance /= oneMinusAlpha;
        for (size_t channel = 0; channel < 3; ++channel)
            directBoundary.suffix[channel] = value.alpha * scene.gaussians[id].color[channel] +
                                             oneMinusAlpha * directBoundary.suffix[channel];
    }
    require(checkpoints.boundaries[0].value().suffix == directBoundary.suffix &&
                checkpoints.boundaries[0].value().transmittance == directBoundary.transmittance,
            "Invalid boundary item changed the far-side reverse state");

    auto noncontributing = makeScene<float>(65u, 2u);
    noncontributing.gaussians[32].opacity = 0.001f;
    const auto noncontributingForward = reference::forward(noncontributing, 0.0f, 0.0f);
    const auto noncontributingState =
        reference::buildReverseCheckpoints(noncontributing, noncontributingForward, 0.0f, 0.0f);
    require(noncontributingState.boundaries[0].has_value(),
            "Noncontributing boundary item prevented reverse checkpoint publication");

    auto earlyStop = makeScene<float>(65u, 0u);
    for (auto& gaussian : earlyStop.gaussians) {
        gaussian.centerX = gaussian.centerY = 0.0f;
        gaussian.opacity = 0.99f;
    }
    earlyStop.gaussians[0].opacity = 0.9f;
    const auto stopped = reference::forward(earlyStop, 0.0f, 0.0f);
    const auto stoppedCheckpoints =
        reference::buildReverseCheckpoints(earlyStop, stopped, 0.0f, 0.0f);
    require(stopped.processedCount == 2u &&
                std::none_of(stoppedCheckpoints.boundaries.begin(),
                             stoppedCheckpoints.boundaries.end(),
                             [](const auto& boundary) { return boundary.has_value(); }),
            "Early-stopped pixel wrote an unreachable reverse boundary");

    const auto invalidPixel = reference::buildReverseCheckpoints(scene, state, 0.0f, 0.0f, false);
    require(std::none_of(invalidPixel.boundaries.begin(), invalidPixel.boundaries.end(),
                         [](const auto& boundary) { return boundary.has_value(); }),
            "Invalid pixel wrote reverse checkpoint state");
    auto poisoned = checkpoints;
    poisoned.boundaries[0].reset();
    bool rejected = false;
    try {
        (void)reference::reverseBoundaryCandidate(scene, state, poisoned, 0.0f, 0.0f,
                                                  reference::Color<float>{0.3f, -0.2f, 0.1f});
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, "Reverse reader accepted an unwritten active checkpoint");
    (void)reference::reverseBoundaryCandidate(scene, state, poisoned, 0.0f, 0.0f,
                                              reference::Color<float>{0.3f, -0.2f, 0.1f}, false);
}

void testReverseBoundaryTaskOrder() {
    auto scene = makeScene<float>(1025u, 2u);
    for (auto& gaussian : scene.gaussians) {
        gaussian.centerX = gaussian.centerY = 0.0f;
        gaussian.opacity = 0.005f;
    }
    scene.items[32] = 0u;
    scene.items[64] = 1u;
    scene.items[65] = 1u;
    const auto state = reference::forward(scene, 0.0f, 0.0f);
    require(state.processedCount == scene.items.size(),
            "Task-order test did not preserve all repeated bucket items");
    const auto checkpoints = reference::buildReverseCheckpoints(scene, state, 0.0f, 0.0f);
    const reference::Color<float> gradient{0.3f, -0.2f, 0.1f};
    const auto descending = reference::reverseBoundaryCandidate(scene, state, checkpoints, 0.0f,
                                                                0.0f, gradient, true, true);
    const auto ascending = reference::reverseBoundaryCandidate(scene, state, checkpoints, 0.0f,
                                                               0.0f, gradient, true, false);
    for (size_t id = 0; id < descending.size(); ++id)
        for (size_t component = 0; component < 9; ++component)
            require(near(descending[id][component], ascending[id][component]),
                    "Bucket task order changed a gradient beyond the fixed tolerance");
}

void testReverseBoundaryMixedPixelPrefixes() {
    auto scene = makeScene<float>(1500u, 2u);
    for (auto& gaussian : scene.gaussians) {
        gaussian.centerX = gaussian.centerY = 0.0f;
        gaussian.opacity = 0.01f;
    }
    const reference::Color<float> pixelGradient{0.3f, -0.2f, 0.1f};
    std::array<uint32_t, 3> processed{};
    const std::array<float, 3> sampleX{0.0f, 1.0f, 2.0f};
    for (size_t pixel = 0; pixel < sampleX.size(); ++pixel) {
        const auto forward = reference::forward(scene, sampleX[pixel], 0.0f);
        processed[pixel] = forward.processedCount;
        const auto checkpoints =
            reference::buildReverseCheckpoints(scene, forward, sampleX[pixel], 0.0f);
        const auto expected =
            reference::reverse(scene, forward, sampleX[pixel], 0.0f, pixelGradient);
        const auto actual = reference::reverseBoundaryCandidate(
            scene, forward, checkpoints, sampleX[pixel], 0.0f, pixelGradient);
        for (size_t id = 0; id < expected.size(); ++id)
            for (size_t component = 0; component < 9; ++component)
                require(near(expected[id][component], actual[id][component]),
                        "Mixed per-pixel processed prefixes changed a gradient");
    }
    require(processed[0] != processed[1] && processed[1] != processed[2],
            "Mixed-prefix test did not create different active bucket tails");
}

void testResidualLossRegression() {
    const auto scene = makeScene<float>(63u, 9u);
    for (const auto& gaussian : scene.gaussians) {
        require(gaussian.opacity >= 0.0f && gaussian.opacity <= 1.0f,
                "Occlusion repro has an invalid opacity");
        require(gaussian.a * gaussian.c - gaussian.b * gaussian.b > 0.0f,
                "Occlusion repro has an invalid conic");
        for (float channel : gaussian.color)
            require(channel >= 0.0f, "Occlusion repro has negative RGB");
    }
    auto prefixOnly = scene;
    prefixOnly.items.resize(33u);
    const auto prefix = reference::forward(prefixOnly, 0.0f, 0.0f);
    const auto state = reference::forward(scene, 0.0f, 0.0f);
    require(state.processedCount > 34u, "Dim repro is only the last contributor");
    require(state.finalState.prefix == prefix.finalState.prefix,
            "Repro no longer loses the dim contribution in FP32 RGB");
    const reference::Color<float> pixelGradient{0.3f, -0.2f, 0.1f};
    const auto baseline = reference::reverse(scene, state, 0.0f, 0.0f, pixelGradient);
    const auto proposed = reference::prefixCandidate(scene, state, 0.0f, 0.0f, pixelGradient);
    const auto reverseCheckpoints = reference::buildReverseCheckpoints(scene, state, 0.0f, 0.0f);
    const auto reverseCandidate = reference::reverseBoundaryCandidate(
        scene, state, reverseCheckpoints, 0.0f, 0.0f, pixelGradient);
    require(baseline[33][5] > 0.0f, "Reference lost the accepted dim splat gradient");
    require(!near(baseline[33][5], proposed[33][5]),
            "Known-unsafe prefix formula unexpectedly passed; review the regression and readiness "
            "gate");
    require(near(baseline[33][5], reverseCandidate[33][5]),
            "Reverse-boundary path lost the non-last dim Gaussian gradient");
    const auto doubleScene = makeScene<double>(63u, 9u);
    const auto doubleState = reference::forward(doubleScene, 0.0, 0.0);
    const reference::Color<double> doublePixelGradient{0.3, -0.2, 0.1};
    const auto preciseBaseline =
        reference::reverse(doubleScene, doubleState, 0.0, 0.0, doublePixelGradient);
    const auto preciseProposed =
        reference::prefixCandidate(doubleScene, doubleState, 0.0, 0.0, doublePixelGradient);
    const auto doubleBoundaries =
        reference::buildReverseCheckpoints(doubleScene, doubleState, 0.0, 0.0);
    const auto preciseReverseCandidate = reference::reverseBoundaryCandidate(
        doubleScene, doubleState, doubleBoundaries, 0.0, 0.0, doublePixelGradient);
    require(near(preciseBaseline[33][5], preciseProposed[33][5], 1e-9, 1e-10),
            "Occlusion formula is not algebraically equivalent in double");
    require(near(preciseBaseline[33][5], preciseReverseCandidate[33][5], 1e-9, 1e-10),
            "Reverse-boundary formula failed the double occlusion reference");
    std::cout << std::setprecision(10)
              << "Non-last HDR occlusion repro: processedCount=" << state.processedCount
              << " final_red=prefix_before_dim_red=" << state.finalState.prefix[0]
              << " T_before_dim=" << prefix.finalState.transmittance
              << " reference_opacity_grad=" << baseline[33][5]
              << " candidate_opacity_grad=" << proposed[33][5] << '\n';
}

struct NumericReport {
    uint64_t cases = 0, components = 0, mismatches = 0, nonfinite = 0, nonnegativeMismatches = 0;
    double maximumAbsolute = 0;
    std::string firstMismatch;
};

template <typename T> NumericReport numericMatrix(bool useReverseBoundary) {
    NumericReport report{};
    constexpr bool fp32 = sizeof(T) == sizeof(float);
    const reference::Color<T> pixelGradient{T(0.3), -T(0.2), T(0.1)};
    for (uint32_t length : {0u, 1u, 31u, 32u, 33u, 63u, 64u, 65u, 257u, 1025u}) {
        for (uint32_t pattern = 0; pattern < 10; ++pattern) {
            const auto scene = makeScene<T>(length, pattern);
            const auto forward = reference::forward(scene, T(0), T(0));
            const auto baseline = reference::reverse(scene, forward, T(0), T(0), pixelGradient);
            std::vector<reference::Gradient<T>> proposed;
            reference::ReverseCheckpoints<T> reverseCheckpoints;
            if (useReverseBoundary) {
                reverseCheckpoints = reference::buildReverseCheckpoints(scene, forward, T(0), T(0));
                proposed = reference::reverseBoundaryCandidate(scene, forward, reverseCheckpoints,
                                                               T(0), T(0), pixelGradient);
            } else {
                proposed = reference::prefixCandidate(scene, forward, T(0), T(0), pixelGradient);
            }
            ++report.cases;
            for (size_t id = 0; id < baseline.size(); ++id) {
                for (size_t component = 0; component < 9; ++component) {
                    ++report.components;
                    const double a = baseline[id][component], b = proposed[id][component];
                    const bool finite = std::isfinite(a) && std::isfinite(b);
                    if (!finite) ++report.nonfinite;
                    const double error = std::abs(a - b);
                    if (finite) report.maximumAbsolute = std::max(report.maximumAbsolute, error);
                    const bool accepted = near(a, b, fp32 ? kAtol : 1e-9, fp32 ? kRtol : 1e-10);
                    if (!accepted) {
                        ++report.mismatches;
                        if (pattern != 5u) ++report.nonnegativeMismatches;
                        if (report.firstMismatch.empty()) {
                            std::ostringstream detail;
                            uint32_t itemOffset = kCheckpointImplicitSlot;
                            for (uint32_t offset = 0u; offset < forward.processedCount; ++offset)
                                if (scene.items[offset] == id) itemOffset = offset;
                            const uint32_t bucket = itemOffset == kCheckpointImplicitSlot
                                                        ? kCheckpointImplicitSlot
                                                        : itemOffset / kCheckpointBucketSize;
                            detail << std::setprecision(10) << "length=" << length
                                   << " pattern=" << pattern << " candidate="
                                   << (useReverseBoundary ? "reverse-boundary" : "forward-prefix")
                                   << " tile=0 pixel=(0,0) bucket=" << bucket
                                   << " itemOffset=" << itemOffset << " Gaussian=" << id
                                   << " component=" << kComponentNames[component]
                                   << " reference=" << a << " candidate=" << b
                                   << " abs_error=" << error
                                   << " processedCount=" << forward.processedCount;
                            if (useReverseBoundary && itemOffset != kCheckpointImplicitSlot) {
                                const uint32_t boundary = reverseCheckpointBoundary(
                                    static_cast<uint32_t>(scene.items.size()), bucket);
                                if (reverseCheckpointReadRequired(forward.processedCount,
                                                                  bucket * kCheckpointBucketSize,
                                                                  boundary) &&
                                    boundary % kCheckpointBucketSize == 0u) {
                                    const uint32_t slot = boundary / kCheckpointBucketSize - 1u;
                                    if (slot < reverseCheckpoints.boundaries.size() &&
                                        reverseCheckpoints.boundaries[slot]) {
                                        const auto& state = *reverseCheckpoints.boundaries[slot];
                                        detail << " state=(suffix:" << state.suffix[0] << ','
                                               << state.suffix[1] << ',' << state.suffix[2]
                                               << ";T:" << state.transmittance << ')';
                                    } else {
                                        detail << " state=<unwritten>";
                                    }
                                } else {
                                    detail << " state=<implicit-tail>";
                                }
                            }
                            report.firstMismatch = detail.str();
                        }
                    }
                }
            }
        }
    }
    return report;
}

void printReport(const char* label, const NumericReport& report) {
    std::cout << label << ": cases=" << report.cases << " components=" << report.components
              << " mismatches=" << report.mismatches << " nonfinite=" << report.nonfinite
              << " nonnegative_RGB_mismatches=" << report.nonnegativeMismatches
              << " max_abs_error=" << std::setprecision(10) << report.maximumAbsolute << '\n';
    if (!report.firstMismatch.empty())
        std::cout << "  first mismatch: " << report.firstMismatch << '\n';
}

} // namespace

int main(int argc, char** argv) {
    enum class Gate { ReferenceOnly, ForwardPrefix, ReverseBoundary };
    Gate gate = Gate::ReferenceOnly;
    if (argc == 2 && std::string_view(argv[1]) == "--candidate-gate")
        gate = Gate::ForwardPrefix;
    else if (argc == 2 && std::string_view(argv[1]) == "--reverse-boundary-gate")
        gate = Gate::ReverseBoundary;
    else if (argc != 1) {
        std::cerr << "Usage: training_checkpoint_tests [--candidate-gate | "
                     "--reverse-boundary-gate]\n";
        return 2;
    }
    try {
        testLayout();
        testPixelMapping();
        testMultiTilePrefixAndTasks();
        testCaptureAndStopContracts();
        testReverseBoundaryCaptureContracts();
        testReverseBoundaryTaskOrder();
        testReverseBoundaryMixedPixelPrefixes();
        testFiniteDifferences();
        testResidualLossRegression();
        const auto prefixDouble = numericMatrix<double>(false);
        const auto prefixFloat = numericMatrix<float>(false);
        const auto reverseDouble = numericMatrix<double>(true);
        const auto reverseFloat = numericMatrix<float>(true);
        printReport("Double forward-prefix negative control", prefixDouble);
        printReport("FP32 forward-prefix negative control", prefixFloat);
        printReport("Double reverse-boundary candidate", reverseDouble);
        printReport("FP32 reverse-boundary gate (atol=1e-5, rtol=1e-3)", reverseFloat);
        require(prefixDouble.mismatches == 0 && prefixDouble.nonfinite == 0,
                "Double forward-prefix algebraic reference failed");
        require(reverseDouble.mismatches == 0 && reverseDouble.nonfinite == 0,
                "Double reverse-boundary reference failed");
        if (gate == Gate::ForwardPrefix) {
            if (prefixFloat.mismatches || prefixFloat.nonfinite) {
                std::cerr << "EXPECTED REJECTION: FP32 forward-prefix state remains unsafe.\n";
                return 1;
            }
            std::cout
                << "Forward-prefix negative control unexpectedly passed; review regression.\n";
            return 1;
        } else if (gate == Gate::ReverseBoundary) {
            if (reverseFloat.mismatches || reverseFloat.nonfinite) {
                std::cerr << "P0 BLOCKED: FP32 reverse-boundary candidate failed.\n";
                return 1;
            }
            std::cout
                << "P0 CPU reverse-boundary gate passed; GPU validation and cost gates remain.\n";
        } else {
            // Structural/reference tests are distinct from numerical readiness.
            std::cout << "Reference tests passed. Forward-prefix readiness="
                      << (prefixFloat.mismatches || prefixFloat.nonfinite ? "REJECTED" : "PASS")
                      << "; reverse-boundary readiness="
                      << (reverseFloat.mismatches || reverseFloat.nonfinite ? "BLOCKED" : "PASS")
                      << ". Use --reverse-boundary-gate for P0 admission.\n";
            return 0;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Checkpoint contract test failed: " << error.what() << '\n';
        return 1;
    }
}
