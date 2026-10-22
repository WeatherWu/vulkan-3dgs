#include "training/core/backward/pixel_to_2dgs_dispatcher.hpp"
#include "training/core/gaussian_training.hpp"
#include "training/core/training_pixel_culling_policy.hpp"

#include <cstdio>
#include <array>
#include <string_view>

using namespace vulkan3DGS;

namespace {

bool testFallbacks() {
    const PixelTo2DGSCapabilities unavailable{};
    return resolvePixelTo2DGSMode(TrainingPixelTo2DGSMode::Auto, unavailable) ==
               TrainingPixelTo2DGSMode::Direct &&
           resolvePixelTo2DGSMode(TrainingPixelTo2DGSMode::Subgroup, unavailable) ==
               TrainingPixelTo2DGSMode::Direct &&
           resolvePixelTo2DGSMode(TrainingPixelTo2DGSMode::TileGaussianAtomic, unavailable) ==
               TrainingPixelTo2DGSMode::Direct &&
           resolvePixelTo2DGSMode(TrainingPixelTo2DGSMode::VkSplatPerSplat, unavailable) ==
               TrainingPixelTo2DGSMode::Direct &&
           resolvePixelTo2DGSMode(TrainingPixelTo2DGSMode::VkSplatTensor, unavailable) ==
               TrainingPixelTo2DGSMode::Direct &&
           resolvePixelTo2DGSMode(TrainingPixelTo2DGSMode::CheckpointedWarp, unavailable) ==
               TrainingPixelTo2DGSMode::Direct &&
           resolvePixelTo2DGSMode(TrainingPixelTo2DGSMode::WorkgroupShared, unavailable) ==
               TrainingPixelTo2DGSMode::WorkgroupShared;
}

bool testSupportedModes() {
    PixelTo2DGSCapabilities capabilities{};
    capabilities.subgroup = true;
    capabilities.tileGaussian = true;
    capabilities.vkSplatPerSplat = true;
    capabilities.vkSplatTensor = true;
    return resolvePixelTo2DGSMode(TrainingPixelTo2DGSMode::Auto, capabilities) ==
               TrainingPixelTo2DGSMode::Auto &&
           resolvePixelTo2DGSMode(TrainingPixelTo2DGSMode::Subgroup, capabilities) ==
               TrainingPixelTo2DGSMode::Subgroup &&
           resolvePixelTo2DGSMode(TrainingPixelTo2DGSMode::TileGaussianAtomic, capabilities) ==
               TrainingPixelTo2DGSMode::TileGaussianAtomic &&
           resolvePixelTo2DGSMode(TrainingPixelTo2DGSMode::VkSplatPerSplat, capabilities) ==
               TrainingPixelTo2DGSMode::VkSplatPerSplat &&
           resolvePixelTo2DGSMode(TrainingPixelTo2DGSMode::VkSplatTensor, capabilities) ==
               TrainingPixelTo2DGSMode::VkSplatTensor;
}

bool testAdaptiveSpatialRequiresOptimizedBackward() {
    PixelTo2DGSCapabilities capabilities{};
    capabilities.vkSplatTensor = true;
    if (!supportsAdaptiveSpatialBackward(TrainingPixelTo2DGSMode::VkSplatTensor, capabilities) ||
        supportsAdaptiveSpatialBackward(TrainingPixelTo2DGSMode::Direct, capabilities) ||
        supportsAdaptiveSpatialBackward(TrainingPixelTo2DGSMode::Auto, capabilities) ||
        supportsAdaptiveSpatialBackward(TrainingPixelTo2DGSMode::VkSplatPerSplat, capabilities)) {
        return false;
    }

    capabilities.vkSplatPerSplat = true;
    return supportsAdaptiveSpatialBackward(TrainingPixelTo2DGSMode::VkSplatPerSplat, capabilities);
}

bool testTensorSubgroupPolicy() {
    PixelTo2DGSCapabilities caps{};
    caps.vkSplatTensor = true;
    caps.subgroup = true;
    caps.subgroupSize = 32u;
    using Mode = TrainingPixelTo2DGSMode;
    using Tile = TrainingTileMode;
    if (!supportsTensorSubgroupReduction(caps) ||
        !useTensorSubgroupReduction(true, Mode::VkSplatTensor, Tile::Fixed16, caps) ||
        useTensorSubgroupReduction(false, Mode::VkSplatTensor, Tile::Fixed16, caps) ||
        useTensorSubgroupReduction(true, Mode::Direct, Tile::Fixed16, caps) ||
        useTensorSubgroupReduction(true, Mode::Auto, Tile::Fixed16, caps) ||
        useTensorSubgroupReduction(true, Mode::VkSplatTensor, Tile::Fixed8, caps) ||
        useTensorSubgroupReduction(true, Mode::VkSplatTensor, Tile::AdaptiveSpatial, caps)) return false;
    for (uint32_t size : {0u, 16u, 64u}) {
        caps.subgroupSize = size;
        if (supportsTensorSubgroupReduction(caps)) return false;
    }
    caps.subgroupSize = 32u;
    caps.subgroup = false;
    if (supportsTensorSubgroupReduction(caps)) return false;
    caps.subgroup = true;
    caps.vkSplatTensor = false;
    return !supportsTensorSubgroupReduction(caps);
}

bool testTensorGlobalPixelGradPolicy() {
    using Mode = TrainingPixelTo2DGSMode;
    using Tile = TrainingTileMode;
    PixelTo2DGSCapabilities caps{};
    caps.vkSplatTensor = true;
    caps.subgroup = true;
    caps.subgroupSize = 32u;
    for (bool requested : {false, true}) {
        for (bool subgroup : {false, true}) {
            for (Mode mode : {Mode::Auto, Mode::Direct, Mode::WorkgroupShared, Mode::Subgroup,
                              Mode::TileGaussianAtomic, Mode::VkSplatPerSplat, Mode::VkSplatTensor}) {
                for (Tile tile : {Tile::Fixed8, Tile::Fixed16, Tile::AdaptiveSpatial}) {
                    const bool expected = requested && subgroup && mode == Mode::VkSplatTensor &&
                                          tile == Tile::Fixed16;
                    if (useTensorGlobalPixelGrad(requested, subgroup, mode, tile, caps) != expected) {
                        return false;
                    }
                }
            }
        }
    }
    for (uint32_t size : {0u, 16u, 64u}) {
        caps.subgroupSize = size;
        if (useTensorGlobalPixelGrad(true, true, Mode::VkSplatTensor, Tile::Fixed16, caps)) return false;
    }
    caps.subgroupSize = 32u;
    caps.subgroup = false;
    if (useTensorGlobalPixelGrad(true, true, Mode::VkSplatTensor, Tile::Fixed16, caps)) return false;
    caps.subgroup = true;
    caps.vkSplatTensor = false;
    return !useTensorGlobalPixelGrad(true, true, Mode::VkSplatTensor, Tile::Fixed16, caps);
}

bool testTensorRgbLatencyPolicy() {
    using Mode = TrainingPixelTo2DGSMode;
    using Tile = TrainingTileMode;
    const PixelTo2DGSCapabilities full{32u, true, true, true, true};
    std::array<PixelTo2DGSCapabilities, 6> capabilities{};
    capabilities.fill(full);
    capabilities[1].subgroupSize = 0u;
    capabilities[2].subgroupSize = 16u;
    capabilities[3].subgroupSize = 64u;
    capabilities[4].subgroup = false;
    capabilities[5].vkSplatTensor = false;
    for (size_t cap = 0; cap < capabilities.size(); ++cap) {
        for (uint32_t flags = 0u; flags < 8u; ++flags) {
            for (uint32_t mode = 0u; mode <= 6u; ++mode) {
                for (Tile tile : {Tile::Fixed8, Tile::Fixed16, Tile::AdaptiveSpatial}) {
                    const bool expected = cap == 0u && flags == 7u && mode == 6u && tile == Tile::Fixed16;
                    if (useTensorRgbLatencyHiding((flags & 1u) != 0u, (flags & 2u) != 0u,
                                                  (flags & 4u) != 0u, static_cast<Mode>(mode),
                                                  tile, capabilities[cap]) != expected) return false;
                }
            }
        }
    }
    return true;
}

bool testTrainingPixelCullPolicy() {
    using Status = TrainingPixelCullStatus;
    for (uint32_t flags = 0u; flags < 16u; ++flags) {
        for (uint32_t mode = 0u; mode <= 7u; ++mode) {
            for (auto tile : {TrainingTileMode::Fixed8, TrainingTileMode::Fixed16,
                              TrainingTileMode::AdaptiveSpatial}) {
                for (auto forward : {TrainingForwardCompositeMode::Direct,
                                     TrainingForwardCompositeMode::WorkgroupShared}) {
                    const TrainingPixelCullContext context{
                        .rendererReady = (flags & 1u) != 0u,
                        .pipelineAvailable = (flags & 2u) != 0u,
                        .tileMode = tile, .forwardMode = forward,
                        .pixelMode = static_cast<TrainingPixelTo2DGSMode>(mode),
                        .globalPixelGrad = (flags & 4u) != 0u,
                        .rgbLatencyHiding = (flags & 8u) != 0u,
                    };
                    if (resolveTrainingPixelCull(false, context) != Status::Off) return false;
                    const bool expected = flags == 7u && mode == 6u &&
                        tile == TrainingTileMode::Fixed16 &&
                        forward == TrainingForwardCompositeMode::WorkgroupShared;
                    const auto status = resolveTrainingPixelCull(true, context);
                    if ((status == Status::Active) != expected) return false;
                    if (std::string_view(trainingPixelCullStatusName(status)).empty()) return false;
                    if (trainingPixelCullForStep(false, false, status) != expected ||
                        trainingPixelCullForStep(false, true, status) != expected ||
                        trainingPixelCullForStep(true, false, status) ||
                        !trainingPixelCullForStep(true, true, status)) return false;
                }
            }
        }
    }
    TrainingPixelCullContext context{true, true, TrainingTileMode::Fixed16,
        TrainingForwardCompositeMode::WorkgroupShared, TrainingPixelTo2DGSMode::VkSplatTensor,
        true, false};
    const auto expect = [&](Status expected) { return resolveTrainingPixelCull(true, context) == expected; };
    context.rendererReady = false;
    if (!expect(Status::RendererNotReady)) return false;
    context.rendererReady = true;
    context.tileMode = TrainingTileMode::AdaptiveSpatial;
    if (!expect(Status::RequiresFixed16)) return false;
    context.tileMode = TrainingTileMode::Fixed16;
    context.forwardMode = TrainingForwardCompositeMode::Direct;
    if (!expect(Status::RequiresWorkgroupForward)) return false;
    context.forwardMode = TrainingForwardCompositeMode::WorkgroupShared;
    context.pixelMode = TrainingPixelTo2DGSMode::Direct;
    if (!expect(Status::RequiresTensor)) return false;
    context.pixelMode = TrainingPixelTo2DGSMode::VkSplatTensor;
    context.globalPixelGrad = false;
    if (!expect(Status::RequiresSubgroupGlobal)) return false;
    context.globalPixelGrad = true;
    context.rgbLatencyHiding = true;
    if (!expect(Status::RequiresBaselineRgb)) return false;
    context.rgbLatencyHiding = false;
    context.pipelineAvailable = false;
    if (!expect(Status::PipelineUnavailable)) return false;

    TrainingPixelCullRunStats stats;
    stats.record(Status::Off);
    stats.record(Status::Active);
    stats.record(Status::PipelineUnavailable);
    if (stats.completedSteps != 3u || stats.activeSteps != 1u || stats.fallbackSteps != 1u) return false;
    stats = {};
    if (stats.completedSteps || stats.activeSteps || stats.fallbackSteps) return false;
    GaussianTraining training;
    if (training.tensorEarlyCullStatus() != Status::Off) return false;
    training.setTensorEarlyCull(true);
    if (training.tensorEarlyCullStatus() != Status::RendererNotReady) return false;
    training.resetTensorEarlyCullRunStats();
    if (training.tensorEarlyCullRunStats().completedSteps != 0u) return false;
    training.setTensorEarlyCull(false);
    return training.tensorEarlyCullStatus() == Status::Off;
}

bool testTensorSegmentReduction() {
    // CPU invariant test of the shader's actual-lane mapping, independent of
    // subgroup scheduling/rank allocation order. GPU equivalence is tested
    // separately with exported projected gradients.
    constexpr std::array<uint32_t, 8> ranks{7u, 2u, 5u, 0u, 6u, 3u, 1u, 4u};
    for (uint32_t batchCount : {0u, 1u, 7u, 15u, 16u}) {
        for (bool zeroDerivatives : {false, true}) {
            std::array<uint32_t, 16> outputCounts{};
            for (uint32_t rank : ranks) {
                std::array<std::array<float, 9>, 32> values{};
                for (uint32_t lane = 0; lane < 32u; ++lane) {
                    const uint32_t gaussian = rank * 2u + lane / 16u;
                    if (gaussian >= batchCount || zeroDerivatives) continue;
                    for (uint32_t component = 0; component < 9u; ++component) {
                        values[lane][component] = static_cast<float>(
                            1000u * gaussian + 20u * component + lane % 16u + 1u);
                    }
                }
                for (uint32_t offset : {8u, 4u, 2u, 1u}) {
                    const auto previous = values; // Simultaneous shuffle, not in-place peer reads.
                    for (uint32_t lane = 0; lane < 32u; ++lane) {
                        for (uint32_t component = 0; component < 9u; ++component) {
                            values[lane][component] += previous[lane ^ offset][component];
                        }
                    }
                }
                for (uint32_t lane = 0; lane < 32u; lane += 16u) {
                    const uint32_t gaussian = rank * 2u + lane / 16u;
                    for (uint32_t component = 0; component < 9u; ++component) {
                        const float expected = gaussian < batchCount && !zeroDerivatives
                            ? static_cast<float>(16u * (1000u * gaussian + 20u * component) + 136u)
                            : 0.0f;
                        if (values[lane][component] != expected) return false;
                    }
                    if (gaussian < batchCount) ++outputCounts[gaussian];
                }
            }
            for (uint32_t gaussian = 0; gaussian < 16u; ++gaussian) {
                if (outputCounts[gaussian] != (gaussian < batchCount ? 1u : 0u)) return false;
            }
        }
    }
    return true;
}

} // namespace

int main() {
    if (!testFallbacks() || !testSupportedModes() ||
        !testAdaptiveSpatialRequiresOptimizedBackward() || !testTensorSubgroupPolicy() ||
        !testTensorGlobalPixelGradPolicy() ||
        !testTensorRgbLatencyPolicy() ||
        !testTrainingPixelCullPolicy() ||
        !testTensorSegmentReduction()) {
        std::fputs("Backward dispatch policy tests failed\n", stderr);
        return 1;
    }
    return 0;
}
