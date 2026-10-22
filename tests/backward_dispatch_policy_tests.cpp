#include "training/core/backward/pixel_to_2dgs_dispatcher.hpp"

#include <cstdio>
#include <array>

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
        useTensorSubgroupReduction(true, Mode::VkSplatTensor, Tile::Fixed8, caps)) return false;
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
                for (Tile tile : {Tile::Fixed8, Tile::Fixed16}) {
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
        !testTensorSubgroupPolicy() ||
        !testTensorGlobalPixelGradPolicy() ||
        !testTensorSegmentReduction()) {
        std::fputs("Backward dispatch policy tests failed\n", stderr);
        return 1;
    }
    return 0;
}
