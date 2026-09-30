#include "training/core/backward/pixel_to_2dgs_dispatcher.hpp"

#include <cstdio>

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

} // namespace

int main() {
    if (!testFallbacks() || !testSupportedModes()) {
        std::fputs("Backward dispatch policy tests failed\n", stderr);
        return 1;
    }
    return 0;
}
