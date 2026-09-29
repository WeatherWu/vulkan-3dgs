#include "training/core/training_image_runtime.hpp"

#include <iostream>

using namespace vulkan3DGS;

int main() {
    TrainingDataset dataset{};
    TrainingCameraFrame first{};
    first.width = 640u;
    first.height = 480u;
    first.imagePath = "images/frame_a.png";
    TrainingCameraFrame second = first;
    second.width = 800u;
    second.height = 600u;
    second.imagePath = "images/frame_b.png";
    dataset.frames = {first, second};

    const std::vector<ImageSourceDesc> sources =
        TrainingImageRuntime::makeSources(dataset);
    if (sources.size() != 2u ||
        sources[0].id != 0u ||
        sources[1].id != 1u ||
        sources[0].path != first.imagePath ||
        sources[1].path != second.imagePath ||
        sources[0].expectedWidth != 640u ||
        sources[1].expectedHeight != 600u ||
        sources[0].format != ImagePixelFormat::Rgba8Unorm ||
        sources[0].colorSpace != ImageColorSpace::LinearUnorm) {
        std::cerr << "Training image source mapping failed\n";
        return 1;
    }
    return 0;
}
