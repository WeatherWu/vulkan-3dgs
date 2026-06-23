#include "gaussian_training/training_dataset.hpp"

#include <exception>
#include <iostream>

int main(int argc, char** argv) {
    const char* scenePath = argc > 1 ? argv[1] : "data/mipnerf360/bicycle";

    try {
        auto validation = vk_gs::TrainingDatasetLoader::validateMipNeRF360Scene(scenePath, 4);
        if (!validation.valid) {
            std::cerr << validation.message << '\n';
            return 1;
        }

        auto dataset = vk_gs::TrainingDatasetLoader::loadMipNeRF360Scene(scenePath, 4);
        const auto& firstFrame = dataset.frames.front();
        auto firstImage = vk_gs::TrainingDatasetLoader::loadImage(firstFrame);

        std::cout << "scene=" << dataset.sceneRoot.string() << '\n';
        std::cout << "frames=" << dataset.frames.size() << '\n';
        std::cout << "images=" << dataset.imageDirectory.string() << '\n';
        std::cout << "downscale=" << dataset.imageDownscale << '\n';
        std::cout << "first=" << firstFrame.imageName << '\n';
        std::cout << "intrinsics="
                  << firstFrame.fx << ','
                  << firstFrame.fy << ','
                  << firstFrame.cx << ','
                  << firstFrame.cy << '\n';
        std::cout << "image=" << firstImage.width << 'x' << firstImage.height << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }

    return 0;
}
