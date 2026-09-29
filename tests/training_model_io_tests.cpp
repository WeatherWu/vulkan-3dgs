#include "training/core/training_model_io.hpp"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace vulkan3DGS;

namespace {

TrainingDataset cameraDataset() {
    TrainingDataset dataset{};
    TrainingCameraFrame first{};
    first.width = 16u;
    first.height = 16u;
    first.position = glm::vec3(0.0f);
    TrainingCameraFrame second = first;
    second.position = glm::vec3(2.0f, 0.0f, 0.0f);
    dataset.frames = {first, second};
    return dataset;
}

bool near(float lhs, float rhs, float tolerance = 1e-5f) {
    return std::abs(lhs - rhs) <= tolerance;
}

bool testSparseInitialization() {
    TrainingDataset dataset = cameraDataset();
    dataset.sparsePoints = {
        {glm::vec3(-1.0f, 0.0f, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f)},
        {glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f)},
        {glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f)},
    };
    const TrainingModelInitialization result =
        TrainingModelIO::initialize(dataset, TrainingInitializationConfig{});
    if (result.usedRandomFallback || result.parameters.size() != 3u ||
        !near(result.sceneExtent, 1.1f)) {
        return false;
    }
    const GaussianTrainParam& first = result.parameters.front();
    return first.positionOpacity.x == -1.0f &&
        first.rotation.w == 1.0f &&
        near(first.positionOpacity.w, std::log(0.1f / 0.9f));
}

bool testRandomInitializationDeterminism() {
    const TrainingDataset dataset = cameraDataset();
    TrainingInitializationConfig config{};
    config.randomGaussianCount = 5u;
    config.randomSeed = 91u;
    config.initialOpacity = 0.2f;
    const TrainingModelInitialization first =
        TrainingModelIO::initialize(dataset, config);
    const TrainingModelInitialization second =
        TrainingModelIO::initialize(dataset, config);
    if (!first.usedRandomFallback || first.parameters.size() != 5u ||
        first.parameters.size() != second.parameters.size()) {
        return false;
    }
    for (size_t index = 0; index < first.parameters.size(); ++index) {
        if (std::memcmp(&first.parameters[index], &second.parameters[index],
                        sizeof(GaussianTrainParam)) != 0) {
            return false;
        }
    }

    config.allowRandomFallback = false;
    try {
        (void)TrainingModelIO::initialize(dataset, config);
    } catch (const std::runtime_error&) {
        return true;
    }
    return false;
}

bool testPlyOutput() {
    TrainingDataset dataset = cameraDataset();
    TrainingInitializationConfig config{};
    config.randomGaussianCount = 1u;
    const TrainingModelInitialization model =
        TrainingModelIO::initialize(dataset, config);
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        "vulkan-3dgs-model-io-test.ply";
    const bool written = TrainingModelIO::writePly(path, model.parameters);
    std::string header;
    {
        std::ifstream input(path, std::ios::binary);
        std::getline(input, header);
    }
    std::filesystem::remove(path);
    return written && header == "ply";
}

} // namespace

int main() {
    if (!testSparseInitialization()) {
        std::cerr << "Sparse model initialization failed\n";
        return 1;
    }
    if (!testRandomInitializationDeterminism()) {
        std::cerr << "Random model initialization failed\n";
        return 1;
    }
    if (!testPlyOutput()) {
        std::cerr << "PLY model output failed\n";
        return 1;
    }
    return 0;
}
