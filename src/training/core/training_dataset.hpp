#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <glm/glm.hpp>

namespace vulkan3DGS {

struct TrainingCameraFrame {
    uint32_t imageId = 0;
    uint32_t cameraId = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    float fx = 0.0f;
    float fy = 0.0f;
    float cx = 0.0f;
    float cy = 0.0f;
    glm::mat4 worldToCamera{1.0f};
    glm::mat4 cameraToWorld{1.0f};
    glm::vec3 position{0.0f};
    std::filesystem::path imagePath;
    std::string imageName;
};

struct TrainingSparsePoint {
    glm::vec3 position{0.0f};
    glm::vec3 color{1.0f};
    float error = 0.0f;
    uint32_t trackLength = 0;
};

struct TrainingDataset {
    std::filesystem::path sceneRoot;
    std::filesystem::path imageDirectory;
    uint32_t imageDownscale = 1;
    std::vector<TrainingCameraFrame> frames;
    std::vector<TrainingSparsePoint> sparsePoints;

    bool empty() const { return frames.empty(); }
    size_t size() const { return frames.size(); }
};

struct TrainingDatasetValidation {
    bool valid = false;
    std::string message;
    uint32_t frameCount = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t downscale = 1;
    std::filesystem::path imageDirectory;
};

class TrainingDatasetLoader {
public:
    static TrainingDatasetValidation validateMipNeRF360Scene(const std::filesystem::path& sceneRoot,
                                                             uint32_t preferredDownscale = 4);
    static TrainingDataset loadMipNeRF360Scene(const std::filesystem::path& sceneRoot,
                                               uint32_t preferredDownscale = 4);

private:
    struct ColmapCamera;
    struct ColmapImage;
};

} // namespace vulkan3DGS
