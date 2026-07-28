#include "gaussian_training/training_dataset.hpp"

#include "image/image_decoder.hpp"
#include "utils/file_utils.hpp"
#include "utils/logger.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <unordered_map>

#include <glm/gtc/quaternion.hpp>

namespace vulkan3DGS {

namespace {

struct ColmapCameraData {
    uint32_t id = 0;
    int modelId = 0;
    uint64_t width = 0;
    uint64_t height = 0;
    std::vector<double> params;
};

struct ColmapImageData {
    uint32_t id = 0;
    glm::dquat rotationWorldToCamera{1.0, 0.0, 0.0, 0.0};
    glm::dvec3 translationWorldToCamera{0.0};
    uint32_t cameraId = 0;
    std::string name;
};

uint32_t colmapCameraParamCount(int modelId) {
    switch (modelId) {
        case 0: return 3;   // SIMPLE_PINHOLE
        case 1: return 4;   // PINHOLE
        case 2: return 4;   // SIMPLE_RADIAL
        case 3: return 5;   // RADIAL
        case 4: return 8;   // OPENCV
        case 5: return 8;   // OPENCV_FISHEYE
        case 6: return 12;  // FULL_OPENCV
        case 7: return 5;   // FOV
        case 8: return 4;   // SIMPLE_RADIAL_FISHEYE
        case 9: return 5;   // RADIAL_FISHEYE
        case 10: return 12; // THIN_PRISM_FISHEYE
        default:
            throw std::runtime_error("Unsupported COLMAP camera model id: " + std::to_string(modelId));
    }
}

std::unordered_map<uint32_t, ColmapCameraData> readColmapCameras(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open COLMAP cameras file: " + path.string());
    }

    const auto cameraCount = FileUtils::readBinaryValue<uint64_t>(file);
    std::unordered_map<uint32_t, ColmapCameraData> cameras;
    cameras.reserve(static_cast<size_t>(cameraCount));

    for (uint64_t i = 0; i < cameraCount; ++i) {
        ColmapCameraData camera{};
        camera.id = FileUtils::readBinaryValue<uint32_t>(file);
        camera.modelId = FileUtils::readBinaryValue<int32_t>(file);
        camera.width = FileUtils::readBinaryValue<uint64_t>(file);
        camera.height = FileUtils::readBinaryValue<uint64_t>(file);

        const uint32_t paramCount = colmapCameraParamCount(camera.modelId);
        camera.params.resize(paramCount);
        for (uint32_t paramIndex = 0; paramIndex < paramCount; ++paramIndex) {
            camera.params[paramIndex] = FileUtils::readBinaryValue<double>(file);
        }

        cameras.emplace(camera.id, std::move(camera));
    }

    return cameras;
}

std::vector<ColmapImageData> readColmapImages(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open COLMAP images file: " + path.string());
    }

    const auto imageCount = FileUtils::readBinaryValue<uint64_t>(file);
    std::vector<ColmapImageData> images;
    images.reserve(static_cast<size_t>(imageCount));

    for (uint64_t i = 0; i < imageCount; ++i) {
        ColmapImageData image{};
        image.id = FileUtils::readBinaryValue<uint32_t>(file);

        const double qw = FileUtils::readBinaryValue<double>(file);
        const double qx = FileUtils::readBinaryValue<double>(file);
        const double qy = FileUtils::readBinaryValue<double>(file);
        const double qz = FileUtils::readBinaryValue<double>(file);
        image.rotationWorldToCamera = glm::normalize(glm::dquat(qw, qx, qy, qz));

        image.translationWorldToCamera.x = FileUtils::readBinaryValue<double>(file);
        image.translationWorldToCamera.y = FileUtils::readBinaryValue<double>(file);
        image.translationWorldToCamera.z = FileUtils::readBinaryValue<double>(file);
        image.cameraId = FileUtils::readBinaryValue<uint32_t>(file);
        image.name = FileUtils::readNullTerminatedString(file);

        const auto point2DCount = FileUtils::readBinaryValue<uint64_t>(file);
        file.seekg(static_cast<std::streamoff>(point2DCount * (sizeof(double) * 2 + sizeof(uint64_t))),
                   std::ios::cur);
        if (!file) {
            throw std::runtime_error("Failed to skip COLMAP 2D point data");
        }

        images.push_back(std::move(image));
    }

    return images;
}

std::vector<TrainingSparsePoint> readColmapPoints3D(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open COLMAP points3D file: " + path.string());
    }

    const auto pointCount = FileUtils::readBinaryValue<uint64_t>(file);
    std::vector<TrainingSparsePoint> points;
    points.reserve(static_cast<size_t>(pointCount));

    constexpr float inv255 = 1.0f / 255.0f;
    for (uint64_t i = 0; i < pointCount; ++i) {
        (void)FileUtils::readBinaryValue<uint64_t>(file); // point3D_id

        TrainingSparsePoint point{};
        point.position.x = static_cast<float>(FileUtils::readBinaryValue<double>(file));
        point.position.y = static_cast<float>(FileUtils::readBinaryValue<double>(file));
        point.position.z = static_cast<float>(FileUtils::readBinaryValue<double>(file));

        const auto r = FileUtils::readBinaryValue<uint8_t>(file);
        const auto g = FileUtils::readBinaryValue<uint8_t>(file);
        const auto b = FileUtils::readBinaryValue<uint8_t>(file);
        point.color = glm::vec3(static_cast<float>(r) * inv255,
                                static_cast<float>(g) * inv255,
                                static_cast<float>(b) * inv255);
        point.error = static_cast<float>(FileUtils::readBinaryValue<double>(file));

        const auto trackLength = FileUtils::readBinaryValue<uint64_t>(file);
        point.trackLength = static_cast<uint32_t>(std::min<uint64_t>(trackLength, UINT32_MAX));
        file.seekg(static_cast<std::streamoff>(trackLength * (sizeof(uint32_t) + sizeof(uint32_t))),
                   std::ios::cur);
        if (!file) {
            throw std::runtime_error("Failed to skip COLMAP point3D track data");
        }

        points.push_back(point);
    }

    return points;
}

std::filesystem::path findSparseDirectory(const std::filesystem::path& sceneRoot) {
    const std::array<std::filesystem::path, 3> candidates = {
        sceneRoot / "sparse" / "0",
        sceneRoot / "sparse",
        sceneRoot,
    };

    for (const auto& candidate : candidates) {
        if (std::filesystem::exists(candidate / "cameras.bin") &&
            std::filesystem::exists(candidate / "images.bin")) {
            return candidate;
        }
    }

    throw std::runtime_error("MipNeRF360 scene is missing COLMAP sparse cameras.bin/images.bin: " + sceneRoot.string());
}

uint32_t parseImageDownscale(const std::string& directoryName) {
    const std::string prefix = "images_";
    if (directoryName.rfind(prefix, 0) != 0) {
        return 1;
    }

    const std::string value = directoryName.substr(prefix.size());
    if (value.empty() || !std::all_of(value.begin(), value.end(), [](unsigned char ch) { return std::isdigit(ch); })) {
        return 1;
    }

    uint32_t downscale = static_cast<uint32_t>(std::stoul(value));
    return std::max(downscale, 1u);
}

std::filesystem::path chooseImageDirectory(const std::filesystem::path& sceneRoot,
                                           uint32_t preferredDownscale,
                                           uint32_t& selectedDownscale) {
    std::vector<std::filesystem::path> candidates;
    candidates.push_back(preferredDownscale > 1
        ? sceneRoot / ("images_" + std::to_string(preferredDownscale))
        : sceneRoot / "images");
    candidates.push_back(sceneRoot / "images_4");
    candidates.push_back(sceneRoot / "images_2");
    candidates.push_back(sceneRoot / "images_8");
    candidates.push_back(sceneRoot / "images");

    for (const auto& candidate : candidates) {
        if (std::filesystem::is_directory(candidate)) {
            selectedDownscale = parseImageDownscale(candidate.filename().string());
            return candidate;
        }
    }

    throw std::runtime_error("MipNeRF360 scene is missing images/images_N directory: " + sceneRoot.string());
}

bool fileExistsCaseInsensitiveExtension(const std::filesystem::path& path) {
    if (std::filesystem::exists(path)) {
        return true;
    }

    const auto parent = path.parent_path();
    if (!std::filesystem::is_directory(parent)) {
        return false;
    }

    const std::string targetStem = path.stem().string();
    for (const auto& entry : std::filesystem::directory_iterator(parent)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        if (entry.path().stem().string() == targetStem) {
            return true;
        }
    }
    return false;
}

std::filesystem::path resolveImagePath(const std::filesystem::path& imageDirectory,
                                       const std::string& imageName) {
    std::filesystem::path imagePath = imageDirectory / imageName;
    if (std::filesystem::exists(imagePath)) {
        return imagePath;
    }

    const auto parent = imagePath.parent_path();
    const std::string targetStem = imagePath.stem().string();
    if (std::filesystem::is_directory(parent)) {
        for (const auto& entry : std::filesystem::directory_iterator(parent)) {
            if (entry.is_regular_file() && entry.path().stem().string() == targetStem) {
                return entry.path();
            }
        }
    }

    return imagePath;
}

glm::uvec2 readImageDimensions(const std::filesystem::path& path) {
    const ImageInfo info = ImageDecoder::probe(path);
    return glm::uvec2(info.width, info.height);
}

void fillIntrinsics(const ColmapCameraData& camera,
                    float scaleX,
                    float scaleY,
                    float& fx,
                    float& fy,
                    float& cx,
                    float& cy) {
    if (camera.params.empty()) {
        throw std::runtime_error("COLMAP camera has no intrinsic parameters");
    }

    switch (camera.modelId) {
        case 0:
        case 2:
        case 3:
        case 7:
        case 8:
        case 9:
            fx = static_cast<float>(camera.params[0]) * scaleX;
            fy = static_cast<float>(camera.params[0]) * scaleY;
            cx = static_cast<float>(camera.params[1]) * scaleX;
            cy = static_cast<float>(camera.params[2]) * scaleY;
            break;
        default:
            fx = static_cast<float>(camera.params[0]) * scaleX;
            fy = static_cast<float>(camera.params[1]) * scaleY;
            cx = static_cast<float>(camera.params[2]) * scaleX;
            cy = static_cast<float>(camera.params[3]) * scaleY;
            break;
    }
}

glm::mat4 makeWorldToCamera(const ColmapImageData& image) {
    const glm::dmat3 rotation = glm::mat3_cast(image.rotationWorldToCamera);
    glm::mat4 worldToCamera(1.0f);
    for (int column = 0; column < 3; ++column) {
        for (int row = 0; row < 3; ++row) {
            worldToCamera[column][row] = static_cast<float>(rotation[column][row]);
        }
    }
    worldToCamera[3][0] = static_cast<float>(image.translationWorldToCamera.x);
    worldToCamera[3][1] = static_cast<float>(image.translationWorldToCamera.y);
    worldToCamera[3][2] = static_cast<float>(image.translationWorldToCamera.z);
    return worldToCamera;
}

} // namespace

TrainingDataset TrainingDatasetLoader::loadMipNeRF360Scene(const std::filesystem::path& sceneRoot,
                                                           uint32_t preferredDownscale) {
    if (!std::filesystem::is_directory(sceneRoot)) {
        throw std::runtime_error("MipNeRF360 scene path is not a directory: " + sceneRoot.string());
    }

    uint32_t selectedDownscale = 1;
    const auto imageDirectory = chooseImageDirectory(sceneRoot, preferredDownscale, selectedDownscale);
    const auto sparseDirectory = findSparseDirectory(sceneRoot);
    auto cameras = readColmapCameras(sparseDirectory / "cameras.bin");
    auto images = readColmapImages(sparseDirectory / "images.bin");

    std::sort(images.begin(), images.end(), [](const ColmapImageData& lhs, const ColmapImageData& rhs) {
        return lhs.name < rhs.name;
    });

    TrainingDataset dataset{};
    dataset.sceneRoot = sceneRoot;
    dataset.imageDirectory = imageDirectory;
    dataset.imageDownscale = selectedDownscale;
    dataset.frames.reserve(images.size());
    glm::uvec2 datasetImageSize(0u);

    for (const auto& image : images) {
        auto cameraIt = cameras.find(image.cameraId);
        if (cameraIt == cameras.end()) {
            LOG_WARN("Skipping image {} because COLMAP camera {} is missing", image.name, image.cameraId);
            continue;
        }

        const auto resolvedImagePath = resolveImagePath(imageDirectory, image.name);
        if (!fileExistsCaseInsensitiveExtension(resolvedImagePath)) {
            LOG_WARN("Skipping image because file is missing: {}", resolvedImagePath.string());
            continue;
        }

        const ColmapCameraData& camera = cameraIt->second;
        const glm::uvec2 imageSize = readImageDimensions(resolvedImagePath);
        if (datasetImageSize.x == 0u || datasetImageSize.y == 0u) {
            datasetImageSize = imageSize;
        } else if (imageSize != datasetImageSize) {
            throw std::runtime_error("Training images must have a consistent size. Expected " +
                                     std::to_string(datasetImageSize.x) + "x" +
                                     std::to_string(datasetImageSize.y) + " but found " +
                                     std::to_string(imageSize.x) + "x" +
                                     std::to_string(imageSize.y) + ": " +
                                     resolvedImagePath.string());
        }

        const float scaleX = static_cast<float>(imageSize.x) /
                             static_cast<float>(std::max<uint64_t>(camera.width, 1u));
        const float scaleY = static_cast<float>(imageSize.y) /
                             static_cast<float>(std::max<uint64_t>(camera.height, 1u));

        TrainingCameraFrame frame{};
        frame.imageId = image.id;
        frame.cameraId = image.cameraId;
        frame.width = imageSize.x;
        frame.height = imageSize.y;
        fillIntrinsics(camera, scaleX, scaleY, frame.fx, frame.fy, frame.cx, frame.cy);
        frame.worldToCamera = makeWorldToCamera(image);
        frame.cameraToWorld = glm::inverse(frame.worldToCamera);
        frame.position = glm::vec3(frame.cameraToWorld[3]);
        frame.imagePath = resolvedImagePath;
        frame.imageName = image.name;
        dataset.frames.push_back(std::move(frame));
    }

    if (dataset.frames.empty()) {
        throw std::runtime_error("MipNeRF360 scene contains no loadable training frames: " + sceneRoot.string());
    }

    const auto points3DPath = sparseDirectory / "points3D.bin";
    if (std::filesystem::exists(points3DPath)) {
        dataset.sparsePoints = readColmapPoints3D(points3DPath);
    }

    LOG_INFO("Loaded MipNeRF360 scene {} with {} frames and {} sparse points from {}",
             sceneRoot.string(), dataset.frames.size(), dataset.sparsePoints.size(), imageDirectory.string());
    return dataset;
}

TrainingDatasetValidation TrainingDatasetLoader::validateMipNeRF360Scene(const std::filesystem::path& sceneRoot,
                                                                         uint32_t preferredDownscale) {
    TrainingDatasetValidation validation{};
    try {
        TrainingDataset dataset = loadMipNeRF360Scene(sceneRoot, preferredDownscale);
        const auto& firstFrame = dataset.frames.front();
        validation.valid = true;
        validation.message = "Dataset is valid";
        validation.frameCount = static_cast<uint32_t>(dataset.frames.size());
        validation.width = firstFrame.width;
        validation.height = firstFrame.height;
        validation.downscale = dataset.imageDownscale;
        validation.imageDirectory = dataset.imageDirectory;
    } catch (const std::exception& error) {
        validation.valid = false;
        validation.message = error.what();
    }
    return validation;
}

} // namespace vulkan3DGS
