#include "training/core/training_model_io.hpp"

#include "utils/logger.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>

namespace vulkan3DGS {
namespace {

float logit(float value) {
    constexpr float epsilon = 1e-6f;
    const float clamped = std::clamp(value, epsilon, 1.0f - epsilon);
    return std::log(clamped / (1.0f - clamped));
}

glm::vec3 colorToSH0(const glm::vec3& color) {
    constexpr float shC0 = 0.28209479177387814f;
    return (color - glm::vec3(0.5f)) / shC0;
}

struct KdNode {
    uint32_t pointIndex = 0;
    int axis = 0;
    int left = -1;
    int right = -1;
};

int buildKdTreeRecursive(const std::vector<TrainingSparsePoint>& points,
                         std::vector<uint32_t>& indices,
                         std::vector<KdNode>& nodes,
                         size_t begin,
                         size_t end,
                         int depth) {
    if (begin >= end) return -1;
    const int axis = depth % 3;
    const size_t middle = begin + (end - begin) / 2u;
    std::nth_element(indices.begin() + static_cast<std::ptrdiff_t>(begin),
                     indices.begin() + static_cast<std::ptrdiff_t>(middle),
                     indices.begin() + static_cast<std::ptrdiff_t>(end),
                     [&](uint32_t lhs, uint32_t rhs) {
                         return points[lhs].position[axis] <
                                points[rhs].position[axis];
                     });
    const int nodeIndex = static_cast<int>(nodes.size());
    nodes.push_back(KdNode{indices[middle], axis, -1, -1});
    nodes[nodeIndex].left = buildKdTreeRecursive(
        points, indices, nodes, begin, middle, depth + 1);
    nodes[nodeIndex].right = buildKdTreeRecursive(
        points, indices, nodes, middle + 1u, end, depth + 1);
    return nodeIndex;
}

void queryNearestSquaredDistances(
    const std::vector<TrainingSparsePoint>& points,
    const std::vector<KdNode>& nodes,
    int nodeIndex,
    uint32_t queryIndex,
    std::priority_queue<float>& nearestSquaredDistances,
    uint32_t neighborCount) {
    if (nodeIndex < 0) return;

    const KdNode& node = nodes[static_cast<size_t>(nodeIndex)];
    const glm::vec3 query = points[queryIndex].position;
    const glm::vec3 candidate = points[node.pointIndex].position;
    if (node.pointIndex != queryIndex) {
        const float distanceSquared = glm::dot(query - candidate,
                                                query - candidate);
        if (nearestSquaredDistances.size() < neighborCount) {
            nearestSquaredDistances.push(distanceSquared);
        } else if (distanceSquared < nearestSquaredDistances.top()) {
            nearestSquaredDistances.pop();
            nearestSquaredDistances.push(distanceSquared);
        }
    }

    const float axisDelta = query[node.axis] - candidate[node.axis];
    const int nearChild = axisDelta <= 0.0f ? node.left : node.right;
    const int farChild = axisDelta <= 0.0f ? node.right : node.left;
    queryNearestSquaredDistances(points, nodes, nearChild, queryIndex,
                                 nearestSquaredDistances, neighborCount);

    const float worstDistanceSquared = nearestSquaredDistances.empty()
        ? std::numeric_limits<float>::infinity()
        : nearestSquaredDistances.top();
    if (nearestSquaredDistances.size() < neighborCount ||
        axisDelta * axisDelta < worstDistanceSquared) {
        queryNearestSquaredDistances(points, nodes, farChild, queryIndex,
                                     nearestSquaredDistances, neighborCount);
    }
}

std::vector<float> estimateSparsePointScaleDistances(
    const std::vector<TrainingSparsePoint>& points,
    float fallbackScale) {
    std::vector<float> scales(points.size(), fallbackScale);
    if (points.size() < 2u) return scales;

    std::vector<uint32_t> indices(points.size());
    std::iota(indices.begin(), indices.end(), 0u);
    std::vector<KdNode> nodes;
    nodes.reserve(points.size());
    const int root = buildKdTreeRecursive(
        points, indices, nodes, 0u, indices.size(), 0);
    const uint32_t neighborCount = std::min<uint32_t>(
        3u, static_cast<uint32_t>(points.size() - 1u));

    for (uint32_t index = 0; index < points.size(); ++index) {
        std::priority_queue<float> nearestDistances;
        queryNearestSquaredDistances(points, nodes, root, index,
                                     nearestDistances, neighborCount);
        if (nearestDistances.empty()) continue;

        float sum = 0.0f;
        uint32_t count = 0u;
        while (!nearestDistances.empty()) {
            sum += nearestDistances.top();
            nearestDistances.pop();
            ++count;
        }
        scales[index] = std::sqrt(std::max(
            sum / static_cast<float>(std::max(count, 1u)), 1e-7f));
    }
    return scales;
}

} // namespace

TrainingModelInitialization TrainingModelIO::initialize(
    const TrainingDataset& dataset,
    const TrainingInitializationConfig& config) {
    TrainingModelInitialization result{};
    result.usedRandomFallback = dataset.sparsePoints.empty();
    result.parameters = result.usedRandomFallback
        ? createRandomInitialGaussians(dataset, config)
        : createSparsePointInitialGaussians(dataset);
    result.sceneExtent = estimateSceneExtent(dataset);
    return result;
}

std::vector<GaussianTrainParam>
TrainingModelIO::createSparsePointInitialGaussians(
    const TrainingDataset& dataset) {
    std::vector<GaussianTrainParam> parameters;
    parameters.reserve(dataset.sparsePoints.size());

    glm::vec3 minimum(std::numeric_limits<float>::max());
    glm::vec3 maximum(std::numeric_limits<float>::lowest());
    for (const TrainingSparsePoint& point : dataset.sparsePoints) {
        minimum = glm::min(minimum, point.position);
        maximum = glm::max(maximum, point.position);
    }
    const float radius = std::max(
        glm::length(maximum - minimum) * 0.5f, 1.0f);
    const float count = std::max(
        static_cast<float>(dataset.sparsePoints.size()), 1.0f);
    const float fallbackScale = std::clamp(
        radius / std::cbrt(count), 1e-4f, radius * 0.05f);
    const std::vector<float> scales = estimateSparsePointScaleDistances(
        dataset.sparsePoints, fallbackScale);
    constexpr float initialOpacity = 0.1f;
    const float rawOpacity = logit(initialOpacity);

    for (size_t index = 0; index < dataset.sparsePoints.size(); ++index) {
        const TrainingSparsePoint& point = dataset.sparsePoints[index];
        const float rawScale = std::log(scales[index]);
        GaussianTrainParam parameter{};
        parameter.positionOpacity = glm::vec4(point.position, rawOpacity);
        parameter.scale = glm::vec4(glm::vec3(rawScale), 0.0f);
        parameter.rotation = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
        setTrainingSHCoefficient(parameter, 0u, colorToSH0(point.color));
        parameters.push_back(parameter);
    }
    return parameters;
}

std::vector<GaussianTrainParam>
TrainingModelIO::createRandomInitialGaussians(
    const TrainingDataset& dataset,
    const TrainingInitializationConfig& config) {
    if (!config.allowRandomFallback) {
        throw std::runtime_error(
            "Dataset has no COLMAP sparse points3D.bin for Gaussian initialization");
    }

    const uint32_t gaussianCount = std::max(config.randomGaussianCount, 1u);
    std::vector<GaussianTrainParam> parameters;
    parameters.reserve(gaussianCount);

    glm::vec3 minimum(std::numeric_limits<float>::max());
    glm::vec3 maximum(std::numeric_limits<float>::lowest());
    glm::vec3 center(0.0f);
    for (const TrainingCameraFrame& frame : dataset.frames) {
        minimum = glm::min(minimum, frame.position);
        maximum = glm::max(maximum, frame.position);
        center += frame.position;
    }
    center /= std::max(static_cast<float>(dataset.frames.size()), 1.0f);

    const float cameraRadius = std::max(
        glm::length(maximum - minimum) * 0.5f, 1.0f);
    const float sceneRadius = std::max(
        cameraRadius * std::max(config.sceneRadiusScale, 0.01f), 1.0f);
    const float initialScale = std::clamp(
        sceneRadius / std::cbrt(static_cast<float>(gaussianCount)),
        1e-4f, sceneRadius * 0.05f);
    const float rawOpacity = logit(config.initialOpacity);
    const float rawScale = std::log(std::max(initialScale, 1e-6f));

    std::mt19937 rng(config.randomSeed);
    std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
    std::uniform_real_distribution<float> colorJitter(-0.05f, 0.05f);
    for (uint32_t index = 0; index < gaussianCount; ++index) {
        GaussianTrainParam parameter{};
        const glm::vec3 offset(unit(rng), unit(rng), unit(rng));
        parameter.positionOpacity =
            glm::vec4(center + offset * sceneRadius, rawOpacity);
        parameter.scale = glm::vec4(glm::vec3(rawScale), 0.0f);
        parameter.rotation = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
        const glm::vec3 color = glm::clamp(
            glm::vec3(0.5f + colorJitter(rng),
                      0.5f + colorJitter(rng),
                      0.5f + colorJitter(rng)),
            glm::vec3(0.0f), glm::vec3(1.0f));
        setTrainingSHCoefficient(parameter, 0u, colorToSH0(color));
        parameters.push_back(parameter);
    }
    return parameters;
}

float TrainingModelIO::estimateSceneExtent(const TrainingDataset& dataset) {
    if (!dataset.frames.empty()) {
        glm::vec3 center(0.0f);
        for (const TrainingCameraFrame& frame : dataset.frames) {
            center += frame.position;
        }
        center /= std::max(static_cast<float>(dataset.frames.size()), 1.0f);

        float diagonal = 0.0f;
        for (const TrainingCameraFrame& frame : dataset.frames) {
            diagonal = std::max(
                diagonal, glm::length(frame.position - center));
        }
        return std::max(diagonal * 1.1f, 1e-6f);
    }

    if (!dataset.sparsePoints.empty()) {
        glm::vec3 minimum(std::numeric_limits<float>::max());
        glm::vec3 maximum(std::numeric_limits<float>::lowest());
        for (const TrainingSparsePoint& point : dataset.sparsePoints) {
            minimum = glm::min(minimum, point.position);
            maximum = glm::max(maximum, point.position);
        }
        return std::max(glm::length(maximum - minimum), 1e-6f);
    }
    return 1.0f;
}

bool TrainingModelIO::writePly(
    const std::filesystem::path& requestedPath,
    std::span<const GaussianTrainParam> parameters) {
    std::filesystem::path outputPath = requestedPath;
    if (outputPath.extension().empty()) outputPath += ".ply";
    if (outputPath.has_parent_path()) {
        std::filesystem::create_directories(outputPath.parent_path());
    }

    std::ofstream file(outputPath, std::ios::binary);
    if (!file.is_open()) {
        LOG_ERROR("Failed to create training PLY file: {}", outputPath.string());
        return false;
    }
    file << "ply\n";
    file << "format binary_little_endian 1.0\n";
    file << "element vertex " << parameters.size() << "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "property float f_dc_0\n";
    file << "property float f_dc_1\n";
    file << "property float f_dc_2\n";
    for (int index = 0; index < 45; ++index) {
        file << "property float f_rest_" << index << "\n";
    }
    file << "property float opacity\n";
    file << "property float scale_0\n";
    file << "property float scale_1\n";
    file << "property float scale_2\n";
    file << "property float rot_0\n";
    file << "property float rot_1\n";
    file << "property float rot_2\n";
    file << "property float rot_3\n";
    file << "end_header\n";

    for (const GaussianTrainParam& gaussian : parameters) {
        const glm::vec3 position(gaussian.positionOpacity);
        file.write(reinterpret_cast<const char*>(&position.x), sizeof(float));
        file.write(reinterpret_cast<const char*>(&position.y), sizeof(float));
        file.write(reinterpret_cast<const char*>(&position.z), sizeof(float));

        const glm::vec3 shDc = trainingSHCoefficient(gaussian, 0u);
        const float fDc[3] = {shDc.x, shDc.y, shDc.z};
        file.write(reinterpret_cast<const char*>(fDc), sizeof(fDc));

        float fRest[45]{};
        int outputIndex = 0;
        for (int channel = 0; channel < 3; ++channel) {
            for (int basis = 1; basis < 16; ++basis) {
                fRest[outputIndex++] = trainingSHCoefficient(
                    gaussian, static_cast<uint32_t>(basis))[channel];
            }
        }
        file.write(reinterpret_cast<const char*>(fRest), sizeof(fRest));
        file.write(reinterpret_cast<const char*>(&gaussian.positionOpacity.w),
                   sizeof(float));
        const float logScale[3] = {
            gaussian.scale.x, gaussian.scale.y, gaussian.scale.z};
        file.write(reinterpret_cast<const char*>(logScale), sizeof(logScale));
        file.write(reinterpret_cast<const char*>(&gaussian.rotation.w),
                   sizeof(float));
        file.write(reinterpret_cast<const char*>(&gaussian.rotation.x),
                   sizeof(float));
        file.write(reinterpret_cast<const char*>(&gaussian.rotation.y),
                   sizeof(float));
        file.write(reinterpret_cast<const char*>(&gaussian.rotation.z),
                   sizeof(float));
    }

    if (!file.good()) {
        LOG_ERROR("Failed while writing training PLY file: {}", outputPath.string());
        return false;
    }
    LOG_INFO("Exported trained Gaussian model to {} ({} points)",
             outputPath.string(), parameters.size());
    return true;
}

} // namespace vulkan3DGS
