#pragma once

#include "training/core/training_dataset.hpp"
#include "training/core/training_types.hpp"

#include <filesystem>
#include <span>
#include <vector>

namespace vulkan3DGS {

struct TrainingModelInitialization {
    std::vector<GaussianTrainParam> parameters;
    bool usedRandomFallback = false;
    float sceneExtent = 1.0f;
};

// Stateless model-domain conversion. GPU parameter download remains outside
// this class; callers provide host-side GaussianTrainParam values for export.
class TrainingModelIO {
public:
    [[nodiscard]] static TrainingModelInitialization initialize(
        const TrainingDataset& dataset,
        const TrainingInitializationConfig& config);
    [[nodiscard]] static std::vector<GaussianTrainParam>
    createSparsePointInitialGaussians(const TrainingDataset& dataset);
    [[nodiscard]] static std::vector<GaussianTrainParam>
    createRandomInitialGaussians(const TrainingDataset& dataset,
                                 const TrainingInitializationConfig& config);
    [[nodiscard]] static float estimateSceneExtent(
        const TrainingDataset& dataset);
    static bool writePly(const std::filesystem::path& path,
                         std::span<const GaussianTrainParam> parameters);
};

} // namespace vulkan3DGS
