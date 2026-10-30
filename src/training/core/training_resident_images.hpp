#pragma once

#include "training/cache/device_image_cache.hpp"
#include "training/core/training_dataset.hpp"
#include "vulkan/buffer.hpp"

#include <memory>
#include <vector>

namespace vulkan3DGS {

class TrainingResidentImages {
public:
    static std::unique_ptr<TrainingResidentImages>
    prepare(vk::Device device, vk::PhysicalDevice physical, vk::Queue transfer,
            uint32_t transferFamily, uint32_t computeFamily, const TrainingDataset& dataset,
            vk::Queue compute = nullptr);
    vk::DescriptorBufferInfo trainingImage(size_t frame);
    DeviceImageCacheStats stats() const;
    double preparationSeconds() const {
        return preparationSeconds_;
    }

private:
    std::vector<Buffer> images_;
    size_t trainingCount_ = 0;
    uint64_t bytes_ = 0;
    uint64_t hits_ = 0;
    double preparationSeconds_ = 0.0;
    vk::PhysicalDevice physical_ = nullptr;
    bool memoryBudgetAvailable_ = false;
};

} // namespace vulkan3DGS
