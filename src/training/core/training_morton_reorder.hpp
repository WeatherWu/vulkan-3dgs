#pragma once

#include "training/core/backward/backward_command_utils.hpp"
#include "training/core/training_buffers.hpp"
#include "vulkan/compute_pipeline.hpp"

#include <optional>
#include <vk_radix_sort.h>

namespace vulkan3DGS {

struct TrainingMortonResult {
    std::optional<float> gpuMs;
    uint64_t workspaceBytes = 0;
};

// Compute-queue owner. Call only between completed steps, before rebuilding transient inputs.
class TrainingMortonReorder {
public:
    TrainingMortonReorder(vk::Device device, vk::PhysicalDevice physical, vk::Queue queue,
                          uint32_t family);
    ~TrainingMortonReorder();
    TrainingMortonReorder(const TrainingMortonReorder&) = delete;
    TrainingMortonReorder& operator=(const TrainingMortonReorder&) = delete;

    TrainingMortonResult execute(TrainingBuffers& buffers, uint32_t count);

private:
    vk::Device device_;
    vk::PhysicalDevice physical_;
    vk::Queue queue_;
    uint32_t family_;
    ComputePipeline pipeline_;
    BackwardDescriptorPool descriptors_;
    vk::DescriptorSet set_;
    CommandPool pool_;
    vk::CommandBuffer command_;
    vk::UniqueFence fence_;
    vk::UniqueQueryPool queries_;
    struct SorterDeleter {
        void operator()(VrdxSorter_T* sorter) const noexcept;
    };
    std::unique_ptr<VrdxSorter_T, SorterDeleter> sorter_;
    uint32_t timestampBits_ = 0;
    float timestampPeriod_ = 0;
};

} // namespace vulkan3DGS
