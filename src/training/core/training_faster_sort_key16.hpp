#pragma once
#include "training/core/backward/backward_command_utils.hpp"
#include "training/core/training_buffers.hpp"
#include "vulkan/compute_pipeline.hpp"
#include "vulkan/radix_key16.hpp"

namespace vulkan3DGS {
class TrainingFasterSort;
class TrainingFasterSortKey16 {
public:
    void build(TrainingFasterSort& owner, vk::CommandBuffer command, TrainingBuffers& buffers,
               const TrainingPushConstants& push, VrdxSorter sorter);
    uint64_t allocatedBytes() const {
        return key16_ ? key16_->allocatedBytes() : 0;
    }

private:
    std::unique_ptr<RadixKey16> key16_;
    std::array<ComputePipeline, 2> key16Pipelines_;
    BackwardDescriptorPool key16Descriptors_;
    std::array<vk::DescriptorSet, 2> key16Sets_{};
};
} // namespace vulkan3DGS
