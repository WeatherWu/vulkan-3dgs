#pragma once
#include "training/core/backward/backward_command_utils.hpp"
#include "training/core/training_buffers.hpp"
#include "vulkan/compute_pipeline.hpp"
#include <array>
#include <vk_radix_sort.h>

namespace vulkan3DGS {
// Forward-renderer owned experimental workspace. Call with no pending commands using its buffers.
class TrainingFasterSort {
public:
    TrainingFasterSort(vk::Device device, vk::PhysicalDevice physical, vk::Queue queue,
                       uint32_t family);
    ~TrainingFasterSort();
    void prepare(vk::CommandBuffer command, TrainingBuffers& buffers,
                 const TrainingPushConstants& push);
    void build(vk::CommandBuffer command, TrainingBuffers& buffers,
               const TrainingPushConstants& push, VrdxSorter sorter);
    uint32_t participatingCount() const {
        return visibleCount_;
    }
    uint32_t capacity() const {
        return capacity_;
    }

private:
    void ensureCapacity(uint32_t count);
    void bindInputs(TrainingBuffers& buffers);
    void prefix(vk::CommandBuffer command, uint32_t bank, uint32_t count);
    void dispatch(vk::CommandBuffer command, size_t pipeline, uint32_t count,
                  const TrainingPushConstants& push);
    static void barrier(vk::CommandBuffer command);
    vk::Device device_;
    vk::PhysicalDevice physical_;
    vk::Queue queue_;
    uint32_t family_;
    uint32_t capacity_ = 0, visibleCount_ = 0, scratchWords_ = 0;
    bool prepared_ = false;
    TrainingPushConstants preparedPush_{};
    std::array<Buffer, 2> ranges_, counters_;
    Buffer depth_, ids_, prefixScratch_, readback_, sortStorage_;
    void* mapped_ = nullptr;
    std::array<ComputePipeline, 4> pipelines_;
    std::array<ComputePipeline, 5> prefixPipelines_;
    BackwardDescriptorPool descriptors_;
    std::array<vk::DescriptorSet, 4> sets_{};
    std::array<std::array<vk::DescriptorSet, 5>, 2> prefixSets_{};
};
} // namespace vulkan3DGS
