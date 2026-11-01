#pragma once

#include "training/core/training_types.hpp"
#include "vulkan/buffer.hpp"
#include "vulkan/compute_pipeline.hpp"
#include <array>
#include <map>
#include <random>
#include <stdexcept>

namespace vulkan3DGS {
class TrainingVkSplatBackend;
struct BackwardPassContext;
struct VkSplatAdcResult {
    TrainingDensificationStats stats{};
    uint32_t input = 0, capacity = 0, seed = 0;
    bool reset = false;
    double cpuMs = 0, gpuMs = 0;
    std::vector<double> gpuBatches;
};

class VkSplatAdcEmptyResult : public std::runtime_error {
public:
    explicit VkSplatAdcEmptyResult(VkSplatAdcResult value)
        : std::runtime_error(
              "VkSplat official keep mask is empty; stop and explicitly reload the dataset"),
          result(std::move(value)) {}
    VkSplatAdcResult result;
};

// Official default strategy operating directly on the native model and moments.
class TrainingVkSplatAdc {
public:
    explicit TrainingVkSplatAdc(TrainingVkSplatBackend& backend);
    void prepare(const TrainingDensificationConfig& config, float extent, uint32_t iteration,
                 bool benchmark);
    void ensureStatistics(uint32_t capacity);
    void recordStatistics(const BackwardPassContext& context);
    VkSplatAdcResult execute(uint32_t count, bool refine, bool reset);
    vk::DescriptorBufferInfo gradients() const {
        return grad_.getDescriptorInfo();
    }
    vk::DescriptorBufferInfo radii() const {
        return radius_.getDescriptorInfo();
    }
    uint64_t allocatedBytes() const;
    bool inspectOpacityReset(uint32_t count, float minimum);
    void markFailed() {
        failed_ = true;
    }
    bool failed() const {
        return failed_;
    }

private:
    struct Uniforms {
        uint32_t count = 0, oldCount = 0, seed = 0, width = 0, height = 0, step = 0;
        float pruneOpacity = 0, growGradient = 0, growScale = 0, growRadius = 0;
        float pruneScale = 0, pruneRadius = 0;
        int32_t scale2dStop = 0, start = 0, stop = 0, resetEvery = 0, every = 0, pause = 0;
    } u_;
    static_assert(sizeof(Uniforms) == 72);
    struct Pipeline {
        ComputePipeline pipeline;
    };
    Buffer allocate(uint64_t bytes, bool host = false);
    void begin();
    void submit();
    void barrier(vk::CommandBuffer command);
    void dispatch(const std::string& name, const std::vector<vk::DescriptorBufferInfo>& buffers,
                  uint32_t groups, const void* push, uint32_t pushSize);
    void scan(Buffer& input, Buffer& output, uint32_t count);
    uint32_t readCount(Buffer& prefix, uint32_t count);
    void grow(uint32_t count);
    TrainingVkSplatBackend& core_;
    vk::Device device_;
    vk::PhysicalDevice physical_;
    vk::Queue queue_;
    uint32_t family_;
    Buffer grad_, radius_;
    Buffer readback_;
    uint32_t statisticsCapacity_ = 0;
    std::map<std::string, Pipeline> pipelines_;
    vk::UniqueDescriptorPool descriptors_;
    ComputePipeline statisticsPipeline_;
    vk::UniqueDescriptorPool statisticsPool_;
    vk::DescriptorSet statisticsSet_;
    CommandPool commandPool_;
    vk::CommandBuffer command_;
    vk::UniqueFence fence_;
    vk::UniqueQueryPool timestamps_;
    float timestampPeriod_ = 0;
    std::vector<Buffer> scratch_, retired_;
    TrainingDensificationConfig config_{};
    std::default_random_engine rng_{42};
    bool benchmark_ = true, failed_ = false;
    double gpuMs_ = 0;
    std::vector<double> gpuBatches_;
};
} // namespace vulkan3DGS
