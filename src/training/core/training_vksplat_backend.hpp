#pragma once

#include "training/core/backward/backward_command_utils.hpp"
#include "training/core/backward/backward_pass_context.hpp"
#include "vulkan/compute_pipeline.hpp"

#include <array>

namespace vulkan3DGS {

// Persistent VkSplat model; canonical storage exists only at explicit state boundaries.
class TrainingVkSplatBackend {
public:
    TrainingVkSplatBackend(vk::Device device, vk::PhysicalDevice physical, vk::Queue queue,
                           uint32_t family);
    static bool supportsLifecycle(vk::PhysicalDevice physical);
    void importCanonical(TrainingBuffers& buffers, uint32_t count);
    void exportCanonical(TrainingBuffers& buffers, uint32_t count);
    void recordProject(const BackwardPassContext& context);
    void recordComposite(const BackwardPassContext& context, bool subtile);
    void recordPixel(const BackwardPassContext& context, bool subgroupGlobal);
    void recordOptimizer(const BackwardPassContext& context);
    void recordValidation(const BackwardPassContext& context);
    vk::DescriptorBufferInfo rectangles() const {
        return rectangles_.getDescriptorInfo();
    }
    uint64_t allocatedBytes() const;
    double boundarySeconds() const {
        return boundarySeconds_;
    }

private:
    enum Stage : uint32_t {
        Project,
        Optimize,
        Import,
        Export,
        Composite,
        Subtile,
        PixelShared,
        PixelGlobal,
        Validate,
        Composite64,
        Subtile64,
        Count
    };
    void ensureCapacity(uint32_t count);
    void initializePipeline(Stage stage);
    void record(Stage stage, const BackwardPassContext& context);
    void boundary(Stage stage, TrainingBuffers& buffers, uint32_t count);
    vk::Device device_;
    vk::PhysicalDevice physical_;
    vk::Queue queue_;
    uint32_t family_;
    uint32_t capacity_ = 0;
    uint32_t subgroupSize_ = 0;
    bool pixelSupported_ = false;
    std::array<std::vector<vk::DescriptorBufferInfo>, Count> lastBindings_;
    std::array<Buffer, 10> model_;
    Buffer rectangles_;
    std::array<ComputePipeline, Count> pipelines_;
    std::array<BackwardDescriptorPool, Count> descriptors_;
    std::array<vk::DescriptorSet, Count> sets_{};
    CommandPool pool_;
    vk::CommandBuffer command_;
    vk::UniqueFence fence_;
    double boundarySeconds_ = 0;
};

} // namespace vulkan3DGS
