#pragma once

#include "training/core/backward/backward_command_utils.hpp"
#include "training/core/backward/backward_pass_context.hpp"
#include "vulkan/compute_pipeline.hpp"

#include "training/core/training_vksplat_adc.hpp"
#include <array>

namespace vulkan3DGS {

// Persistent VkSplat model; canonical storage exists only at explicit state boundaries.
class TrainingVkSplatBackend {
public:
    TrainingVkSplatBackend(vk::Device device, vk::PhysicalDevice physical, vk::Queue queue,
                           uint32_t family, bool adcReuse = false, bool projectionSoA = false);
    ~TrainingVkSplatBackend();
    TrainingVkSplatAdc& adc() {
        return *adc_;
    }
    static bool supportsLifecycle(vk::PhysicalDevice physical);
    void importCanonical(TrainingBuffers& buffers, uint32_t count);
    void exportCanonical(TrainingBuffers& buffers, uint32_t count);
    bool projectionSoA() const {
        return projectionSoA_;
    }
    std::array<vk::DescriptorBufferInfo, 7> projectionInfos() const;
    vk::DescriptorBufferInfo officialRadiiInfo() const {
        return projectionSoA_ ? soa_[5].getDescriptorInfo() : officialRadii_.getDescriptorInfo();
    }
    void recordSoAPrefix(const BackwardPassContext& context,
                         const TrainingPrefixPushConstants& prefix, uint32_t groups);
    void recordSoALegacyEmit(const BackwardPassContext& context);
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
    friend class TrainingVkSplatAdc;
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
        ProjectSoA,
        CompositeSoA,
        SubtileSoA,
        PixelSharedSoA,
        PixelGlobalSoA,
        Composite64SoA,
        Subtile64SoA,
        LegacyEmitSoA,
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
    bool projectionSoA_ = false;
    bool optimizedRecording_ = false;
    std::array<std::vector<uint32_t>, Count> cachedBindings_;
    std::array<std::vector<vk::DescriptorBufferInfo>, Count> cachedInfos_;
    std::array<std::vector<vk::WriteDescriptorSet>, Count> cachedWrites_;
    uint32_t soaCapacity_ = 0;
    std::array<Buffer, 7> soa_;
    ComputePipeline soaPrefixPipeline_;
    BackwardDescriptorPool soaPrefixDescriptors_;
    vk::DescriptorSet soaPrefixSet_{};
    std::array<std::vector<vk::DescriptorBufferInfo>, Count> lastBindings_;
    std::array<Buffer, 10> model_;
    Buffer rectangles_;
    Buffer officialRadii_;
    std::unique_ptr<TrainingVkSplatAdc> adc_;
    std::array<ComputePipeline, Count> pipelines_;
    std::array<BackwardDescriptorPool, Count> descriptors_;
    std::array<vk::DescriptorSet, Count> sets_{};
    CommandPool pool_;
    vk::CommandBuffer command_;
    vk::UniqueFence fence_;
    double boundarySeconds_ = 0;
};

} // namespace vulkan3DGS
