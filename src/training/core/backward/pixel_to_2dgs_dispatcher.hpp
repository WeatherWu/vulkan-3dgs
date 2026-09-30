#pragma once

#include "training/core/backward/backward_command_utils.hpp"
#include "training/core/backward/backward_pass_context.hpp"
#include "vulkan/compute_pipeline.hpp"

namespace vulkan3DGS {

struct PixelTo2DGSCapabilities {
    uint32_t subgroupSize = 0;
    bool subgroup = false;
    bool tileGaussian = false;
    bool vkSplatPerSplat = false;
    bool vkSplatTensor = false;
};

[[nodiscard]] TrainingPixelTo2DGSMode
resolvePixelTo2DGSMode(TrainingPixelTo2DGSMode requested,
                       const PixelTo2DGSCapabilities& capabilities) noexcept;

class PixelTo2DGSDispatcher {
public:
    void initialize(vk::Device device, vk::PhysicalDevice physicalDevice);
    void cleanup();

    void setRequestedMode(TrainingPixelTo2DGSMode mode) {
        requestedMode_ = mode;
    }
    [[nodiscard]] TrainingPixelTo2DGSMode activeMode() const noexcept;
    [[nodiscard]] const PixelTo2DGSCapabilities& capabilities() const noexcept {
        return capabilities_;
    }
    void record(const BackwardPassContext& context);

private:
    void recordTileGaussian(const BackwardPassContext& context);

    vk::Device device_ = nullptr;
    ComputePipeline directPipeline_;
    ComputePipeline workgroupPipeline_;
    ComputePipeline subgroupPipeline_;
    ComputePipeline adaptivePipeline_;
    ComputePipeline tileGaussianPipeline_;
    ComputePipeline vkSplatPerSplatPipeline_;
    ComputePipeline vkSplatTensorPipeline_;
    BackwardDescriptorPool descriptorPool_;
    vk::DescriptorSet directDescriptorSet_ = nullptr;
    vk::DescriptorSet workgroupDescriptorSet_ = nullptr;
    vk::DescriptorSet subgroupDescriptorSet_ = nullptr;
    vk::DescriptorSet adaptiveDescriptorSet_ = nullptr;
    vk::DescriptorSet tileGaussianDescriptorSet_ = nullptr;
    vk::DescriptorSet vkSplatPerSplatDescriptorSet_ = nullptr;
    vk::DescriptorSet vkSplatTensorDescriptorSet_ = nullptr;
    TrainingPixelTo2DGSMode requestedMode_ = TrainingPixelTo2DGSMode::Auto;
    PixelTo2DGSCapabilities capabilities_{};
};

} // namespace vulkan3DGS
