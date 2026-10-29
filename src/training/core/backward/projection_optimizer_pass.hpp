#pragma once

#include "training/core/backward/backward_command_utils.hpp"
#include "training/core/backward/backward_pass_context.hpp"
#include "training/core/training_direct_projection.hpp"
#include "vulkan/compute_pipeline.hpp"
#include <filesystem>

namespace vulkan3DGS {

class ProjectionOptimizerPass {
public:
    void initialize(vk::Device device, const std::filesystem::path& shaderDirectory = "shaders");
    void cleanup();

    [[nodiscard]] bool fusedEnabled() const noexcept {
        return fusedEnabled_;
    }
    void recordProjection(const BackwardPassContext& context);
    void recordProjectionAndOptimize(const BackwardPassContext& context);
    void recordOptimizer(const BackwardPassContext& context);
    void releaseDirectResources() {
        directProjection_.reset();
    }
    bool hasDirectResources() const {
        return bool(directProjection_);
    }

private:
    void recordDirect(const BackwardPassContext& context, bool fused);
    std::unique_ptr<TrainingDirectProjection> directProjection_;
    std::filesystem::path directShaderDirectory_;
    vk::Device device_ = nullptr;
    ComputePipeline projectionPipeline_;
    ComputePipeline fusedPipeline_;
    ComputePipeline optimizerPipeline_;
    BackwardDescriptorPool descriptorPool_;
    vk::DescriptorSet projectionDescriptorSet_ = nullptr;
    vk::DescriptorSet fusedDescriptorSet_ = nullptr;
    vk::DescriptorSet optimizerDescriptorSet_ = nullptr;
    bool fusedEnabled_ = true;
    uint64_t fusedShaderHash_ = 0;
};

} // namespace vulkan3DGS
