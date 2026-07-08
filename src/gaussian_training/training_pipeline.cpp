#include "gaussian_training/training_pipeline.hpp"

namespace vulkan3DGS {

namespace {

vk::DescriptorSetLayoutBinding storageBinding(uint32_t binding) {
    vk::DescriptorSetLayoutBinding layoutBinding{};
    layoutBinding.setBinding(binding)
                 .setDescriptorType(vk::DescriptorType::eStorageBuffer)
                 .setDescriptorCount(1)
                 .setStageFlags(vk::ShaderStageFlagBits::eCompute);
    return layoutBinding;
}

} // namespace

void TrainingPipelines::initialize(vk::Device device) {
    const ComputePipelineConfig config = createCommonConfig();

    clear_.initialize(device, "shaders/train_clear.comp.spv", config);
    project_.initialize(device, "shaders/train_project.comp.spv", config);
    forward_.initialize(device, "shaders/train_forward.comp.spv", config);
    loss_.initialize(device, "shaders/train_loss.comp.spv", config);
    packRenderBuffer_.initialize(device, "shaders/train_pack_render_buffer.comp.spv", config);
}

void TrainingPipelines::cleanup() {
    packRenderBuffer_.cleanup();
    loss_.cleanup();
    forward_.cleanup();
    project_.cleanup();
    clear_.cleanup();
}

ComputePipelineConfig TrainingPipelines::createCommonConfig() {
    ComputePipelineConfig config{};
    config.descriptorBindings = {
        storageBinding(0),  // GaussianTrainParam
        storageBinding(1),  // GaussianGrad
        storageBinding(2),  // AdamState
        storageBinding(3),  // ProjectedGaussian
        storageBinding(4),  // tile items
        storageBinding(5),  // tile ranges
        storageBinding(6),  // rendered color
        storageBinding(7),  // target color
        storageBinding(8),  // loss
        storageBinding(9),  // counters
        storageBinding(10), // preview instances
    };
    config.pushConstantSize = sizeof(TrainingPushConstants);
    return config;
}

} // namespace vulkan3DGS
