#include "training/core/backward/projection_optimizer_pass.hpp"

#include "utils/logger.hpp"

#include <array>
#include <cstdlib>
#include <cstring>

namespace vulkan3DGS {

namespace {

bool forceSeparateProjectionOptimizer() {
#ifdef _WIN32
    char* value = nullptr;
    size_t length = 0;
    if (_dupenv_s(&value, &length, "VULKAN_3DGS_SEPARATE_PROJECTION_OPTIMIZER") != 0 || !value) {
        return false;
    }
    const bool enabled = std::strcmp(value, "1") == 0;
    std::free(value);
    return enabled;
#else
    const char* value = std::getenv("VULKAN_3DGS_SEPARATE_PROJECTION_OPTIMIZER");
    return value && std::strcmp(value, "1") == 0;
#endif
}

} // namespace

void ProjectionOptimizerPass::initialize(vk::Device device) {
    cleanup();
    device_ = device;
    fusedEnabled_ = !forceSeparateProjectionOptimizer();

    ComputePipelineConfig projectionConfig{};
    projectionConfig.descriptorBindings = {
        backwardStorageBinding(0),  backwardStorageBinding(1),  backwardStorageBinding(3),
        backwardStorageBinding(13), backwardStorageBinding(16), backwardUniformBinding(11),
    };
    projectionConfig.pushConstantSize = sizeof(TrainingPushConstants);
    ComputePipelineConfig fusedConfig{};
    fusedConfig.descriptorBindings = {
        backwardStorageBinding(0),  backwardStorageBinding(2),  backwardStorageBinding(3),
        backwardStorageBinding(13), backwardStorageBinding(16), backwardUniformBinding(11),
    };
    fusedConfig.pushConstantSize = sizeof(TrainingPushConstants);
    ComputePipelineConfig optimizerConfig{};
    optimizerConfig.descriptorBindings = {
        backwardStorageBinding(0),
        backwardStorageBinding(1),
        backwardStorageBinding(2),
    };
    optimizerConfig.pushConstantSize = sizeof(TrainingPushConstants);

    projectionPipeline_.initialize(device_, "shaders/train_backward_2dgs_to_3dgs.comp.spv",
                                   projectionConfig);
    fusedPipeline_.initialize(device_, "shaders/train_backward_2dgs_to_3dgs_optimizer.comp.spv",
                              fusedConfig);
    optimizerPipeline_.initialize(device_, "shaders/train_optimizer.comp.spv", optimizerConfig);
    const std::vector<vk::DescriptorSet> sets = descriptorPool_.create(
        device_, 13u, 2u,
        {projectionPipeline_.getDescriptorSetLayout(), fusedPipeline_.getDescriptorSetLayout(),
         optimizerPipeline_.getDescriptorSetLayout()});
    projectionDescriptorSet_ = sets[0];
    fusedDescriptorSet_ = sets[1];
    optimizerDescriptorSet_ = sets[2];

    if (fusedEnabled_) {
        LOG_INFO("Fused projection backward and optimizer enabled");
    } else {
        LOG_WARN("Using separate projection backward and optimizer because "
                 "VULKAN_3DGS_SEPARATE_PROJECTION_OPTIMIZER=1");
    }
}

void ProjectionOptimizerPass::cleanup() {
    descriptorPool_.reset();
    projectionDescriptorSet_ = nullptr;
    fusedDescriptorSet_ = nullptr;
    optimizerDescriptorSet_ = nullptr;
    optimizerPipeline_.cleanup();
    fusedPipeline_.cleanup();
    projectionPipeline_.cleanup();
    fusedEnabled_ = true;
    device_ = nullptr;
}

void ProjectionOptimizerPass::recordProjection(const BackwardPassContext& context) {
    const auto gaussianParamsInfo = context.buffers.gaussianParamsInfo();
    const auto gaussianGradsInfo = context.buffers.gaussianGradsInfo();
    const auto projectedInfo = context.buffers.projectedInfo();
    const auto projectedGradsInfo = context.buffers.projectedGradsInfo();
    const auto densificationStatesInfo = context.buffers.densificationStatesInfo();
    const auto cameraInfo = context.buffers.cameraInfo();
    std::array<vk::DescriptorBufferInfo, 6> infos = {gaussianParamsInfo,      gaussianGradsInfo,
                                                     projectedInfo,           projectedGradsInfo,
                                                     densificationStatesInfo, cameraInfo};
    constexpr std::array<uint32_t, 6> bindings = {0u, 1u, 3u, 13u, 16u, 11u};
    constexpr std::array<vk::DescriptorType, 6> types = {
        vk::DescriptorType::eStorageBuffer, vk::DescriptorType::eStorageBuffer,
        vk::DescriptorType::eStorageBuffer, vk::DescriptorType::eStorageBuffer,
        vk::DescriptorType::eStorageBuffer, vk::DescriptorType::eUniformBuffer};
    std::array<vk::WriteDescriptorSet, 6> writes{};
    for (size_t index = 0; index < writes.size(); ++index) {
        writes[index]
            .setDstSet(projectionDescriptorSet_)
            .setDstBinding(bindings[index])
            .setDescriptorCount(1)
            .setDescriptorType(types[index])
            .setPBufferInfo(&infos[index]);
    }
    device_.updateDescriptorSets(writes, {});
    context.commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute,
                                       projectionPipeline_.getPipeline());
    context.commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                             projectionPipeline_.getPipelineLayout(), 0,
                                             projectionDescriptorSet_, {});
    context.commandBuffer.pushConstants(projectionPipeline_.getPipelineLayout(),
                                        vk::ShaderStageFlagBits::eCompute, 0,
                                        sizeof(TrainingPushConstants), &context.pushConstants);
    if (context.pushConstants.gaussianCount == 0u) return;
    context.commandBuffer.dispatch((context.pushConstants.gaussianCount + 255u) / 256u, 1u, 1u);
    recordBackwardBufferBarrier(context.commandBuffer, {gaussianGradsInfo, densificationStatesInfo},
                                vk::AccessFlagBits::eShaderRead |
                                    vk::AccessFlagBits::eTransferRead);
}

void ProjectionOptimizerPass::recordProjectionAndOptimize(const BackwardPassContext& context) {
    const auto gaussianParamsInfo = context.buffers.gaussianParamsInfo();
    const auto adamStatesInfo = context.buffers.adamStatesInfo();
    const auto projectedInfo = context.buffers.projectedInfo();
    const auto projectedGradsInfo = context.buffers.projectedGradsInfo();
    const auto densificationStatesInfo = context.buffers.densificationStatesInfo();
    const auto cameraInfo = context.buffers.cameraInfo();
    std::array<vk::DescriptorBufferInfo, 6> infos = {gaussianParamsInfo,      adamStatesInfo,
                                                     projectedInfo,           projectedGradsInfo,
                                                     densificationStatesInfo, cameraInfo};
    constexpr std::array<uint32_t, 6> bindings = {0u, 2u, 3u, 13u, 16u, 11u};
    constexpr std::array<vk::DescriptorType, 6> types = {
        vk::DescriptorType::eStorageBuffer, vk::DescriptorType::eStorageBuffer,
        vk::DescriptorType::eStorageBuffer, vk::DescriptorType::eStorageBuffer,
        vk::DescriptorType::eStorageBuffer, vk::DescriptorType::eUniformBuffer};
    std::array<vk::WriteDescriptorSet, 6> writes{};
    for (size_t index = 0; index < writes.size(); ++index) {
        writes[index]
            .setDstSet(fusedDescriptorSet_)
            .setDstBinding(bindings[index])
            .setDescriptorCount(1)
            .setDescriptorType(types[index])
            .setPBufferInfo(&infos[index]);
    }
    device_.updateDescriptorSets(writes, {});
    context.commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute,
                                       fusedPipeline_.getPipeline());
    context.commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                             fusedPipeline_.getPipelineLayout(), 0,
                                             fusedDescriptorSet_, {});
    context.commandBuffer.pushConstants(fusedPipeline_.getPipelineLayout(),
                                        vk::ShaderStageFlagBits::eCompute, 0,
                                        sizeof(TrainingPushConstants), &context.pushConstants);
    if (context.pushConstants.gaussianCount == 0u) return;
    context.commandBuffer.dispatch((context.pushConstants.gaussianCount + 255u) / 256u, 1u, 1u);
    recordBackwardBufferBarrier(context.commandBuffer,
                                {gaussianParamsInfo, adamStatesInfo, densificationStatesInfo},
                                vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite |
                                    vk::AccessFlagBits::eTransferRead);
}

void ProjectionOptimizerPass::recordOptimizer(const BackwardPassContext& context) {
    const auto gaussianParamsInfo = context.buffers.gaussianParamsInfo();
    const auto gaussianGradsInfo = context.buffers.gaussianGradsInfo();
    const auto adamStatesInfo = context.buffers.adamStatesInfo();
    std::array<vk::DescriptorBufferInfo, 3> infos = {gaussianParamsInfo, gaussianGradsInfo,
                                                     adamStatesInfo};
    constexpr std::array<uint32_t, 3> bindings = {0u, 1u, 2u};
    std::array<vk::WriteDescriptorSet, 3> writes{};
    for (size_t index = 0; index < writes.size(); ++index) {
        writes[index]
            .setDstSet(optimizerDescriptorSet_)
            .setDstBinding(bindings[index])
            .setDescriptorCount(1)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer)
            .setPBufferInfo(&infos[index]);
    }
    device_.updateDescriptorSets(writes, {});
    context.commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute,
                                       optimizerPipeline_.getPipeline());
    context.commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                             optimizerPipeline_.getPipelineLayout(), 0,
                                             optimizerDescriptorSet_, {});
    context.commandBuffer.pushConstants(optimizerPipeline_.getPipelineLayout(),
                                        vk::ShaderStageFlagBits::eCompute, 0,
                                        sizeof(TrainingPushConstants), &context.pushConstants);
    if (context.pushConstants.gaussianCount == 0u) return;
    context.commandBuffer.dispatch((context.pushConstants.gaussianCount + 255u) / 256u, 1u, 1u);
    recordBackwardBufferBarrier(context.commandBuffer, {gaussianParamsInfo, adamStatesInfo},
                                vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite |
                                    vk::AccessFlagBits::eTransferRead);
}

} // namespace vulkan3DGS
