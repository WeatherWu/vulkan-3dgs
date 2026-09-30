#include "training/core/backward/backward_validation_pass.hpp"

#include <array>

namespace vulkan3DGS {

void BackwardValidationPass::initialize(vk::Device device) {
    cleanup();
    device_ = device;
    ComputePipelineConfig gaussianConfig{};
    gaussianConfig.descriptorBindings = {
        backwardStorageBinding(0),
        backwardStorageBinding(19),
        backwardStorageBinding(30),
    };
    gaussianConfig.pushConstantSize = sizeof(TrainingPushConstants);
    ComputePipelineConfig finalizeConfig{};
    finalizeConfig.descriptorBindings = {
        backwardStorageBinding(29),
        backwardStorageBinding(30),
        backwardStorageBinding(31),
    };
    finalizeConfig.pushConstantSize = sizeof(TrainingPushConstants);
    gaussianValidationPipeline_.initialize(device_, "shaders/train_gaussian_validation.comp.spv",
                                           gaussianConfig);
    finalizePipeline_.initialize(device_, "shaders/train_validation_finalize.comp.spv",
                                 finalizeConfig);
    const std::vector<vk::DescriptorSet> sets =
        descriptorPool_.create(device_, 6u, 0u,
                               {gaussianValidationPipeline_.getDescriptorSetLayout(),
                                finalizePipeline_.getDescriptorSetLayout()});
    gaussianValidationDescriptorSet_ = sets[0];
    finalizeDescriptorSet_ = sets[1];
}

void BackwardValidationPass::cleanup() {
    descriptorPool_.reset();
    gaussianValidationDescriptorSet_ = nullptr;
    finalizeDescriptorSet_ = nullptr;
    finalizePipeline_.cleanup();
    gaussianValidationPipeline_.cleanup();
    device_ = nullptr;
}

void BackwardValidationPass::record(const BackwardPassContext& context) {
    if (context.pushConstants.validationEnabled == 0u) {
        return;
    }
    recordGaussianValidation(context);
    recordFinalize(context);
}

void BackwardValidationPass::recordGaussianValidation(const BackwardPassContext& context) {
    const bool densified = context.pushConstants.validationUsesDensifiedGaussians != 0u;
    const auto gaussianParamsInfo =
        densified ? context.buffers.densifiedParamsInfo() : context.buffers.gaussianParamsInfo();
    const auto countersInfo =
        densified ? context.buffers.densificationCountersInfo() : context.buffers.countersInfo();
    const auto partialsInfo = densified ? context.buffers.densifiedGaussianValidationPartialsInfo()
                                        : context.buffers.gaussianValidationPartialsInfo();
    std::array<vk::DescriptorBufferInfo, 3> infos = {gaussianParamsInfo, countersInfo,
                                                     partialsInfo};
    constexpr std::array<uint32_t, 3> bindings = {0u, 19u, 30u};
    std::array<vk::WriteDescriptorSet, 3> writes{};
    for (size_t index = 0; index < writes.size(); ++index) {
        writes[index]
            .setDstSet(gaussianValidationDescriptorSet_)
            .setDstBinding(bindings[index])
            .setDescriptorCount(1)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer)
            .setPBufferInfo(&infos[index]);
    }
    device_.updateDescriptorSets(writes, {});
    context.commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute,
                                       gaussianValidationPipeline_.getPipeline());
    context.commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                             gaussianValidationPipeline_.getPipelineLayout(), 0,
                                             gaussianValidationDescriptorSet_, {});
    context.commandBuffer.pushConstants(gaussianValidationPipeline_.getPipelineLayout(),
                                        vk::ShaderStageFlagBits::eCompute, 0,
                                        sizeof(TrainingPushConstants), &context.pushConstants);
    if (context.pushConstants.gaussianCount == 0u) return;
    context.commandBuffer.dispatch((context.pushConstants.gaussianCount + 255u) / 256u, 1u, 1u);
    recordBackwardBufferBarrier(context.commandBuffer, {partialsInfo},
                                vk::AccessFlagBits::eShaderRead);
}

void BackwardValidationPass::recordFinalize(const BackwardPassContext& context) {
    const auto pixelPartialsInfo = context.buffers.pixelValidationPartialsInfo();
    const auto gaussianPartialsInfo =
        context.pushConstants.validationUsesDensifiedGaussians != 0u
            ? context.buffers.densifiedGaussianValidationPartialsInfo()
            : context.buffers.gaussianValidationPartialsInfo();
    const auto resultInfo = context.buffers.validationFinalResultInfo();
    std::array<vk::DescriptorBufferInfo, 3> infos = {pixelPartialsInfo, gaussianPartialsInfo,
                                                     resultInfo};
    constexpr std::array<uint32_t, 3> bindings = {29u, 30u, 31u};
    std::array<vk::WriteDescriptorSet, 3> writes{};
    for (size_t index = 0; index < writes.size(); ++index) {
        writes[index]
            .setDstSet(finalizeDescriptorSet_)
            .setDstBinding(bindings[index])
            .setDescriptorCount(1)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer)
            .setPBufferInfo(&infos[index]);
    }
    device_.updateDescriptorSets(writes, {});
    context.commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute,
                                       finalizePipeline_.getPipeline());
    context.commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                             finalizePipeline_.getPipelineLayout(), 0,
                                             finalizeDescriptorSet_, {});
    context.commandBuffer.pushConstants(finalizePipeline_.getPipelineLayout(),
                                        vk::ShaderStageFlagBits::eCompute, 0,
                                        sizeof(TrainingPushConstants), &context.pushConstants);
    context.commandBuffer.dispatch(1u, 1u, 1u);
    recordBackwardBufferBarrier(context.commandBuffer, {resultInfo},
                                vk::AccessFlagBits::eTransferRead);
}

} // namespace vulkan3DGS
