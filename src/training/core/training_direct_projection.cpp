#include "training/core/training_direct_projection.hpp"
#include "utils/file_utils.hpp"
#include <stdexcept>

namespace vulkan3DGS {
TrainingDirectProjection::TrainingDirectProjection(vk::Device device, bool forward,
                                                   const std::filesystem::path& directory)
    : device_(device), forward_(forward), pipelines_(forward ? 1 : 2) {
    std::vector<vk::DescriptorSetLayout> layouts;
    for (size_t i = 0; i < pipelines_.size(); ++i) {
        ComputePipelineConfig config;
        config.pushConstantSize = sizeof(TrainingPushConstants);
        const std::vector<uint32_t> bindings =
            forward ? std::vector<uint32_t>{0, 3, 11, 64}
                    : std::vector<uint32_t>{0, i ? 2u : 1u, 3, 13, 16, 32, 11, 64};
        for (auto binding : bindings)
            config.descriptorBindings.push_back(binding == 11 || binding == 64
                                                    ? backwardUniformBinding(binding)
                                                    : backwardStorageBinding(binding));
        const char* name = forward ? "train_forward_project_direct_intrinsics.comp.spv"
                           : i ? "train_backward_2dgs_to_3dgs_optimizer_direct_intrinsics.comp.spv"
                               : "train_backward_2dgs_to_3dgs_direct_intrinsics.comp.spv";
        pipelines_[i].initialize(device_, FileUtils::resolvePath((directory / name).string()),
                                 config);
        layouts.push_back(pipelines_[i].getDescriptorSetLayout());
    }
    sets_ = descriptors_.create(device_, forward ? 2 : 12, forward ? 2 : 4, layouts);
}
void TrainingDirectProjection::bind(vk::CommandBuffer command, size_t index,
                                    const TrainingBuffers& buffers,
                                    const TrainingPushConstants& push, bool fused) {
    if (!buffers.hasExactIntrinsics())
        throw std::runtime_error("Direct projection requires original camera intrinsics");
    const std::vector<uint32_t> bindings =
        forward_ ? std::vector<uint32_t>{0, 3, 11, 64}
                 : std::vector<uint32_t>{0, fused ? 2u : 1u, 3, 13, 16, 32, 11, 64};
    const std::vector<vk::DescriptorBufferInfo> infos =
        forward_
            ? std::vector<vk::DescriptorBufferInfo>{buffers.gaussianParamsInfo(),
                                                    buffers.projectedInfo(), buffers.cameraInfo(),
                                                    buffers.cameraIntrinsicsInfo()}
            : std::vector<vk::DescriptorBufferInfo>{buffers.gaussianParamsInfo(),
                                                    fused ? buffers.adamStatesInfo()
                                                          : buffers.gaussianGradsInfo(),
                                                    buffers.projectedInfo(),
                                                    buffers.projectedGradsInfo(),
                                                    buffers.densificationStatesInfo(),
                                                    buffers.gaussianTileRangesInfo(),
                                                    buffers.cameraInfo(),
                                                    buffers.cameraIntrinsicsInfo()};
    std::vector<vk::WriteDescriptorSet> writes(bindings.size());
    for (size_t i = 0; i < writes.size(); ++i)
        writes[i]
            .setDstSet(sets_[index])
            .setDstBinding(bindings[i])
            .setDescriptorCount(1)
            .setDescriptorType(bindings[i] == 11 || bindings[i] == 64
                                   ? vk::DescriptorType::eUniformBuffer
                                   : vk::DescriptorType::eStorageBuffer)
            .setPBufferInfo(&infos[i]);
    device_.updateDescriptorSets(writes, {});
    auto& pipeline = pipelines_[index];
    command.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.getPipeline());
    command.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipeline.getPipelineLayout(), 0,
                               sets_[index], {});
    command.pushConstants(pipeline.getPipelineLayout(), vk::ShaderStageFlagBits::eCompute, 0,
                          sizeof(push), &push);
    if (push.gaussianCount) command.dispatch((push.gaussianCount + 255u) / 256u, 1, 1);
}
void TrainingDirectProjection::recordForward(vk::CommandBuffer command,
                                             const TrainingBuffers& buffers,
                                             const TrainingPushConstants& push) {
    bind(command, 0, buffers, push, false);
}
void TrainingDirectProjection::recordBackward(const BackwardPassContext& context, bool fused) {
    if (context.optimizerCapture)
        throw std::runtime_error("Direct projection cannot write legacy optimizer captures");
    bind(context.commandBuffer, fused ? 1 : 0, context.buffers, context.pushConstants, fused);
    if (fused)
        recordBackwardBufferBarrier(
            context.commandBuffer,
            {context.buffers.gaussianParamsInfo(), context.buffers.adamStatesInfo(),
             context.buffers.densificationStatesInfo()},
            vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite |
                vk::AccessFlagBits::eTransferRead);
    else
        recordBackwardBufferBarrier(
            context.commandBuffer,
            {context.buffers.gaussianGradsInfo(), context.buffers.densificationStatesInfo()},
            vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eTransferRead);
}
} // namespace vulkan3DGS
