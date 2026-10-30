#include "training/core/backward/vksplat_loss_pass.hpp"

#include "utils/file_utils.hpp"

#include <algorithm>
#include <stdexcept>

namespace vulkan3DGS {

VkSplatLossPass::VkSplatLossPass(vk::Device device, vk::PhysicalDevice physical, vk::Queue queue,
                                 uint32_t family)
    : device_(device), physical_(physical), queue_(queue), family_(family) {
    const std::array<const char*, 3> names = {"train_vksplat_ssim_forward",
                                              "train_vksplat_ssim_backward",
                                              "train_vksplat_loss_statistics"};
    const std::array<std::array<uint32_t, 4>, 3> bindings = {
        std::array<uint32_t, 4>{0, 1, 2, 4}, {0, 1, 2, 3}, {6, 8, 14, 29}};
    std::vector<vk::DescriptorSetLayout> layouts;
    for (uint32_t i = 0; i < 3; ++i) {
        ComputePipelineConfig config{};
        for (auto binding : bindings[i])
            config.descriptorBindings.push_back(backwardStorageBinding(binding));
        config.pushConstantSize = i == 2 ? sizeof(TrainingPushConstants) : 16u;
        pipelines_[i].initialize(
            device_, FileUtils::resolvePath(std::string("shaders/") + names[i] + ".comp.spv"),
            config);
        layouts.push_back(pipelines_[i].getDescriptorSetLayout());
    }
    sets_ = descriptors_.create(device_, 12u, 0u, layouts);
}

void VkSplatLossPass::recordForward(const BackwardPassContext& context) {
    const uint64_t bytes = uint64_t(context.pushConstants.pixelCount) * 48u;
    if (!physical_ || bytes > physical_.getProperties().limits.maxStorageBufferRange)
        throw std::runtime_error("VkSplat SSIM state exceeds storage buffer limits");
    if (state_.getSize() != bytes) {
        Buffer state;
        state.create(device_, physical_, queue_, family_, nullptr, bytes,
                     vk::BufferUsageFlagBits::eStorageBuffer |
                         vk::BufferUsageFlagBits::eTransferSrc,
                     vk::MemoryPropertyFlagBits::eDeviceLocal, {family_});
        state_ = std::move(state);
    }
    record(context, 0);
    recordBackwardBufferBarrier(
        context.commandBuffer, {state_.getDescriptorInfo(), context.buffers.lossInfo()},
        vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eTransferRead);
    if (context.pushConstants.validationEnabled != 0u) {
        record(context, 2);
        recordBackwardBufferBarrier(context.commandBuffer,
                                    {context.buffers.pixelValidationPartialsInfo()},
                                    vk::AccessFlagBits::eShaderRead);
    }
}

void VkSplatLossPass::recordBackward(const BackwardPassContext& context) {
    record(context, 1);
    recordBackwardBufferBarrier(context.commandBuffer, {context.buffers.pixelGradsInfo()},
                                vk::AccessFlagBits::eShaderRead);
}

void VkSplatLossPass::record(const BackwardPassContext& context, uint32_t stage) {
    const auto& buffers = context.buffers;
    const auto state = state_.getDescriptorInfo();
    std::array<vk::DescriptorBufferInfo, 4> infos;
    std::array<uint32_t, 4> bindings;
    if (stage == 0) {
        infos = {buffers.renderedColorInfo(), buffers.targetColorInfo(), state, buffers.lossInfo()};
        bindings = {0, 1, 2, 4};
    } else if (stage == 1) {
        infos = {buffers.renderedColorInfo(), buffers.targetColorInfo(), state,
                 buffers.pixelGradsInfo()};
        bindings = {0, 1, 2, 3};
    } else {
        infos = {buffers.renderedColorInfo(), buffers.lossInfo(), buffers.pixelBlendStatesInfo(),
                 buffers.pixelValidationPartialsInfo()};
        bindings = {6, 8, 14, 29};
    }
    std::array<vk::WriteDescriptorSet, 4> writes{};
    for (uint32_t i = 0; i < 4; ++i)
        writes[i]
            .setDstSet(sets_[stage])
            .setDstBinding(bindings[i])
            .setDescriptorCount(1)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer)
            .setPBufferInfo(&infos[i]);
    device_.updateDescriptorSets(writes, {});
    const auto layout = pipelines_[stage].getPipelineLayout();
    context.commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute,
                                       pipelines_[stage].getPipeline());
    context.commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, layout, 0,
                                             sets_[stage], {});
    if (stage == 2) {
        context.commandBuffer.pushConstants(layout, vk::ShaderStageFlagBits::eCompute, 0,
                                            sizeof(TrainingPushConstants), &context.pushConstants);
    } else {
        struct Uniforms {
            uint32_t width, height;
            float l1, ssim;
        };
        const auto& p = context.pushConstants;
        const float lambda = std::clamp(p.lossDssimWeight, 0.0f, 1.0f);
        const Uniforms uniforms{
            p.width, p.height,
            (1.0f - lambda) / (3.0f * static_cast<float>(p.width) * static_cast<float>(p.height)),
            -lambda /
                (3.0f * static_cast<float>(p.width - 10u) * static_cast<float>(p.height - 10u))};
        context.commandBuffer.pushConstants(layout, vk::ShaderStageFlagBits::eCompute, 0,
                                            sizeof(uniforms), &uniforms);
    }
    context.commandBuffer.dispatch((context.pushConstants.width + 15u) / 16u,
                                   (context.pushConstants.height + 15u) / 16u, 1u);
}

} // namespace vulkan3DGS
