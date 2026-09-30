#include "training/core/backward/backward_loss_pass.hpp"

#include "utils/logger.hpp"

#include <array>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace vulkan3DGS {

namespace {

bool forceComputeBackwardClear() {
#ifdef _WIN32
    char* value = nullptr;
    size_t length = 0;
    if (_dupenv_s(&value, &length, "VULKAN_3DGS_COMPUTE_BACKWARD_CLEAR") != 0 || !value) {
        return false;
    }
    const bool enabled = std::strcmp(value, "1") == 0;
    std::free(value);
    return enabled;
#else
    const char* value = std::getenv("VULKAN_3DGS_COMPUTE_BACKWARD_CLEAR");
    return value && std::strcmp(value, "1") == 0;
#endif
}

} // namespace

void BackwardLossPass::initialize(vk::Device device) {
    cleanup();
    device_ = device;
    computeClear_ = forceComputeBackwardClear();

    ComputePipelineConfig lossConfig{};
    lossConfig.descriptorBindings = {
        backwardStorageBinding(6),  backwardStorageBinding(7),  backwardStorageBinding(8),
        backwardStorageBinding(14), backwardStorageBinding(28), backwardStorageBinding(29),
    };
    lossConfig.pushConstantSize = sizeof(TrainingPushConstants);

    ComputePipelineConfig clearConfig{};
    clearConfig.descriptorBindings = {
        backwardStorageBinding(1),
        backwardStorageBinding(13),
    };
    clearConfig.pushConstantSize = sizeof(TrainingPushConstants);

    ComputePipelineConfig lossToPixelConfig{};
    lossToPixelConfig.descriptorBindings = {
        backwardStorageBinding(6),
        backwardStorageBinding(7),
        backwardStorageBinding(12),
        backwardStorageBinding(28),
    };
    lossToPixelConfig.pushConstantSize = sizeof(TrainingPushConstants);

    clearPipeline_.initialize(device_, "shaders/train_backward_clear.comp.spv", clearConfig);
    lossPipeline_.initialize(device_, "shaders/train_loss.comp.spv", lossConfig);
    lossToPixelPipeline_.initialize(device_, "shaders/train_backward_loss_to_pixel.comp.spv",
                                    lossToPixelConfig);

    const std::vector<vk::DescriptorSet> sets = descriptorPool_.create(
        device_, 12u, 0u,
        {clearPipeline_.getDescriptorSetLayout(), lossPipeline_.getDescriptorSetLayout(),
         lossToPixelPipeline_.getDescriptorSetLayout()});
    clearDescriptorSet_ = sets[0];
    lossDescriptorSet_ = sets[1];
    lossToPixelDescriptorSet_ = sets[2];

    if (computeClear_) {
        LOG_WARN("Using compute backward clear because VULKAN_3DGS_COMPUTE_BACKWARD_CLEAR=1");
    }
}

void BackwardLossPass::cleanup() {
    descriptorPool_.reset();
    clearDescriptorSet_ = nullptr;
    lossDescriptorSet_ = nullptr;
    lossToPixelDescriptorSet_ = nullptr;
    lossToPixelPipeline_.cleanup();
    lossPipeline_.cleanup();
    clearPipeline_.cleanup();
    computeClear_ = false;
    device_ = nullptr;
}

void BackwardLossPass::recordLoss(const BackwardPassContext& context) {
    if (!lossDescriptorSet_) {
        throw std::runtime_error("Loss descriptor set is not initialized");
    }
    const auto renderedColorInfo = context.buffers.renderedColorInfo();
    const auto targetColorInfo = context.buffers.targetColorInfo();
    const auto lossInfo = context.buffers.lossInfo();
    const auto pixelBlendStatesInfo = context.buffers.pixelBlendStatesInfo();
    const auto ssimBackwardStatesInfo = context.buffers.ssimBackwardStatesInfo();
    const auto pixelValidationPartialsInfo = context.buffers.pixelValidationPartialsInfo();

    std::array<vk::WriteDescriptorSet, 6> writes{};
    const std::array<uint32_t, 6> bindings = {6u, 7u, 8u, 14u, 28u, 29u};
    const std::array<const vk::DescriptorBufferInfo*, 6> infos = {
        &renderedColorInfo,    &targetColorInfo,        &lossInfo,
        &pixelBlendStatesInfo, &ssimBackwardStatesInfo, &pixelValidationPartialsInfo,
    };
    for (size_t index = 0; index < writes.size(); ++index) {
        writes[index]
            .setDstSet(lossDescriptorSet_)
            .setDstBinding(bindings[index])
            .setDescriptorCount(1)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer)
            .setPBufferInfo(infos[index]);
    }
    device_.updateDescriptorSets(writes, {});

    context.commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute,
                                       lossPipeline_.getPipeline());
    context.commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                             lossPipeline_.getPipelineLayout(), 0,
                                             lossDescriptorSet_, {});
    context.commandBuffer.pushConstants(lossPipeline_.getPipelineLayout(),
                                        vk::ShaderStageFlagBits::eCompute, 0,
                                        sizeof(TrainingPushConstants), &context.pushConstants);
    if (context.pushConstants.pixelCount == 0u) {
        return;
    }
    context.commandBuffer.dispatch((context.pushConstants.width + 15u) / 16u,
                                   (context.pushConstants.height + 15u) / 16u, 1u);

    vk::BufferMemoryBarrier lossReady{};
    lossReady.setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
        .setDstAccessMask(vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eTransferRead)
        .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setBuffer(lossInfo.buffer)
        .setOffset(lossInfo.offset)
        .setSize(lossInfo.range);
    context.commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                          vk::PipelineStageFlagBits::eComputeShader |
                                              vk::PipelineStageFlagBits::eTransfer,
                                          {}, {}, lossReady, {});
    recordBackwardBufferBarrier(context.commandBuffer, {ssimBackwardStatesInfo},
                                vk::AccessFlagBits::eShaderRead);
    if (context.pushConstants.validationEnabled != 0u) {
        recordBackwardBufferBarrier(context.commandBuffer, {pixelValidationPartialsInfo},
                                    vk::AccessFlagBits::eShaderRead);
    }
}

void BackwardLossPass::recordClear(const BackwardPassContext& context,
                                   bool fusedProjectionOptimizerEnabled) {
    const auto gaussianGradsInfo = context.buffers.gaussianGradsInfo();
    const auto projectedGradsInfo = context.buffers.projectedGradsInfo();
    const bool clearGaussianGrads =
        context.pushConstants.optimizerEnabled == 0u || !fusedProjectionOptimizerEnabled;
    if (context.pushConstants.gaussianCount == 0u) {
        return;
    }

    if (!computeClear_) {
        const vk::DeviceSize gaussianGradsSize =
            sizeof(GaussianGrad) * static_cast<vk::DeviceSize>(context.pushConstants.gaussianCount);
        const vk::DeviceSize projectedGradsSize =
            sizeof(ProjectedGaussianGrad) *
            static_cast<vk::DeviceSize>(context.pushConstants.gaussianCount);
        context.commandBuffer.fillBuffer(projectedGradsInfo.buffer, projectedGradsInfo.offset,
                                         projectedGradsSize, 0u);
        if (clearGaussianGrads) {
            context.commandBuffer.fillBuffer(gaussianGradsInfo.buffer, gaussianGradsInfo.offset,
                                             gaussianGradsSize, 0u);
        }
        std::array<vk::BufferMemoryBarrier, 2> barriers{};
        barriers[0]
            .setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
            .setDstAccessMask(vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite)
            .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setBuffer(projectedGradsInfo.buffer)
            .setOffset(projectedGradsInfo.offset)
            .setSize(projectedGradsSize);
        barriers[1]
            .setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
            .setDstAccessMask(vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite)
            .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setBuffer(gaussianGradsInfo.buffer)
            .setOffset(gaussianGradsInfo.offset)
            .setSize(gaussianGradsSize);
        context.commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                              vk::PipelineStageFlagBits::eComputeShader, {}, {},
                                              vk::ArrayProxy<const vk::BufferMemoryBarrier>(
                                                  clearGaussianGrads ? 2u : 1u, barriers.data()),
                                              {});
        return;
    }

    std::array<vk::DescriptorBufferInfo, 2> infos = {gaussianGradsInfo, projectedGradsInfo};
    std::array<vk::WriteDescriptorSet, 2> writes{};
    constexpr std::array<uint32_t, 2> bindings = {1u, 13u};
    for (size_t index = 0; index < writes.size(); ++index) {
        writes[index]
            .setDstSet(clearDescriptorSet_)
            .setDstBinding(bindings[index])
            .setDescriptorCount(1)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer)
            .setPBufferInfo(&infos[index]);
    }
    device_.updateDescriptorSets(writes, {});
    context.commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute,
                                       clearPipeline_.getPipeline());
    context.commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                             clearPipeline_.getPipelineLayout(), 0,
                                             clearDescriptorSet_, {});
    TrainingPushConstants clearPushConstants = context.pushConstants;
    clearPushConstants.clearGaussianGrads = clearGaussianGrads ? 1u : 0u;
    context.commandBuffer.pushConstants(clearPipeline_.getPipelineLayout(),
                                        vk::ShaderStageFlagBits::eCompute, 0,
                                        sizeof(TrainingPushConstants), &clearPushConstants);
    context.commandBuffer.dispatch((context.pushConstants.gaussianCount + 255u) / 256u, 1u, 1u);
    if (clearGaussianGrads) {
        recordBackwardBufferBarrier(context.commandBuffer, {gaussianGradsInfo, projectedGradsInfo},
                                    vk::AccessFlagBits::eShaderRead |
                                        vk::AccessFlagBits::eShaderWrite);
    } else {
        recordBackwardBufferBarrier(context.commandBuffer, {projectedGradsInfo},
                                    vk::AccessFlagBits::eShaderRead |
                                        vk::AccessFlagBits::eShaderWrite);
    }
}

void BackwardLossPass::recordLossToPixel(const BackwardPassContext& context) {
    const auto renderedColorInfo = context.buffers.renderedColorInfo();
    const auto targetColorInfo = context.buffers.targetColorInfo();
    const auto pixelGradsInfo = context.buffers.pixelGradsInfo();
    const auto ssimBackwardStatesInfo = context.buffers.ssimBackwardStatesInfo();
    std::array<vk::DescriptorBufferInfo, 4> infos = {renderedColorInfo, targetColorInfo,
                                                     pixelGradsInfo, ssimBackwardStatesInfo};
    constexpr std::array<uint32_t, 4> bindings = {6u, 7u, 12u, 28u};
    std::array<vk::WriteDescriptorSet, 4> writes{};
    for (size_t index = 0; index < writes.size(); ++index) {
        writes[index]
            .setDstSet(lossToPixelDescriptorSet_)
            .setDstBinding(bindings[index])
            .setDescriptorCount(1)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer)
            .setPBufferInfo(&infos[index]);
    }
    device_.updateDescriptorSets(writes, {});
    context.commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute,
                                       lossToPixelPipeline_.getPipeline());
    context.commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                             lossToPixelPipeline_.getPipelineLayout(), 0,
                                             lossToPixelDescriptorSet_, {});
    context.commandBuffer.pushConstants(lossToPixelPipeline_.getPipelineLayout(),
                                        vk::ShaderStageFlagBits::eCompute, 0,
                                        sizeof(TrainingPushConstants), &context.pushConstants);
    if (context.pushConstants.pixelCount == 0u) {
        return;
    }
    context.commandBuffer.dispatch((context.pushConstants.width + 15u) / 16u,
                                   (context.pushConstants.height + 15u) / 16u, 1u);
    recordBackwardBufferBarrier(context.commandBuffer, {pixelGradsInfo},
                                vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
}

} // namespace vulkan3DGS
