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

    ComputePipelineConfig momentsConfig{};
    momentsConfig.descriptorBindings = {backwardStorageBinding(6), backwardStorageBinding(7),
                                        backwardStorageBinding(42)};
    momentsConfig.pushConstantSize = sizeof(TrainingPushConstants);
    ComputePipelineConfig coefficientsConfig{};
    coefficientsConfig.descriptorBindings = {backwardStorageBinding(28),
                                             backwardStorageBinding(42)};
    coefficientsConfig.pushConstantSize = sizeof(TrainingPushConstants);
    lossConfig.descriptorBindings.push_back(backwardStorageBinding(42));
    lossToPixelConfig.descriptorBindings.push_back(backwardStorageBinding(42));
    ssimMomentsHorizontalPipeline_.initialize(
        device_, "shaders/train_ssim_moments_horizontal.comp.spv", momentsConfig);
    ssimCoefficientsHorizontalPipeline_.initialize(
        device_, "shaders/train_ssim_coefficients_horizontal.comp.spv", coefficientsConfig);
    separableLossPipeline_.initialize(device_, "shaders/train_loss_separable.comp.spv", lossConfig);
    separableLossToPixelPipeline_.initialize(
        device_, "shaders/train_backward_loss_to_pixel_separable.comp.spv", lossToPixelConfig);

    const std::vector<vk::DescriptorSet> sets = descriptorPool_.create(
        device_, 29u, 0u,
        {clearPipeline_.getDescriptorSetLayout(), lossPipeline_.getDescriptorSetLayout(),
         lossToPixelPipeline_.getDescriptorSetLayout(),
         ssimMomentsHorizontalPipeline_.getDescriptorSetLayout(),
         ssimCoefficientsHorizontalPipeline_.getDescriptorSetLayout(),
         separableLossPipeline_.getDescriptorSetLayout(),
         separableLossToPixelPipeline_.getDescriptorSetLayout()});
    clearDescriptorSet_ = sets[0];
    lossDescriptorSet_ = sets[1];
    lossToPixelDescriptorSet_ = sets[2];
    ssimMomentsHorizontalSet_ = sets[3];
    ssimCoefficientsHorizontalSet_ = sets[4];
    separableLossSet_ = sets[5];
    separableLossToPixelSet_ = sets[6];

    if (computeClear_) {
        LOG_WARN("Using compute backward clear because VULKAN_3DGS_COMPUTE_BACKWARD_CLEAR=1");
    }
}

void BackwardLossPass::cleanup() {
    vkSplat_.reset();
    vkSplatActive_ = false;
    descriptorPool_.reset();
    clearDescriptorSet_ = nullptr;
    lossDescriptorSet_ = nullptr;
    lossToPixelDescriptorSet_ = nullptr;
    ssimMomentsHorizontalSet_ = nullptr;
    ssimCoefficientsHorizontalSet_ = nullptr;
    separableLossSet_ = nullptr;
    separableLossToPixelSet_ = nullptr;
    ssimMomentsHorizontalPipeline_.cleanup();
    ssimCoefficientsHorizontalPipeline_.cleanup();
    separableLossPipeline_.cleanup();
    separableLossToPixelPipeline_.cleanup();
    ssimMode_ = TrainingSsimMode::TwoDimensional;
    lossToPixelPipeline_.cleanup();
    lossPipeline_.cleanup();
    clearPipeline_.cleanup();
    computeClear_ = false;
    device_ = nullptr;
}

void BackwardLossPass::recordLoss(const BackwardPassContext& context) {
    vkSplatActive_ = vkSplatRequested_ && context.pushConstants.width > 10u &&
                     context.pushConstants.height > 10u;
    if (vkSplatActive_) {
        if (!vkSplat_)
            vkSplat_ = std::make_unique<VkSplatLossPass>(device_, physical_, queue_, family_);
        vkSplat_->recordForward(context);
        return;
    }
    const bool separable = ssimUsesSeparableLoss(ssimMode_);
    if (context.pushConstants.pixelCount == 0u) return;
    if (!lossDescriptorSet_) {
        throw std::runtime_error("Loss descriptor set is not initialized");
    }
    const auto renderedColorInfo = context.buffers.renderedColorInfo();
    const auto targetColorInfo = context.buffers.targetColorInfo();
    const auto lossInfo = context.buffers.lossInfo();
    const auto pixelBlendStatesInfo = context.buffers.pixelBlendStatesInfo();
    const auto ssimBackwardStatesInfo = context.buffers.ssimBackwardStatesInfo();
    const auto pixelValidationPartialsInfo = context.buffers.pixelValidationPartialsInfo();
    const auto scratchInfo = context.buffers.ssimScratchInfo();
    if (separable) recordSsimHorizontal(context, false);
    const auto& pipeline = separable ? separableLossPipeline_ : lossPipeline_;
    const auto descriptorSet = separable ? separableLossSet_ : lossDescriptorSet_;

    std::array<vk::WriteDescriptorSet, 7> writes{};
    const uint32_t writeCount = separable ? 7u : 6u;
    const std::array<uint32_t, 7> bindings = {6u, 7u, 8u, 14u, 28u, 29u, 42u};
    const std::array<const vk::DescriptorBufferInfo*, 7> infos = {
        &renderedColorInfo,    &targetColorInfo,        &lossInfo,
        &pixelBlendStatesInfo, &ssimBackwardStatesInfo, &pixelValidationPartialsInfo,
        &scratchInfo,
    };
    for (size_t index = 0; index < writeCount; ++index) {
        writes[index]
            .setDstSet(descriptorSet)
            .setDstBinding(bindings[index])
            .setDescriptorCount(1)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer)
            .setPBufferInfo(infos[index]);
    }
    device_.updateDescriptorSets(
        vk::ArrayProxy<const vk::WriteDescriptorSet>(writeCount, writes.data()), {});

    context.commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.getPipeline());
    context.commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                             pipeline.getPipelineLayout(), 0, descriptorSet, {});
    context.commandBuffer.pushConstants(pipeline.getPipelineLayout(),
                                        vk::ShaderStageFlagBits::eCompute, 0,
                                        sizeof(TrainingPushConstants), &context.pushConstants);
    if (context.pushConstants.pixelCount == 0u) {
        return;
    }
    context.commandBuffer.dispatch((context.pushConstants.width + 15u) / 16u,
                                   (context.pushConstants.height + 15u) / 16u, 1u);
    if (separable) recordScratchReadComplete(context);

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
    if (vkSplatActive_) {
        vkSplat_->recordBackward(context);
        return;
    }
    const bool separable = ssimUsesSeparableBackward(ssimMode_);
    if (context.pushConstants.pixelCount == 0u) return;
    const auto renderedColorInfo = context.buffers.renderedColorInfo();
    const auto targetColorInfo = context.buffers.targetColorInfo();
    const auto pixelGradsInfo = context.buffers.pixelGradsInfo();
    const auto ssimBackwardStatesInfo = context.buffers.ssimBackwardStatesInfo();
    if (separable) recordSsimHorizontal(context, true);
    const auto& pipeline = separable ? separableLossToPixelPipeline_ : lossToPixelPipeline_;
    const auto descriptorSet =
        separable ? separableLossToPixelSet_ : lossToPixelDescriptorSet_;
    std::array<vk::DescriptorBufferInfo, 5> infos = {renderedColorInfo, targetColorInfo,
                                                     pixelGradsInfo, ssimBackwardStatesInfo,
                                                     context.buffers.ssimScratchInfo()};
    constexpr std::array<uint32_t, 5> bindings = {6u, 7u, 12u, 28u, 42u};
    std::array<vk::WriteDescriptorSet, 5> writes{};
    const uint32_t writeCount = separable ? 5u : 4u;
    for (size_t index = 0; index < writeCount; ++index) {
        writes[index]
            .setDstSet(descriptorSet)
            .setDstBinding(bindings[index])
            .setDescriptorCount(1)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer)
            .setPBufferInfo(&infos[index]);
    }
    device_.updateDescriptorSets(
        vk::ArrayProxy<const vk::WriteDescriptorSet>(writeCount, writes.data()), {});
    context.commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.getPipeline());
    context.commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                             pipeline.getPipelineLayout(), 0, descriptorSet, {});
    context.commandBuffer.pushConstants(pipeline.getPipelineLayout(),
                                        vk::ShaderStageFlagBits::eCompute, 0,
                                        sizeof(TrainingPushConstants), &context.pushConstants);
    if (context.pushConstants.pixelCount == 0u) {
        return;
    }
    context.commandBuffer.dispatch((context.pushConstants.width + 15u) / 16u,
                                   (context.pushConstants.height + 15u) / 16u, 1u);
    if (separable) recordScratchReadComplete(context);
    recordBackwardBufferBarrier(context.commandBuffer, {pixelGradsInfo},
                                vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite |
                                    vk::AccessFlagBits::eTransferRead);
}

void BackwardLossPass::recordSsimHorizontal(const BackwardPassContext& context, bool coefficients) {
    if (context.pushConstants.pixelCount == 0u) return;
    const auto scratch = context.buffers.ssimScratchInfo();
    if (!scratch.buffer) throw std::runtime_error("Separable SSIM scratch is not allocated");
    const auto& pipeline =
        coefficients ? ssimCoefficientsHorizontalPipeline_ : ssimMomentsHorizontalPipeline_;
    const auto descriptorSet =
        coefficients ? ssimCoefficientsHorizontalSet_ : ssimMomentsHorizontalSet_;
    const std::array<vk::DescriptorBufferInfo, 3> infos = {
        coefficients ? context.buffers.ssimBackwardStatesInfo()
                     : context.buffers.renderedColorInfo(),
        scratch, context.buffers.targetColorInfo()};
    const std::array<uint32_t, 3> bindings = {coefficients ? 28u : 6u, 42u, 7u};
    std::array<vk::WriteDescriptorSet, 3> writes{};
    const uint32_t writeCount = coefficients ? 2u : 3u;
    for (size_t i = 0; i < writeCount; ++i) {
        writes[i]
            .setDstSet(descriptorSet)
            .setDstBinding(bindings[i])
            .setDescriptorCount(1)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer)
            .setPBufferInfo(&infos[i]);
    }
    device_.updateDescriptorSets(
        vk::ArrayProxy<const vk::WriteDescriptorSet>(writeCount, writes.data()), {});
    context.commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.getPipeline());
    context.commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                             pipeline.getPipelineLayout(), 0, descriptorSet, {});
    context.commandBuffer.pushConstants(pipeline.getPipelineLayout(),
                                        vk::ShaderStageFlagBits::eCompute, 0,
                                        sizeof(TrainingPushConstants), &context.pushConstants);
    context.commandBuffer.dispatch((context.pushConstants.width + 15u) / 16u,
                                   (context.pushConstants.height + 15u) / 16u, 1u);
    recordBackwardBufferBarrier(context.commandBuffer, {scratch}, vk::AccessFlagBits::eShaderRead);
}

void BackwardLossPass::recordScratchReadComplete(const BackwardPassContext& context) {
    // WAR: vertical consumers must finish before backward/next-step horizontal
    // overwrites the same allocation. A shader-write-only barrier is insufficient.
    const auto scratch = context.buffers.ssimScratchInfo();
    vk::BufferMemoryBarrier barrier{};
    barrier.setSrcAccessMask(vk::AccessFlagBits::eShaderRead)
        .setDstAccessMask(vk::AccessFlagBits::eShaderWrite)
        .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setBuffer(scratch.buffer)
        .setOffset(scratch.offset)
        .setSize(scratch.range);
    context.commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                          vk::PipelineStageFlagBits::eComputeShader, {}, {},
                                          barrier, {});
}

} // namespace vulkan3DGS
