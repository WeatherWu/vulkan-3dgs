#include "gaussian_training/gaussian_backward_renderer.hpp"

#include "utils/logger.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <vector>

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

vk::DescriptorSetLayoutBinding uniformBinding(uint32_t binding) {
    vk::DescriptorSetLayoutBinding layoutBinding{};
    layoutBinding.setBinding(binding)
                 .setDescriptorType(vk::DescriptorType::eUniformBuffer)
                 .setDescriptorCount(1)
                 .setStageFlags(vk::ShaderStageFlagBits::eCompute);
    return layoutBinding;
}

} // namespace

void GaussianBackwardRenderer::initialize(vk::Device device,
                                          vk::PhysicalDevice physicalDevice,
                                          vk::Queue computeQueue,
                                          uint32_t computeQueueFamilyIndex,
                                          uint32_t gaussianCount,
                                          TrainingExtent extent) {
    device_ = device;
    physicalDevice_ = physicalDevice;
    computeQueue_ = computeQueue;
    computeQueueFamilyIndex_ = computeQueueFamilyIndex;
    gaussianCount_ = gaussianCount;
    extent_ = extent;
    createBackwardResources();
    initialized_ = true;

    LOG_INFO("GaussianBackwardRenderer initialized ({} gaussians, {}x{})",
             gaussianCount_, extent_.width, extent_.height);
}

void GaussianBackwardRenderer::cleanup() {
    destroyBackwardResources();
    device_ = nullptr;
    physicalDevice_ = nullptr;
    computeQueue_ = nullptr;
    computeQueueFamilyIndex_ = 0;
    gaussianCount_ = 0;
    extent_ = {};
    trainingBuffers_ = nullptr;
    commandBuffer_ = nullptr;
    profilingQueryPool_ = nullptr;
    pushConstants_ = {};
    initialized_ = false;
}

void GaussianBackwardRenderer::setTrainingBuffers(const TrainingBuffers& trainingBuffers,
                                                   vk::CommandBuffer commandBuffer,
                                                   TrainingPushConstants pushConstants) {
    trainingBuffers_ = &trainingBuffers;
    commandBuffer_ = commandBuffer;
    pushConstants_ = pushConstants;
}

void GaussianBackwardRenderer::setProfilingQueryPool(vk::QueryPool queryPool) {
    profilingQueryPool_ = queryPool;
}

void GaussianBackwardRenderer::backward() {
    if (!initialized_) {
        LOG_WARN("GaussianBackwardRenderer::backward called before initialization");
        return;
    }

    if (!trainingBuffers_) {
        LOG_WARN("GaussianBackwardRenderer::backward called without training buffers");
        return;
    }

    if (!commandBuffer_) {
        LOG_WARN("GaussianBackwardRenderer::backward called without a command buffer");
        return;
    }

    writeProfilingTimestamp(TrainingGpuProfileStage::Loss, false);
    computeLoss();
    writeProfilingTimestamp(TrainingGpuProfileStage::Loss, true);

    writeProfilingTimestamp(TrainingGpuProfileStage::LossToPixel, false);
    clearBackwardBuffers();
    computeLossToPixel();
    writeProfilingTimestamp(TrainingGpuProfileStage::LossToPixel, true);

    writeProfilingTimestamp(TrainingGpuProfileStage::PixelTo2DGS, false);
    backpropPixelTo2DGS();
    writeProfilingTimestamp(TrainingGpuProfileStage::PixelTo2DGS, true);

    writeProfilingTimestamp(TrainingGpuProfileStage::TwoDGSTo3DGS, false);
    backprop2DGSTo3DGS();
    writeProfilingTimestamp(TrainingGpuProfileStage::TwoDGSTo3DGS, true);
}

void GaussianBackwardRenderer::gradientDescent() {
    if (!initialized_) {
        LOG_WARN("GaussianBackwardRenderer::gradientDescent called before initialization");
        return;
    }

    if (!trainingBuffers_) {
        LOG_WARN("GaussianBackwardRenderer::gradientDescent called without training buffers");
        return;
    }

    if (!commandBuffer_) {
        LOG_WARN("GaussianBackwardRenderer::gradientDescent called without a command buffer");
        return;
    }

    writeProfilingTimestamp(TrainingGpuProfileStage::Optimizer, false);
    optimizeParameters();
    finalizeValidation();
    writeProfilingTimestamp(TrainingGpuProfileStage::Optimizer, true);
}

void GaussianBackwardRenderer::writeProfilingTimestamp(TrainingGpuProfileStage stage, bool end) {
    if (profilingQueryPool_ && commandBuffer_) {
        commandBuffer_.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands,
                                      profilingQueryPool_,
                                      trainingGpuTimestampQuery(stage, end));
    }
}

void GaussianBackwardRenderer::createBackwardResources() {
    ComputePipelineConfig lossConfig{};
    lossConfig.descriptorBindings = {
        storageBinding(6),
        storageBinding(7),
        storageBinding(8),
        storageBinding(28),
        storageBinding(29),
    };
    lossConfig.pushConstantSize = sizeof(TrainingPushConstants);

    ComputePipelineConfig clearConfig{};
    clearConfig.descriptorBindings = {
        storageBinding(1),
        storageBinding(12),
        storageBinding(13),
    };
    clearConfig.pushConstantSize = sizeof(TrainingPushConstants);

    ComputePipelineConfig lossToPixelConfig{};
    lossToPixelConfig.descriptorBindings = {
        storageBinding(6),
        storageBinding(7),
        storageBinding(12),
        storageBinding(28),
    };
    lossToPixelConfig.pushConstantSize = sizeof(TrainingPushConstants);

    ComputePipelineConfig pixelTo2DGSConfig{};
    pixelTo2DGSConfig.descriptorBindings = {
        storageBinding(3),
        storageBinding(4),
        storageBinding(5),
        storageBinding(12),
        storageBinding(13),
        storageBinding(14),
    };
    pixelTo2DGSConfig.pushConstantSize = sizeof(TrainingPushConstants);

    ComputePipelineConfig twoDGSTo3DGSConfig{};
    twoDGSTo3DGSConfig.descriptorBindings = {
        storageBinding(0),
        storageBinding(1),
        storageBinding(13),
        storageBinding(16),
        uniformBinding(11),
    };
    twoDGSTo3DGSConfig.pushConstantSize = sizeof(TrainingPushConstants);

    ComputePipelineConfig optimizerConfig{};
    optimizerConfig.descriptorBindings = {
        storageBinding(0),
        storageBinding(1),
        storageBinding(2),
        storageBinding(9),
        storageBinding(30),
    };
    optimizerConfig.pushConstantSize = sizeof(TrainingPushConstants);

    ComputePipelineConfig validationFinalizeConfig{};
    validationFinalizeConfig.descriptorBindings = {
        storageBinding(29),
        storageBinding(30),
        storageBinding(31),
    };
    validationFinalizeConfig.pushConstantSize = sizeof(TrainingPushConstants);

    backwardClearPipeline_.initialize(device_, "shaders/train_backward_clear.comp.spv", clearConfig);
    lossPipeline_.initialize(device_, "shaders/train_loss.comp.spv", lossConfig);
    lossToPixelPipeline_.initialize(device_, "shaders/train_backward_loss_to_pixel.comp.spv", lossToPixelConfig);
    pixelTo2DGSPipeline_.initialize(device_, "shaders/train_backward_pixel_to_2dgs.comp.spv", pixelTo2DGSConfig);
    twoDGSTo3DGSPipeline_.initialize(device_, "shaders/train_backward_2dgs_to_3dgs.comp.spv", twoDGSTo3DGSConfig);
    optimizerPipeline_.initialize(device_, "shaders/train_optimizer.comp.spv", optimizerConfig);
    validationFinalizePipeline_.initialize(device_,
                                           "shaders/train_validation_finalize.comp.spv",
                                           validationFinalizeConfig);

    std::array<vk::DescriptorPoolSize, 2> poolSizes{};
    poolSizes[0].setType(vk::DescriptorType::eStorageBuffer)
                .setDescriptorCount(31);
    poolSizes[1].setType(vk::DescriptorType::eUniformBuffer)
                .setDescriptorCount(1);

    vk::DescriptorPoolCreateInfo poolInfo{};
    poolInfo.setPoolSizeCount(static_cast<uint32_t>(poolSizes.size()))
            .setPPoolSizes(poolSizes.data())
            .setMaxSets(7);

    descriptorPool_ = device_.createDescriptorPool(poolInfo);

    std::array<vk::DescriptorSetLayout, 7> layouts = {
        backwardClearPipeline_.getDescriptorSetLayout(),
        lossPipeline_.getDescriptorSetLayout(),
        lossToPixelPipeline_.getDescriptorSetLayout(),
        pixelTo2DGSPipeline_.getDescriptorSetLayout(),
        twoDGSTo3DGSPipeline_.getDescriptorSetLayout(),
        optimizerPipeline_.getDescriptorSetLayout(),
        validationFinalizePipeline_.getDescriptorSetLayout(),
    };

    vk::DescriptorSetAllocateInfo allocInfo{};
    allocInfo.setDescriptorPool(descriptorPool_)
             .setDescriptorSetCount(static_cast<uint32_t>(layouts.size()))
             .setPSetLayouts(layouts.data());

    std::vector<vk::DescriptorSet> descriptorSets = device_.allocateDescriptorSets(allocInfo);
    backwardClearDescriptorSet_ = descriptorSets[0];
    lossDescriptorSet_ = descriptorSets[1];
    lossToPixelDescriptorSet_ = descriptorSets[2];
    pixelTo2DGSDescriptorSet_ = descriptorSets[3];
    twoDGSTo3DGSDescriptorSet_ = descriptorSets[4];
    optimizerDescriptorSet_ = descriptorSets[5];
    validationFinalizeDescriptorSet_ = descriptorSets[6];
}

void GaussianBackwardRenderer::destroyBackwardResources() {
    if (descriptorPool_) {
        device_.destroyDescriptorPool(descriptorPool_);
        descriptorPool_ = nullptr;
        backwardClearDescriptorSet_ = nullptr;
        lossDescriptorSet_ = nullptr;
        lossToPixelDescriptorSet_ = nullptr;
        pixelTo2DGSDescriptorSet_ = nullptr;
        twoDGSTo3DGSDescriptorSet_ = nullptr;
        optimizerDescriptorSet_ = nullptr;
        validationFinalizeDescriptorSet_ = nullptr;
    }

    validationFinalizePipeline_.cleanup();
    optimizerPipeline_.cleanup();
    twoDGSTo3DGSPipeline_.cleanup();
    pixelTo2DGSPipeline_.cleanup();
    lossToPixelPipeline_.cleanup();
    lossPipeline_.cleanup();
    backwardClearPipeline_.cleanup();
}

void GaussianBackwardRenderer::computeLoss() {
    if (!lossDescriptorSet_) {
        throw std::runtime_error("Loss descriptor set is not initialized");
    }

    const auto renderedColorInfo = trainingBuffers_->renderedColorInfo();
    const auto targetColorInfo = trainingBuffers_->targetColorInfo();
    const auto lossInfo = trainingBuffers_->lossInfo();
    const auto ssimBackwardStatesInfo = trainingBuffers_->ssimBackwardStatesInfo();
    const auto pixelValidationPartialsInfo = trainingBuffers_->pixelValidationPartialsInfo();

    std::array<vk::WriteDescriptorSet, 5> writes{};
    writes[0].setDstSet(lossDescriptorSet_)
             .setDstBinding(6)
             .setDescriptorCount(1)
             .setDescriptorType(vk::DescriptorType::eStorageBuffer)
             .setPBufferInfo(&renderedColorInfo);
    writes[1].setDstSet(lossDescriptorSet_)
             .setDstBinding(7)
             .setDescriptorCount(1)
             .setDescriptorType(vk::DescriptorType::eStorageBuffer)
             .setPBufferInfo(&targetColorInfo);
    writes[2].setDstSet(lossDescriptorSet_)
             .setDstBinding(8)
             .setDescriptorCount(1)
             .setDescriptorType(vk::DescriptorType::eStorageBuffer)
             .setPBufferInfo(&lossInfo);
    writes[3].setDstSet(lossDescriptorSet_)
             .setDstBinding(28)
             .setDescriptorCount(1)
             .setDescriptorType(vk::DescriptorType::eStorageBuffer)
             .setPBufferInfo(&ssimBackwardStatesInfo);
    writes[4].setDstSet(lossDescriptorSet_)
             .setDstBinding(29)
             .setDescriptorCount(1)
             .setDescriptorType(vk::DescriptorType::eStorageBuffer)
             .setPBufferInfo(&pixelValidationPartialsInfo);

    device_.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

    commandBuffer_.bindPipeline(vk::PipelineBindPoint::eCompute, lossPipeline_.getPipeline());
    commandBuffer_.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                                     lossPipeline_.getPipelineLayout(),
                                                     0,
                                                     1,
                                                     &lossDescriptorSet_,
                                                     0,
                                                     nullptr);
    commandBuffer_.pushConstants(lossPipeline_.getPipelineLayout(),
                                                vk::ShaderStageFlagBits::eCompute,
                                                0,
                                                sizeof(TrainingPushConstants),
                                                &pushConstants_);
    if (pushConstants_.pixelCount == 0) {
        return;
    }

    uint32_t groupCount = (pushConstants_.pixelCount + 255u) / 256u;
    commandBuffer_.dispatch(groupCount, 1, 1);

    vk::BufferMemoryBarrier lossReady{};
    lossReady.setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
             .setDstAccessMask(vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eTransferRead)
             .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
             .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
             .setBuffer(lossInfo.buffer)
             .setOffset(lossInfo.offset)
             .setSize(lossInfo.range);

    commandBuffer_.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                                  vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eTransfer,
                                                  vk::DependencyFlagBits{},
                                                  0,
                                                  nullptr,
                                                  1,
                                                  &lossReady,
                                                  0,
                                                  nullptr);
    shaderBufferBarrier({ssimBackwardStatesInfo}, vk::AccessFlagBits::eShaderRead);
    if (pushConstants_.validationEnabled != 0u) {
        shaderBufferBarrier({pixelValidationPartialsInfo}, vk::AccessFlagBits::eShaderRead);
    }
}

void GaussianBackwardRenderer::clearBackwardBuffers() {
    const auto gaussianGradsInfo = trainingBuffers_->gaussianGradsInfo();
    const auto pixelGradsInfo = trainingBuffers_->pixelGradsInfo();
    const auto projectedGradsInfo = trainingBuffers_->projectedGradsInfo();

    std::array<vk::WriteDescriptorSet, 3> writes{};
    writes[0].setDstSet(backwardClearDescriptorSet_)
             .setDstBinding(1)
             .setDescriptorCount(1)
             .setDescriptorType(vk::DescriptorType::eStorageBuffer)
             .setPBufferInfo(&gaussianGradsInfo);
    writes[1].setDstSet(backwardClearDescriptorSet_)
             .setDstBinding(12)
             .setDescriptorCount(1)
             .setDescriptorType(vk::DescriptorType::eStorageBuffer)
             .setPBufferInfo(&pixelGradsInfo);
    writes[2].setDstSet(backwardClearDescriptorSet_)
             .setDstBinding(13)
             .setDescriptorCount(1)
             .setDescriptorType(vk::DescriptorType::eStorageBuffer)
             .setPBufferInfo(&projectedGradsInfo);
    device_.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

    commandBuffer_.bindPipeline(vk::PipelineBindPoint::eCompute, backwardClearPipeline_.getPipeline());
    commandBuffer_.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                                     backwardClearPipeline_.getPipelineLayout(),
                                                     0,
                                                     1,
                                                     &backwardClearDescriptorSet_,
                                                     0,
                                                     nullptr);
    commandBuffer_.pushConstants(backwardClearPipeline_.getPipelineLayout(),
                                                vk::ShaderStageFlagBits::eCompute,
                                                0,
                                                sizeof(TrainingPushConstants),
                                                &pushConstants_);

    uint32_t count = std::max(pushConstants_.gaussianCount, pushConstants_.pixelCount);
    if (count == 0) {
        return;
    }

    commandBuffer_.dispatch((count + 255u) / 256u, 1, 1);
    shaderBufferBarrier({gaussianGradsInfo, pixelGradsInfo, projectedGradsInfo},
                        vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
}

void GaussianBackwardRenderer::computeLossToPixel() {
    const auto renderedColorInfo = trainingBuffers_->renderedColorInfo();
    const auto targetColorInfo = trainingBuffers_->targetColorInfo();
    const auto pixelGradsInfo = trainingBuffers_->pixelGradsInfo();
    const auto ssimBackwardStatesInfo = trainingBuffers_->ssimBackwardStatesInfo();

    std::array<vk::WriteDescriptorSet, 4> writes{};
    writes[0].setDstSet(lossToPixelDescriptorSet_)
             .setDstBinding(6)
             .setDescriptorCount(1)
             .setDescriptorType(vk::DescriptorType::eStorageBuffer)
             .setPBufferInfo(&renderedColorInfo);
    writes[1].setDstSet(lossToPixelDescriptorSet_)
             .setDstBinding(7)
             .setDescriptorCount(1)
             .setDescriptorType(vk::DescriptorType::eStorageBuffer)
             .setPBufferInfo(&targetColorInfo);
    writes[2].setDstSet(lossToPixelDescriptorSet_)
             .setDstBinding(12)
             .setDescriptorCount(1)
             .setDescriptorType(vk::DescriptorType::eStorageBuffer)
             .setPBufferInfo(&pixelGradsInfo);
    writes[3].setDstSet(lossToPixelDescriptorSet_)
             .setDstBinding(28)
             .setDescriptorCount(1)
             .setDescriptorType(vk::DescriptorType::eStorageBuffer)
             .setPBufferInfo(&ssimBackwardStatesInfo);
    device_.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

    commandBuffer_.bindPipeline(vk::PipelineBindPoint::eCompute, lossToPixelPipeline_.getPipeline());
    commandBuffer_.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                                     lossToPixelPipeline_.getPipelineLayout(),
                                                     0,
                                                     1,
                                                     &lossToPixelDescriptorSet_,
                                                     0,
                                                     nullptr);
    commandBuffer_.pushConstants(lossToPixelPipeline_.getPipelineLayout(),
                                                vk::ShaderStageFlagBits::eCompute,
                                                0,
                                                sizeof(TrainingPushConstants),
                                                &pushConstants_);
    if (pushConstants_.pixelCount == 0) {
        return;
    }

    commandBuffer_.dispatch((pushConstants_.pixelCount + 255u) / 256u, 1, 1);
    shaderBufferBarrier({pixelGradsInfo},
                        vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
}

void GaussianBackwardRenderer::backpropPixelTo2DGS() {
    const auto projectedInfo = trainingBuffers_->projectedInfo();
    const auto tileItemsInfo = trainingBuffers_->tileItemsInfo();
    const auto tileRangesInfo = trainingBuffers_->tileRangesInfo();
    const auto pixelGradsInfo = trainingBuffers_->pixelGradsInfo();
    const auto projectedGradsInfo = trainingBuffers_->projectedGradsInfo();
    const auto pixelBlendStatesInfo = trainingBuffers_->pixelBlendStatesInfo();

    std::array<vk::WriteDescriptorSet, 6> writes{};
    writes[0].setDstSet(pixelTo2DGSDescriptorSet_)
             .setDstBinding(3)
             .setDescriptorCount(1)
             .setDescriptorType(vk::DescriptorType::eStorageBuffer)
             .setPBufferInfo(&projectedInfo);
    writes[1].setDstSet(pixelTo2DGSDescriptorSet_)
             .setDstBinding(4)
             .setDescriptorCount(1)
             .setDescriptorType(vk::DescriptorType::eStorageBuffer)
             .setPBufferInfo(&tileItemsInfo);
    writes[2].setDstSet(pixelTo2DGSDescriptorSet_)
             .setDstBinding(5)
             .setDescriptorCount(1)
             .setDescriptorType(vk::DescriptorType::eStorageBuffer)
             .setPBufferInfo(&tileRangesInfo);
    writes[3].setDstSet(pixelTo2DGSDescriptorSet_)
             .setDstBinding(12)
             .setDescriptorCount(1)
             .setDescriptorType(vk::DescriptorType::eStorageBuffer)
             .setPBufferInfo(&pixelGradsInfo);
    writes[4].setDstSet(pixelTo2DGSDescriptorSet_)
             .setDstBinding(13)
             .setDescriptorCount(1)
             .setDescriptorType(vk::DescriptorType::eStorageBuffer)
             .setPBufferInfo(&projectedGradsInfo);
    writes[5].setDstSet(pixelTo2DGSDescriptorSet_)
             .setDstBinding(14)
             .setDescriptorCount(1)
             .setDescriptorType(vk::DescriptorType::eStorageBuffer)
             .setPBufferInfo(&pixelBlendStatesInfo);
    device_.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

    commandBuffer_.bindPipeline(vk::PipelineBindPoint::eCompute, pixelTo2DGSPipeline_.getPipeline());
    commandBuffer_.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                                     pixelTo2DGSPipeline_.getPipelineLayout(),
                                                     0,
                                                     1,
                                                     &pixelTo2DGSDescriptorSet_,
                                                     0,
                                                     nullptr);
    commandBuffer_.pushConstants(pixelTo2DGSPipeline_.getPipelineLayout(),
                                                vk::ShaderStageFlagBits::eCompute,
                                                0,
                                                sizeof(TrainingPushConstants),
                                                &pushConstants_);
    if (pushConstants_.width == 0 || pushConstants_.height == 0) {
        return;
    }

    commandBuffer_.dispatch((pushConstants_.width + 15u) / 16u,
                                           (pushConstants_.height + 15u) / 16u,
                                           1);
    shaderBufferBarrier({projectedGradsInfo},
                        vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
}

void GaussianBackwardRenderer::backprop2DGSTo3DGS() {
    const auto gaussianParamsInfo = trainingBuffers_->gaussianParamsInfo();
    const auto gaussianGradsInfo = trainingBuffers_->gaussianGradsInfo();
    const auto projectedGradsInfo = trainingBuffers_->projectedGradsInfo();
    const auto densificationStatesInfo = trainingBuffers_->densificationStatesInfo();
    const auto cameraInfo = trainingBuffers_->cameraInfo();

    std::array<vk::WriteDescriptorSet, 5> writes{};
    writes[0].setDstSet(twoDGSTo3DGSDescriptorSet_)
             .setDstBinding(0)
             .setDescriptorCount(1)
             .setDescriptorType(vk::DescriptorType::eStorageBuffer)
             .setPBufferInfo(&gaussianParamsInfo);
    writes[1].setDstSet(twoDGSTo3DGSDescriptorSet_)
             .setDstBinding(1)
             .setDescriptorCount(1)
             .setDescriptorType(vk::DescriptorType::eStorageBuffer)
             .setPBufferInfo(&gaussianGradsInfo);
    writes[2].setDstSet(twoDGSTo3DGSDescriptorSet_)
             .setDstBinding(13)
             .setDescriptorCount(1)
             .setDescriptorType(vk::DescriptorType::eStorageBuffer)
             .setPBufferInfo(&projectedGradsInfo);
    writes[3].setDstSet(twoDGSTo3DGSDescriptorSet_)
             .setDstBinding(16)
             .setDescriptorCount(1)
             .setDescriptorType(vk::DescriptorType::eStorageBuffer)
             .setPBufferInfo(&densificationStatesInfo);
    writes[4].setDstSet(twoDGSTo3DGSDescriptorSet_)
             .setDstBinding(11)
             .setDescriptorCount(1)
             .setDescriptorType(vk::DescriptorType::eUniformBuffer)
             .setPBufferInfo(&cameraInfo);
    device_.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

    commandBuffer_.bindPipeline(vk::PipelineBindPoint::eCompute, twoDGSTo3DGSPipeline_.getPipeline());
    commandBuffer_.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                                     twoDGSTo3DGSPipeline_.getPipelineLayout(),
                                                     0,
                                                     1,
                                                     &twoDGSTo3DGSDescriptorSet_,
                                                     0,
                                                     nullptr);
    commandBuffer_.pushConstants(twoDGSTo3DGSPipeline_.getPipelineLayout(),
                                                vk::ShaderStageFlagBits::eCompute,
                                                0,
                                                sizeof(TrainingPushConstants),
                                                &pushConstants_);
    if (pushConstants_.gaussianCount == 0) {
        return;
    }

    commandBuffer_.dispatch((pushConstants_.gaussianCount + 255u) / 256u, 1, 1);
    shaderBufferBarrier({gaussianGradsInfo, densificationStatesInfo},
                        vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eTransferRead);
}

void GaussianBackwardRenderer::optimizeParameters() {
    const auto gaussianParamsInfo = trainingBuffers_->gaussianParamsInfo();
    const auto gaussianGradsInfo = trainingBuffers_->gaussianGradsInfo();
    const auto adamStatesInfo = trainingBuffers_->adamStatesInfo();
    const auto countersInfo = trainingBuffers_->countersInfo();
    const auto gaussianValidationPartialsInfo = trainingBuffers_->gaussianValidationPartialsInfo();

    std::array<vk::WriteDescriptorSet, 5> writes{};
    writes[0].setDstSet(optimizerDescriptorSet_)
             .setDstBinding(0)
             .setDescriptorCount(1)
             .setDescriptorType(vk::DescriptorType::eStorageBuffer)
             .setPBufferInfo(&gaussianParamsInfo);
    writes[1].setDstSet(optimizerDescriptorSet_)
             .setDstBinding(1)
             .setDescriptorCount(1)
             .setDescriptorType(vk::DescriptorType::eStorageBuffer)
             .setPBufferInfo(&gaussianGradsInfo);
    writes[2].setDstSet(optimizerDescriptorSet_)
             .setDstBinding(2)
             .setDescriptorCount(1)
             .setDescriptorType(vk::DescriptorType::eStorageBuffer)
             .setPBufferInfo(&adamStatesInfo);
    writes[3].setDstSet(optimizerDescriptorSet_)
             .setDstBinding(9)
             .setDescriptorCount(1)
             .setDescriptorType(vk::DescriptorType::eStorageBuffer)
             .setPBufferInfo(&countersInfo);
    writes[4].setDstSet(optimizerDescriptorSet_)
             .setDstBinding(30)
             .setDescriptorCount(1)
             .setDescriptorType(vk::DescriptorType::eStorageBuffer)
             .setPBufferInfo(&gaussianValidationPartialsInfo);
    device_.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

    commandBuffer_.bindPipeline(vk::PipelineBindPoint::eCompute, optimizerPipeline_.getPipeline());
    commandBuffer_.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                      optimizerPipeline_.getPipelineLayout(),
                                      0,
                                      1,
                                      &optimizerDescriptorSet_,
                                      0,
                                      nullptr);
    commandBuffer_.pushConstants(optimizerPipeline_.getPipelineLayout(),
                                 vk::ShaderStageFlagBits::eCompute,
                                 0,
                                 sizeof(TrainingPushConstants),
                                 &pushConstants_);
    if (pushConstants_.gaussianCount == 0) {
        return;
    }

    commandBuffer_.dispatch((pushConstants_.gaussianCount + 255u) / 256u, 1, 1);
    shaderBufferBarrier({gaussianParamsInfo, adamStatesInfo, countersInfo},
                        vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferRead);
    if (pushConstants_.validationEnabled != 0u) {
        shaderBufferBarrier({gaussianValidationPartialsInfo}, vk::AccessFlagBits::eShaderRead);
    }
}

void GaussianBackwardRenderer::finalizeValidation() {
    if (pushConstants_.validationEnabled == 0u) {
        return;
    }

    const auto pixelValidationPartialsInfo = trainingBuffers_->pixelValidationPartialsInfo();
    const auto gaussianValidationPartialsInfo = trainingBuffers_->gaussianValidationPartialsInfo();
    const auto validationFinalResultInfo = trainingBuffers_->validationFinalResultInfo();

    std::array<vk::WriteDescriptorSet, 3> writes{};
    writes[0].setDstSet(validationFinalizeDescriptorSet_)
             .setDstBinding(29)
             .setDescriptorCount(1)
             .setDescriptorType(vk::DescriptorType::eStorageBuffer)
             .setPBufferInfo(&pixelValidationPartialsInfo);
    writes[1].setDstSet(validationFinalizeDescriptorSet_)
             .setDstBinding(30)
             .setDescriptorCount(1)
             .setDescriptorType(vk::DescriptorType::eStorageBuffer)
             .setPBufferInfo(&gaussianValidationPartialsInfo);
    writes[2].setDstSet(validationFinalizeDescriptorSet_)
             .setDstBinding(31)
             .setDescriptorCount(1)
             .setDescriptorType(vk::DescriptorType::eStorageBuffer)
             .setPBufferInfo(&validationFinalResultInfo);
    device_.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

    commandBuffer_.bindPipeline(vk::PipelineBindPoint::eCompute,
                                validationFinalizePipeline_.getPipeline());
    commandBuffer_.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                      validationFinalizePipeline_.getPipelineLayout(),
                                      0,
                                      1,
                                      &validationFinalizeDescriptorSet_,
                                      0,
                                      nullptr);
    commandBuffer_.pushConstants(validationFinalizePipeline_.getPipelineLayout(),
                                 vk::ShaderStageFlagBits::eCompute,
                                 0,
                                 sizeof(TrainingPushConstants),
                                 &pushConstants_);
    commandBuffer_.dispatch(1, 1, 1);
    shaderBufferBarrier({validationFinalResultInfo}, vk::AccessFlagBits::eTransferRead);
}

void GaussianBackwardRenderer::shaderBufferBarrier(std::initializer_list<vk::DescriptorBufferInfo> buffers,
                                                   vk::AccessFlags dstAccessMask) {
    std::vector<vk::BufferMemoryBarrier> barriers;
    barriers.reserve(buffers.size());
    for (const auto& bufferInfo : buffers) {
        vk::BufferMemoryBarrier barrier{};
        barrier.setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
               .setDstAccessMask(dstAccessMask)
               .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
               .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
               .setBuffer(bufferInfo.buffer)
               .setOffset(bufferInfo.offset)
               .setSize(bufferInfo.range);
        barriers.push_back(barrier);
    }

    if (barriers.empty()) {
        return;
    }

    commandBuffer_.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                                  vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eTransfer,
                                                  vk::DependencyFlagBits{},
                                                  0,
                                                  nullptr,
                                                  static_cast<uint32_t>(barriers.size()),
                                                  barriers.data(),
                                                  0,
                                                  nullptr);
}

} // namespace vulkan3DGS

