#include "gaussian_training/gaussian_backward_renderer.hpp"

#include "utils/logger.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
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

bool supportsPixelTo2DGSSubgroup(vk::PhysicalDevice physicalDevice, uint32_t& subgroupSize) {
    vk::PhysicalDeviceSubgroupProperties subgroupProperties{};
    vk::PhysicalDeviceProperties2 properties{};
    properties.pNext = &subgroupProperties;
    physicalDevice.getProperties2(&properties);

    const bool supportsCompute = static_cast<bool>(
        subgroupProperties.supportedStages & vk::ShaderStageFlagBits::eCompute);
    const bool supportsBasic = static_cast<bool>(
        subgroupProperties.supportedOperations & vk::SubgroupFeatureFlagBits::eBasic);
    const bool supportsShuffle = static_cast<bool>(
        subgroupProperties.supportedOperations & vk::SubgroupFeatureFlagBits::eShuffle);
    subgroupSize = subgroupProperties.subgroupSize;
    return subgroupSize > 0u && supportsCompute && supportsBasic && supportsShuffle;
}

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
    subgroupPixelTo2DGSSupported_ = supportsPixelTo2DGSSubgroup(physicalDevice_, subgroupSize_);
    computeBackwardClear_ = forceComputeBackwardClear();
    createBackwardResources();
    initialized_ = true;

    LOG_INFO("GaussianBackwardRenderer initialized ({} gaussians, {}x{})",
             gaussianCount_, extent_.width, extent_.height);
    if (subgroupPixelTo2DGSSupported_) {
        LOG_INFO("Pixel-to-2DGS subgroup and adaptive paths available (native subgroup size {})",
                 subgroupSize_);
    } else {
        LOG_INFO("Pixel-to-2DGS subgroup path unavailable; Auto and Subgroup use Direct");
    }
    if (computeBackwardClear_) {
        LOG_WARN("Using compute backward clear because VULKAN_3DGS_COMPUTE_BACKWARD_CLEAR=1");
    }
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
    subgroupSize_ = 0;
    subgroupPixelTo2DGSSupported_ = false;
    computeBackwardClear_ = false;
    initialized_ = false;
}

TrainingPixelTo2DGSMode GaussianBackwardRenderer::activePixelTo2DGSMode() const {
    if (pixelTo2DGSMode_ == TrainingPixelTo2DGSMode::Auto) {
        return subgroupPixelTo2DGSSupported_
            ? TrainingPixelTo2DGSMode::Auto
            : TrainingPixelTo2DGSMode::Direct;
    }
    if (pixelTo2DGSMode_ == TrainingPixelTo2DGSMode::Subgroup &&
        !subgroupPixelTo2DGSSupported_) {
        return TrainingPixelTo2DGSMode::Direct;
    }
    return pixelTo2DGSMode_;
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

    writeProfilingTimestamp(TrainingGpuProfileStage::BackwardClear, false);
    clearBackwardBuffers();
    writeProfilingTimestamp(TrainingGpuProfileStage::BackwardClear, true);

    writeProfilingTimestamp(TrainingGpuProfileStage::LossToPixel, false);
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

    if (pushConstants_.optimizerEnabled != 0u) {
        writeProfilingTimestamp(TrainingGpuProfileStage::Optimizer, false);
        optimizeParameters();
        writeProfilingTimestamp(TrainingGpuProfileStage::Optimizer, true);
    }
    if (pushConstants_.validationEnabled != 0u) {
        writeProfilingTimestamp(TrainingGpuProfileStage::Validation, false);
        validateGaussians();
        finalizeValidation();
        writeProfilingTimestamp(TrainingGpuProfileStage::Validation, true);
    }
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
        storageBinding(14),
        storageBinding(28),
        storageBinding(29),
    };
    lossConfig.pushConstantSize = sizeof(TrainingPushConstants);

    ComputePipelineConfig clearConfig{};
    clearConfig.descriptorBindings = {
        storageBinding(1),
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
        storageBinding(3),
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
    };
    optimizerConfig.pushConstantSize = sizeof(TrainingPushConstants);

    ComputePipelineConfig gaussianValidationConfig{};
    gaussianValidationConfig.descriptorBindings = {
        storageBinding(0),
        storageBinding(19),
        storageBinding(30),
    };
    gaussianValidationConfig.pushConstantSize = sizeof(TrainingPushConstants);

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
    pixelTo2DGSWorkgroupPipeline_.initialize(device_,
                                             "shaders/train_backward_pixel_to_2dgs_workgroup.comp.spv",
                                             pixelTo2DGSConfig);
    if (subgroupPixelTo2DGSSupported_) {
        pixelTo2DGSSubgroupPipeline_.initialize(
            device_,
            "shaders/train_backward_pixel_to_2dgs_subgroup.comp.spv",
            pixelTo2DGSConfig);
        pixelTo2DGSAdaptivePipeline_.initialize(
            device_,
            "shaders/train_backward_pixel_to_2dgs_adaptive.comp.spv",
            pixelTo2DGSConfig);
    }
    twoDGSTo3DGSPipeline_.initialize(device_, "shaders/train_backward_2dgs_to_3dgs.comp.spv", twoDGSTo3DGSConfig);
    optimizerPipeline_.initialize(device_, "shaders/train_optimizer.comp.spv", optimizerConfig);
    gaussianValidationPipeline_.initialize(device_,
                                           "shaders/train_gaussian_validation.comp.spv",
                                           gaussianValidationConfig);
    validationFinalizePipeline_.initialize(device_,
                                           "shaders/train_validation_finalize.comp.spv",
                                           validationFinalizeConfig);

    std::array<vk::DescriptorPoolSize, 2> poolSizes{};
    poolSizes[0].setType(vk::DescriptorType::eStorageBuffer)
                .setDescriptorCount(54);
    poolSizes[1].setType(vk::DescriptorType::eUniformBuffer)
                .setDescriptorCount(1);

    vk::DescriptorPoolCreateInfo poolInfo{};
    poolInfo.setPoolSizeCount(static_cast<uint32_t>(poolSizes.size()))
            .setPPoolSizes(poolSizes.data())
            .setMaxSets(11);

    descriptorPool_ = device_.createDescriptorPool(poolInfo);

    std::vector<vk::DescriptorSetLayout> layouts = {
        backwardClearPipeline_.getDescriptorSetLayout(),
        lossPipeline_.getDescriptorSetLayout(),
        lossToPixelPipeline_.getDescriptorSetLayout(),
        pixelTo2DGSPipeline_.getDescriptorSetLayout(),
        pixelTo2DGSWorkgroupPipeline_.getDescriptorSetLayout(),
        twoDGSTo3DGSPipeline_.getDescriptorSetLayout(),
        optimizerPipeline_.getDescriptorSetLayout(),
        gaussianValidationPipeline_.getDescriptorSetLayout(),
        validationFinalizePipeline_.getDescriptorSetLayout(),
    };
    if (subgroupPixelTo2DGSSupported_) {
        layouts.push_back(pixelTo2DGSSubgroupPipeline_.getDescriptorSetLayout());
        layouts.push_back(pixelTo2DGSAdaptivePipeline_.getDescriptorSetLayout());
    }

    vk::DescriptorSetAllocateInfo allocInfo{};
    allocInfo.setDescriptorPool(descriptorPool_)
             .setDescriptorSetCount(static_cast<uint32_t>(layouts.size()))
             .setPSetLayouts(layouts.data());

    std::vector<vk::DescriptorSet> descriptorSets = device_.allocateDescriptorSets(allocInfo);
    backwardClearDescriptorSet_ = descriptorSets[0];
    lossDescriptorSet_ = descriptorSets[1];
    lossToPixelDescriptorSet_ = descriptorSets[2];
    pixelTo2DGSDescriptorSet_ = descriptorSets[3];
    pixelTo2DGSWorkgroupDescriptorSet_ = descriptorSets[4];
    twoDGSTo3DGSDescriptorSet_ = descriptorSets[5];
    optimizerDescriptorSet_ = descriptorSets[6];
    gaussianValidationDescriptorSet_ = descriptorSets[7];
    validationFinalizeDescriptorSet_ = descriptorSets[8];
    if (subgroupPixelTo2DGSSupported_) {
        pixelTo2DGSSubgroupDescriptorSet_ = descriptorSets[9];
        pixelTo2DGSAdaptiveDescriptorSet_ = descriptorSets[10];
    }
}

void GaussianBackwardRenderer::destroyBackwardResources() {
    if (descriptorPool_) {
        device_.destroyDescriptorPool(descriptorPool_);
        descriptorPool_ = nullptr;
        backwardClearDescriptorSet_ = nullptr;
        lossDescriptorSet_ = nullptr;
        lossToPixelDescriptorSet_ = nullptr;
        pixelTo2DGSDescriptorSet_ = nullptr;
        pixelTo2DGSWorkgroupDescriptorSet_ = nullptr;
        pixelTo2DGSSubgroupDescriptorSet_ = nullptr;
        pixelTo2DGSAdaptiveDescriptorSet_ = nullptr;
        twoDGSTo3DGSDescriptorSet_ = nullptr;
        optimizerDescriptorSet_ = nullptr;
        gaussianValidationDescriptorSet_ = nullptr;
        validationFinalizeDescriptorSet_ = nullptr;
    }

    validationFinalizePipeline_.cleanup();
    gaussianValidationPipeline_.cleanup();
    optimizerPipeline_.cleanup();
    twoDGSTo3DGSPipeline_.cleanup();
    pixelTo2DGSAdaptivePipeline_.cleanup();
    pixelTo2DGSSubgroupPipeline_.cleanup();
    pixelTo2DGSWorkgroupPipeline_.cleanup();
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
    const auto pixelBlendStatesInfo = trainingBuffers_->pixelBlendStatesInfo();
    const auto ssimBackwardStatesInfo = trainingBuffers_->ssimBackwardStatesInfo();
    const auto pixelValidationPartialsInfo = trainingBuffers_->pixelValidationPartialsInfo();

    std::array<vk::WriteDescriptorSet, 6> writes{};
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
             .setDstBinding(14)
             .setDescriptorCount(1)
             .setDescriptorType(vk::DescriptorType::eStorageBuffer)
             .setPBufferInfo(&pixelBlendStatesInfo);
    writes[4].setDstSet(lossDescriptorSet_)
             .setDstBinding(28)
             .setDescriptorCount(1)
             .setDescriptorType(vk::DescriptorType::eStorageBuffer)
             .setPBufferInfo(&ssimBackwardStatesInfo);
    writes[5].setDstSet(lossDescriptorSet_)
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

    const uint32_t groupCountX = (pushConstants_.width + 15u) / 16u;
    const uint32_t groupCountY = (pushConstants_.height + 15u) / 16u;
    commandBuffer_.dispatch(groupCountX, groupCountY, 1);

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
    const auto projectedGradsInfo = trainingBuffers_->projectedGradsInfo();

    if (pushConstants_.gaussianCount == 0) {
        return;
    }

    if (!computeBackwardClear_) {
        const vk::DeviceSize gaussianGradsSize =
            sizeof(GaussianGrad) * static_cast<vk::DeviceSize>(pushConstants_.gaussianCount);
        const vk::DeviceSize projectedGradsSize =
            sizeof(ProjectedGaussianGrad) * static_cast<vk::DeviceSize>(pushConstants_.gaussianCount);

        commandBuffer_.fillBuffer(
            gaussianGradsInfo.buffer, gaussianGradsInfo.offset, gaussianGradsSize, 0u);
        commandBuffer_.fillBuffer(
            projectedGradsInfo.buffer, projectedGradsInfo.offset, projectedGradsSize, 0u);

        std::array<vk::BufferMemoryBarrier, 2> barriers{};
        barriers[0].setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
                   .setDstAccessMask(vk::AccessFlagBits::eShaderRead |
                                     vk::AccessFlagBits::eShaderWrite)
                   .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                   .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                   .setBuffer(gaussianGradsInfo.buffer)
                   .setOffset(gaussianGradsInfo.offset)
                   .setSize(gaussianGradsSize);
        barriers[1].setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
                   .setDstAccessMask(vk::AccessFlagBits::eShaderRead |
                                     vk::AccessFlagBits::eShaderWrite)
                   .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                   .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                   .setBuffer(projectedGradsInfo.buffer)
                   .setOffset(projectedGradsInfo.offset)
                   .setSize(projectedGradsSize);
        commandBuffer_.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                       vk::PipelineStageFlagBits::eComputeShader,
                                       vk::DependencyFlagBits{},
                                       0,
                                       nullptr,
                                       static_cast<uint32_t>(barriers.size()),
                                       barriers.data(),
                                       0,
                                       nullptr);
        return;
    }

    std::array<vk::WriteDescriptorSet, 2> writes{};
    writes[0].setDstSet(backwardClearDescriptorSet_)
             .setDstBinding(1)
             .setDescriptorCount(1)
             .setDescriptorType(vk::DescriptorType::eStorageBuffer)
             .setPBufferInfo(&gaussianGradsInfo);
    writes[1].setDstSet(backwardClearDescriptorSet_)
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

    commandBuffer_.dispatch((pushConstants_.gaussianCount + 255u) / 256u, 1, 1);
    shaderBufferBarrier({gaussianGradsInfo, projectedGradsInfo},
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

    const uint32_t groupCountX = (pushConstants_.width + 15u) / 16u;
    const uint32_t groupCountY = (pushConstants_.height + 15u) / 16u;
    commandBuffer_.dispatch(groupCountX, groupCountY, 1);
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

    const TrainingPixelTo2DGSMode activeMode = activePixelTo2DGSMode();
    ComputePipeline* pipeline = &pixelTo2DGSPipeline_;
    vk::DescriptorSet descriptorSet = pixelTo2DGSDescriptorSet_;
    if (activeMode == TrainingPixelTo2DGSMode::WorkgroupShared) {
        pipeline = &pixelTo2DGSWorkgroupPipeline_;
        descriptorSet = pixelTo2DGSWorkgroupDescriptorSet_;
    } else if (activeMode == TrainingPixelTo2DGSMode::Subgroup) {
        pipeline = &pixelTo2DGSSubgroupPipeline_;
        descriptorSet = pixelTo2DGSSubgroupDescriptorSet_;
    } else if (activeMode == TrainingPixelTo2DGSMode::Auto) {
        pipeline = &pixelTo2DGSAdaptivePipeline_;
        descriptorSet = pixelTo2DGSAdaptiveDescriptorSet_;
    }

    std::array<vk::WriteDescriptorSet, 6> writes{};
    writes[0].setDstSet(descriptorSet)
             .setDstBinding(3)
             .setDescriptorCount(1)
             .setDescriptorType(vk::DescriptorType::eStorageBuffer)
             .setPBufferInfo(&projectedInfo);
    writes[1].setDstSet(descriptorSet)
             .setDstBinding(4)
             .setDescriptorCount(1)
             .setDescriptorType(vk::DescriptorType::eStorageBuffer)
             .setPBufferInfo(&tileItemsInfo);
    writes[2].setDstSet(descriptorSet)
             .setDstBinding(5)
             .setDescriptorCount(1)
             .setDescriptorType(vk::DescriptorType::eStorageBuffer)
             .setPBufferInfo(&tileRangesInfo);
    writes[3].setDstSet(descriptorSet)
             .setDstBinding(12)
             .setDescriptorCount(1)
             .setDescriptorType(vk::DescriptorType::eStorageBuffer)
             .setPBufferInfo(&pixelGradsInfo);
    writes[4].setDstSet(descriptorSet)
             .setDstBinding(13)
             .setDescriptorCount(1)
             .setDescriptorType(vk::DescriptorType::eStorageBuffer)
             .setPBufferInfo(&projectedGradsInfo);
    writes[5].setDstSet(descriptorSet)
             .setDstBinding(14)
             .setDescriptorCount(1)
             .setDescriptorType(vk::DescriptorType::eStorageBuffer)
             .setPBufferInfo(&pixelBlendStatesInfo);
    device_.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

    commandBuffer_.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline->getPipeline());
    commandBuffer_.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                                     pipeline->getPipelineLayout(),
                                                     0,
                                                     1,
                                                     &descriptorSet,
                                                     0,
                                                     nullptr);
    commandBuffer_.pushConstants(pipeline->getPipelineLayout(),
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
    const auto projectedInfo = trainingBuffers_->projectedInfo();
    const auto projectedGradsInfo = trainingBuffers_->projectedGradsInfo();
    const auto densificationStatesInfo = trainingBuffers_->densificationStatesInfo();
    const auto cameraInfo = trainingBuffers_->cameraInfo();

    std::array<vk::WriteDescriptorSet, 6> writes{};
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
             .setDstBinding(3)
             .setDescriptorCount(1)
             .setDescriptorType(vk::DescriptorType::eStorageBuffer)
             .setPBufferInfo(&projectedInfo);
    writes[3].setDstSet(twoDGSTo3DGSDescriptorSet_)
             .setDstBinding(13)
             .setDescriptorCount(1)
             .setDescriptorType(vk::DescriptorType::eStorageBuffer)
             .setPBufferInfo(&projectedGradsInfo);
    writes[4].setDstSet(twoDGSTo3DGSDescriptorSet_)
             .setDstBinding(16)
             .setDescriptorCount(1)
             .setDescriptorType(vk::DescriptorType::eStorageBuffer)
             .setPBufferInfo(&densificationStatesInfo);
    writes[5].setDstSet(twoDGSTo3DGSDescriptorSet_)
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

    std::array<vk::WriteDescriptorSet, 3> writes{};
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
    shaderBufferBarrier({gaussianParamsInfo, adamStatesInfo},
                        vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferRead);
}

void GaussianBackwardRenderer::validateGaussians() {
    const bool validateDensified = pushConstants_.validationUsesDensifiedGaussians != 0u;
    const auto gaussianParamsInfo = validateDensified
        ? trainingBuffers_->densifiedParamsInfo()
        : trainingBuffers_->gaussianParamsInfo();
    const auto densificationCountersInfo = validateDensified
        ? trainingBuffers_->densificationCountersInfo()
        : trainingBuffers_->countersInfo();
    const auto gaussianValidationPartialsInfo = validateDensified
        ? trainingBuffers_->densifiedGaussianValidationPartialsInfo()
        : trainingBuffers_->gaussianValidationPartialsInfo();

    std::array<vk::WriteDescriptorSet, 3> writes{};
    writes[0].setDstSet(gaussianValidationDescriptorSet_)
             .setDstBinding(0)
             .setDescriptorCount(1)
             .setDescriptorType(vk::DescriptorType::eStorageBuffer)
             .setPBufferInfo(&gaussianParamsInfo);
    writes[1].setDstSet(gaussianValidationDescriptorSet_)
             .setDstBinding(19)
             .setDescriptorCount(1)
             .setDescriptorType(vk::DescriptorType::eStorageBuffer)
             .setPBufferInfo(&densificationCountersInfo);
    writes[2].setDstSet(gaussianValidationDescriptorSet_)
             .setDstBinding(30)
             .setDescriptorCount(1)
             .setDescriptorType(vk::DescriptorType::eStorageBuffer)
             .setPBufferInfo(&gaussianValidationPartialsInfo);
    device_.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

    commandBuffer_.bindPipeline(vk::PipelineBindPoint::eCompute,
                                gaussianValidationPipeline_.getPipeline());
    commandBuffer_.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                      gaussianValidationPipeline_.getPipelineLayout(),
                                      0,
                                      1,
                                      &gaussianValidationDescriptorSet_,
                                      0,
                                      nullptr);
    commandBuffer_.pushConstants(gaussianValidationPipeline_.getPipelineLayout(),
                                 vk::ShaderStageFlagBits::eCompute,
                                 0,
                                 sizeof(TrainingPushConstants),
                                 &pushConstants_);
    if (pushConstants_.gaussianCount == 0) {
        return;
    }

    commandBuffer_.dispatch((pushConstants_.gaussianCount + 255u) / 256u, 1, 1);
    shaderBufferBarrier({gaussianValidationPartialsInfo}, vk::AccessFlagBits::eShaderRead);
}

void GaussianBackwardRenderer::finalizeValidation() {
    if (pushConstants_.validationEnabled == 0u) {
        return;
    }

    const auto pixelValidationPartialsInfo = trainingBuffers_->pixelValidationPartialsInfo();
    const auto gaussianValidationPartialsInfo = pushConstants_.validationUsesDensifiedGaussians != 0u
        ? trainingBuffers_->densifiedGaussianValidationPartialsInfo()
        : trainingBuffers_->gaussianValidationPartialsInfo();
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

