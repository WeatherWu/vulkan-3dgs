#include "training/core/gaussian_densification_renderer.hpp"

#include "utils/logger.hpp"

#include <algorithm>
#include <array>
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

ComputePipelineConfig pipelineConfig(std::initializer_list<vk::DescriptorSetLayoutBinding> bindings) {
    ComputePipelineConfig config{};
    config.descriptorBindings.assign(bindings.begin(), bindings.end());
    config.pushConstantSize = sizeof(TrainingDensificationPushConstants);
    return config;
}

uint32_t ceilDiv(uint32_t value, uint32_t divisor) {
    return (value + divisor - 1u) / divisor;
}

bool supportsShaderBufferFloat32AtomicMinMax(vk::PhysicalDevice physicalDevice) {
    auto extensions = physicalDevice.enumerateDeviceExtensionProperties();
    const bool hasAtomicFloat2 = std::any_of(extensions.begin(), extensions.end(),
        [](const vk::ExtensionProperties& extension) {
            return std::strcmp(extension.extensionName.data(), vk::EXTShaderAtomicFloat2ExtensionName) == 0;
        });
    if (!hasAtomicFloat2) {
        return false;
    }

    vk::PhysicalDeviceShaderAtomicFloat2FeaturesEXT atomicFloat2Features{};
    vk::PhysicalDeviceFeatures2 features2{};
    features2.setPNext(&atomicFloat2Features);
    physicalDevice.getFeatures2(&features2);
    return atomicFloat2Features.shaderBufferFloat32AtomicMinMax;
}

} // namespace

void GaussianDensificationRenderer::initialize(vk::Device device,
                                               vk::PhysicalDevice physicalDevice,
                                               vk::Queue computeQueue,
                                               uint32_t computeQueueFamilyIndex) {
    device_ = device;
    physicalDevice_ = physicalDevice;
    computeQueue_ = computeQueue;
    computeQueueFamilyIndex_ = computeQueueFamilyIndex;
    createResources();
    initialized_ = true;
    LOG_INFO("GaussianDensificationRenderer initialized");
}

void GaussianDensificationRenderer::cleanup() {
    destroyResources();
    device_ = nullptr;
    physicalDevice_ = nullptr;
    computeQueue_ = nullptr;
    computeQueueFamilyIndex_ = 0;
    trainingBuffers_ = nullptr;
    commandBuffer_ = nullptr;
    profilingQueryPool_ = nullptr;
    pushConstants_ = {};
    initialized_ = false;
}

void GaussianDensificationRenderer::setTrainingBuffers(const TrainingBuffers& trainingBuffers,
                                                       vk::CommandBuffer commandBuffer,
                                                       TrainingDensificationPushConstants pushConstants) {
    trainingBuffers_ = &trainingBuffers;
    commandBuffer_ = commandBuffer;
    pushConstants_ = pushConstants;
}

void GaussianDensificationRenderer::setProfilingQueryPool(vk::QueryPool queryPool) {
    profilingQueryPool_ = queryPool;
}

void GaussianDensificationRenderer::densifyAndPrune() {
    if (!initialized_) {
        LOG_WARN("GaussianDensificationRenderer::densifyAndPrune called before initialization");
        return;
    }
    if (!trainingBuffers_) {
        LOG_WARN("GaussianDensificationRenderer::densifyAndPrune called without training buffers");
        return;
    }
    if (!commandBuffer_) {
        LOG_WARN("GaussianDensificationRenderer::densifyAndPrune called without command buffer");
        return;
    }

    writeProfilingTimestamp(TrainingGpuProfileStage::Densification, false);

    updateDescriptorSet(clearDescriptorSet_, {19});
    bindAndDispatch(clearPipeline_, clearDescriptorSet_, 1);
    shaderBufferBarrier({trainingBuffers_->densificationCountersInfo()},
                        vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);

    updateDescriptorSet(densifyPruneDescriptorSet_, {0, 2, 16, 17, 18, 19});
    bindAndDispatch(densifyPrunePipeline_, densifyPruneDescriptorSet_, ceilDiv(pushConstants_.gaussianCount, 256u));
    shaderBufferBarrier({trainingBuffers_->densifiedParamsInfo(),
                         trainingBuffers_->densifiedAdamStatesInfo(),
                         trainingBuffers_->densificationCountersInfo(),
                         trainingBuffers_->densificationStatesInfo()},
                        vk::AccessFlagBits::eShaderRead |
                            vk::AccessFlagBits::eShaderWrite |
                            vk::AccessFlagBits::eTransferWrite);

    updateDescriptorSet(dispatchBuildDescriptorSet_, {19});
    bindAndDispatch(dispatchBuildPipeline_, dispatchBuildDescriptorSet_, 1);
    shaderBufferBarrier({trainingBuffers_->densificationCountersInfo()},
                        vk::AccessFlagBits::eIndirectCommandRead |
                            vk::AccessFlagBits::eShaderRead |
                            vk::AccessFlagBits::eTransferRead);

    if (pushConstants_.resetOpacity != 0u) {
        updateDescriptorSet(opacityResetDescriptorSet_, {17, 18, 19});
        commandBuffer_.bindPipeline(vk::PipelineBindPoint::eCompute,
                                    opacityResetPipeline_.getPipeline());
        commandBuffer_.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                          opacityResetPipeline_.getPipelineLayout(),
                                          0,
                                          1,
                                          &opacityResetDescriptorSet_,
                                          0,
                                          nullptr);
        commandBuffer_.pushConstants(opacityResetPipeline_.getPipelineLayout(),
                                     vk::ShaderStageFlagBits::eCompute,
                                     0,
                                     sizeof(TrainingDensificationPushConstants),
                                     &pushConstants_);
        commandBuffer_.dispatchIndirect(trainingBuffers_->densificationCountersBuffer(),
                                        sizeof(uint32_t) * 12u);
    }
    shaderBufferBarrier({trainingBuffers_->densifiedParamsInfo(),
                         trainingBuffers_->densifiedAdamStatesInfo(),
                         trainingBuffers_->densificationCountersInfo()},
                        vk::AccessFlagBits::eShaderRead |
                            vk::AccessFlagBits::eShaderWrite |
                             vk::AccessFlagBits::eTransferRead);
    clearDensificationStates();
    writeProfilingTimestamp(TrainingGpuProfileStage::Densification, true);
}

void GaussianDensificationRenderer::writeProfilingTimestamp(TrainingGpuProfileStage stage, bool end) {
    if (profilingQueryPool_ && commandBuffer_) {
        commandBuffer_.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands,
                                      profilingQueryPool_,
                                      trainingGpuTimestampQuery(stage, end));
    }
}

void GaussianDensificationRenderer::createResources() {
    clearPipeline_.initialize(device_, "shaders/train_densify_clear.comp.spv",
                              pipelineConfig({storageBinding(19)}));
    const bool useUintRadiusFallback = !supportsShaderBufferFloat32AtomicMinMax(physicalDevice_);
    const char* densifyPruneShader = useUintRadiusFallback
        ? "shaders/train_densify_prune_uint_radius.comp.spv"
        : "shaders/train_densify_prune.comp.spv";
    if (useUintRadiusFallback) {
        LOG_INFO("Using uint maxScreenRadius fallback training densify/prune shader");
    }

    densifyPrunePipeline_.initialize(device_, densifyPruneShader,
                                                     pipelineConfig({storageBinding(0),
                                                                     storageBinding(2),
                                                                     storageBinding(16),
                                                                     storageBinding(17),
                                                                     storageBinding(18),
                                                                     storageBinding(19),
                                                                     }));
    opacityResetPipeline_.initialize(device_, "shaders/train_opacity_reset.comp.spv",
                                     pipelineConfig({storageBinding(17),
                                                     storageBinding(18),
                                                     storageBinding(19)}));
    dispatchBuildPipeline_.initialize(device_, "shaders/train_densify_dispatch.comp.spv",
                                      pipelineConfig({storageBinding(19)}));

    std::array<vk::DescriptorPoolSize, 1> poolSizes{};
    poolSizes[0].setType(vk::DescriptorType::eStorageBuffer)
                .setDescriptorCount(11);

    vk::DescriptorPoolCreateInfo poolInfo{};
    poolInfo.setPoolSizeCount(static_cast<uint32_t>(poolSizes.size()))
            .setPPoolSizes(poolSizes.data())
            .setMaxSets(4);
    descriptorPool_ = device_.createDescriptorPool(poolInfo);

    std::array<vk::DescriptorSetLayout, 4> layouts = {
        clearPipeline_.getDescriptorSetLayout(),
        densifyPrunePipeline_.getDescriptorSetLayout(),
        dispatchBuildPipeline_.getDescriptorSetLayout(),
        opacityResetPipeline_.getDescriptorSetLayout(),
    };

    vk::DescriptorSetAllocateInfo allocInfo{};
    allocInfo.setDescriptorPool(descriptorPool_)
             .setDescriptorSetCount(static_cast<uint32_t>(layouts.size()))
             .setPSetLayouts(layouts.data());

    std::vector<vk::DescriptorSet> sets = device_.allocateDescriptorSets(allocInfo);
    clearDescriptorSet_ = sets[0];
    densifyPruneDescriptorSet_ = sets[1];
    dispatchBuildDescriptorSet_ = sets[2];
    opacityResetDescriptorSet_ = sets[3];
}

void GaussianDensificationRenderer::destroyResources() {
    if (descriptorPool_) {
        device_.destroyDescriptorPool(descriptorPool_);
        descriptorPool_ = nullptr;
        clearDescriptorSet_ = nullptr;
        densifyPruneDescriptorSet_ = nullptr;
        dispatchBuildDescriptorSet_ = nullptr;
        opacityResetDescriptorSet_ = nullptr;
    }

    opacityResetPipeline_.cleanup();
    dispatchBuildPipeline_.cleanup();
    densifyPrunePipeline_.cleanup();
    clearPipeline_.cleanup();
}

void GaussianDensificationRenderer::updateDescriptorSet(vk::DescriptorSet descriptorSet,
                                                        std::initializer_list<uint32_t> bindings) {
    auto gaussianParamsInfo = trainingBuffers_->gaussianParamsInfo();
    auto adamStatesInfo = trainingBuffers_->adamStatesInfo();
    auto densificationStatesInfo = trainingBuffers_->densificationStatesInfo();
    auto densifiedParamsInfo = trainingBuffers_->densifiedParamsInfo();
    auto densifiedAdamStatesInfo = trainingBuffers_->densifiedAdamStatesInfo();
    auto densificationCountersInfo = trainingBuffers_->densificationCountersInfo();
    std::array<vk::DescriptorBufferInfo, 20> infos{};
    infos[0] = gaussianParamsInfo;
    infos[2] = adamStatesInfo;
    infos[16] = densificationStatesInfo;
    infos[17] = densifiedParamsInfo;
    infos[18] = densifiedAdamStatesInfo;
    infos[19] = densificationCountersInfo;

    std::array<vk::WriteDescriptorSet, 6> writes{};
    uint32_t writeCount = 0;
    for (uint32_t binding : bindings) {
        if (!infos[binding].buffer) {
            continue;
        }
        writes[writeCount].setDstSet(descriptorSet)
                          .setDstBinding(binding)
                          .setDescriptorCount(1)
                          .setDescriptorType(vk::DescriptorType::eStorageBuffer)
                          .setPBufferInfo(&infos[binding]);
        ++writeCount;
    }

    device_.updateDescriptorSets(writeCount, writes.data(), 0, nullptr);
}

void GaussianDensificationRenderer::bindAndDispatch(ComputePipeline& pipeline,
                                                    vk::DescriptorSet descriptorSet,
                                                    uint32_t groupCountX) {
    if (groupCountX == 0) {
        return;
    }

    commandBuffer_.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.getPipeline());
    commandBuffer_.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                      pipeline.getPipelineLayout(),
                                      0,
                                      1,
                                      &descriptorSet,
                                      0,
                                      nullptr);
    commandBuffer_.pushConstants(pipeline.getPipelineLayout(),
                                 vk::ShaderStageFlagBits::eCompute,
                                 0,
                                 sizeof(TrainingDensificationPushConstants),
                                 &pushConstants_);
    commandBuffer_.dispatch(groupCountX, 1, 1);
}

void GaussianDensificationRenderer::shaderBufferBarrier(std::initializer_list<vk::DescriptorBufferInfo> buffers,
                                                        vk::AccessFlags dstAccessMask) {
    std::vector<vk::BufferMemoryBarrier> barriers;
    barriers.reserve(buffers.size());
    for (const auto& buffer : buffers) {
        if (!buffer.buffer) {
            continue;
        }

        vk::BufferMemoryBarrier barrier{};
        barrier.setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
               .setDstAccessMask(dstAccessMask)
               .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
               .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
               .setBuffer(buffer.buffer)
               .setOffset(buffer.offset)
               .setSize(buffer.range);
        barriers.push_back(barrier);
    }

    if (barriers.empty()) {
        return;
    }

    vk::PipelineStageFlags dstStages = vk::PipelineStageFlagBits::eComputeShader;
    if (dstAccessMask & (vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite)) {
        dstStages |= vk::PipelineStageFlagBits::eTransfer;
    }
    if (dstAccessMask & vk::AccessFlagBits::eIndirectCommandRead) {
        dstStages |= vk::PipelineStageFlagBits::eDrawIndirect;
    }

    commandBuffer_.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                   dstStages,
                                   vk::DependencyFlagBits{},
                                   0,
                                   nullptr,
                                   static_cast<uint32_t>(barriers.size()),
                                   barriers.data(),
                                   0,
                                   nullptr);
}

void GaussianDensificationRenderer::clearDensificationStates() {
    const vk::DescriptorBufferInfo statesInfo = trainingBuffers_->densificationStatesInfo();
    if (!statesInfo.buffer || statesInfo.range == 0) {
        return;
    }

    commandBuffer_.fillBuffer(statesInfo.buffer, statesInfo.offset, statesInfo.range, 0u);

    vk::BufferMemoryBarrier ready{};
    ready.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
         .setDstAccessMask(vk::AccessFlagBits::eShaderRead |
                           vk::AccessFlagBits::eShaderWrite)
         .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
         .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
         .setBuffer(statesInfo.buffer)
         .setOffset(statesInfo.offset)
         .setSize(statesInfo.range);
    commandBuffer_.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                   vk::PipelineStageFlagBits::eComputeShader,
                                   vk::DependencyFlagBits{},
                                   0,
                                   nullptr,
                                   1,
                                   &ready,
                                   0,
                                   nullptr);
}

} // namespace vulkan3DGS
