#include "gaussian_training/gaussian_forward_renderer.hpp"

#include "utils/logger.hpp"

#include <array>
#include <stdexcept>
#include <vector>

namespace vk_gs {

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

ComputePipelineConfig pipelineConfig(std::initializer_list<vk::DescriptorSetLayoutBinding> bindings) {
    ComputePipelineConfig config{};
    config.descriptorBindings.assign(bindings.begin(), bindings.end());
    config.pushConstantSize = sizeof(TrainingPushConstants);
    return config;
}

uint32_t ceilDiv(uint32_t value, uint32_t divisor) {
    return (value + divisor - 1u) / divisor;
}

uint32_t tileCount(TrainingExtent extent) {
    constexpr uint32_t tileSize = 16;
    return ceilDiv(extent.width, tileSize) * ceilDiv(extent.height, tileSize);
}

} // namespace

void GaussianForwardRenderer::initialize(vk::Device device,
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
    createForwardResources();
    initialized_ = true;

    LOG_INFO("GaussianForwardRenderer initialized ({} gaussians, {}x{})",
             gaussianCount_, extent_.width, extent_.height);
}

void GaussianForwardRenderer::cleanup() {
    destroyForwardResources();
    device_ = nullptr;
    physicalDevice_ = nullptr;
    computeQueue_ = nullptr;
    computeQueueFamilyIndex_ = 0;
    gaussianCount_ = 0;
    extent_ = {};
    trainingBuffers_ = nullptr;
    commandBuffer_ = nullptr;
    pushConstants_ = {};
    initialized_ = false;
}

void GaussianForwardRenderer::setTrainingBuffers(const TrainingBuffers& trainingBuffers,
                                                 vk::CommandBuffer commandBuffer,
                                                 TrainingPushConstants pushConstants) {
    trainingBuffers_ = &trainingBuffers;
    commandBuffer_ = commandBuffer;
    pushConstants_ = pushConstants;
}

void GaussianForwardRenderer::forward() {
    if (!initialized_) {
        LOG_WARN("GaussianForwardRenderer::forward called before initialization");
        return;
    }
    if (!trainingBuffers_) {
        LOG_WARN("GaussianForwardRenderer::forward called without training buffers");
        return;
    }
    if (!commandBuffer_) {
        LOG_WARN("GaussianForwardRenderer::forward called without a command buffer");
        return;
    }

    clearForwardBuffers();
    projectGaussians();
    clearTileRanges();
    countTileCoverage();
    prefixTileRanges();
    emitTileItems();
    sortTileItems();
    compositePixels();
}

void GaussianForwardRenderer::createForwardResources() {
    clearPipeline_.initialize(device_, "shaders/train_clear.comp.spv",
                              pipelineConfig({storageBinding(1), storageBinding(6), storageBinding(8), storageBinding(9)}));
    projectPipeline_.initialize(device_, "shaders/train_forward_project.comp.spv",
                                pipelineConfig({storageBinding(0), storageBinding(3), uniformBinding(11)}));
    tileClearPipeline_.initialize(device_, "shaders/train_forward_tile_clear.comp.spv",
                                  pipelineConfig({storageBinding(5)}));
    tileCountPipeline_.initialize(device_, "shaders/train_forward_tile_count.comp.spv",
                                  pipelineConfig({storageBinding(3), storageBinding(5)}));
    tilePrefixPipeline_.initialize(device_, "shaders/train_forward_tile_prefix.comp.spv",
                                   pipelineConfig({storageBinding(5), storageBinding(9)}));
    tileEmitPipeline_.initialize(device_, "shaders/train_forward_tile_emit.comp.spv",
                                 pipelineConfig({storageBinding(3), storageBinding(4), storageBinding(5)}));
    tileSortPipeline_.initialize(device_, "shaders/train_forward_tile_sort.comp.spv",
                                 pipelineConfig({storageBinding(3), storageBinding(4), storageBinding(5)}));
    forwardPipeline_.initialize(device_, "shaders/train_forward.comp.spv",
                                pipelineConfig({storageBinding(3), storageBinding(4), storageBinding(5), storageBinding(6)}));

    std::array<vk::DescriptorPoolSize, 2> poolSizes{};
    poolSizes[0].setType(vk::DescriptorType::eStorageBuffer)
                .setDescriptorCount(24);
    poolSizes[1].setType(vk::DescriptorType::eUniformBuffer)
                .setDescriptorCount(1);

    vk::DescriptorPoolCreateInfo poolInfo{};
    poolInfo.setPoolSizeCount(static_cast<uint32_t>(poolSizes.size()))
            .setPPoolSizes(poolSizes.data())
            .setMaxSets(8);
    descriptorPool_ = device_.createDescriptorPool(poolInfo);

    std::array<vk::DescriptorSetLayout, 8> layouts = {
        clearPipeline_.getDescriptorSetLayout(),
        projectPipeline_.getDescriptorSetLayout(),
        tileClearPipeline_.getDescriptorSetLayout(),
        tileCountPipeline_.getDescriptorSetLayout(),
        tilePrefixPipeline_.getDescriptorSetLayout(),
        tileEmitPipeline_.getDescriptorSetLayout(),
        tileSortPipeline_.getDescriptorSetLayout(),
        forwardPipeline_.getDescriptorSetLayout(),
    };

    vk::DescriptorSetAllocateInfo allocInfo{};
    allocInfo.setDescriptorPool(descriptorPool_)
             .setDescriptorSetCount(static_cast<uint32_t>(layouts.size()))
             .setPSetLayouts(layouts.data());

    std::vector<vk::DescriptorSet> sets = device_.allocateDescriptorSets(allocInfo);
    clearDescriptorSet_ = sets[0];
    projectDescriptorSet_ = sets[1];
    tileClearDescriptorSet_ = sets[2];
    tileCountDescriptorSet_ = sets[3];
    tilePrefixDescriptorSet_ = sets[4];
    tileEmitDescriptorSet_ = sets[5];
    tileSortDescriptorSet_ = sets[6];
    forwardDescriptorSet_ = sets[7];
}

void GaussianForwardRenderer::destroyForwardResources() {
    if (descriptorPool_) {
        device_.destroyDescriptorPool(descriptorPool_);
        descriptorPool_ = nullptr;
        clearDescriptorSet_ = nullptr;
        projectDescriptorSet_ = nullptr;
        tileClearDescriptorSet_ = nullptr;
        tileCountDescriptorSet_ = nullptr;
        tilePrefixDescriptorSet_ = nullptr;
        tileEmitDescriptorSet_ = nullptr;
        tileSortDescriptorSet_ = nullptr;
        forwardDescriptorSet_ = nullptr;
    }

    forwardPipeline_.cleanup();
    tileSortPipeline_.cleanup();
    tileEmitPipeline_.cleanup();
    tilePrefixPipeline_.cleanup();
    tileCountPipeline_.cleanup();
    tileClearPipeline_.cleanup();
    projectPipeline_.cleanup();
    clearPipeline_.cleanup();
}

void GaussianForwardRenderer::clearForwardBuffers() {
    updateDescriptorSet(clearDescriptorSet_, {1, 6, 8, 9});
    bindAndDispatch(clearPipeline_, clearDescriptorSet_, std::max(ceilDiv(pushConstants_.pixelCount, 256u),
                                                                  ceilDiv(pushConstants_.gaussianCount, 256u)));
    shaderBufferBarrier({trainingBuffers_->gaussianGradsInfo(),
                         trainingBuffers_->renderedColorInfo(),
                         trainingBuffers_->lossInfo(),
                         trainingBuffers_->countersInfo()},
                        vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
}

void GaussianForwardRenderer::projectGaussians() {
    updateDescriptorSet(projectDescriptorSet_, {0, 3, 11});
    bindAndDispatch(projectPipeline_, projectDescriptorSet_, ceilDiv(pushConstants_.gaussianCount, 256u));
    shaderBufferBarrier({trainingBuffers_->projectedInfo()},
                        vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
}

void GaussianForwardRenderer::clearTileRanges() {
    updateDescriptorSet(tileClearDescriptorSet_, {5});
    bindAndDispatch(tileClearPipeline_, tileClearDescriptorSet_, ceilDiv(tileCount(extent_), 256u));
    shaderBufferBarrier({trainingBuffers_->tileRangesInfo()},
                        vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
}

void GaussianForwardRenderer::countTileCoverage() {
    updateDescriptorSet(tileCountDescriptorSet_, {3, 5});
    bindAndDispatch(tileCountPipeline_, tileCountDescriptorSet_, ceilDiv(pushConstants_.gaussianCount, 256u));
    shaderBufferBarrier({trainingBuffers_->tileRangesInfo()},
                        vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
}

void GaussianForwardRenderer::prefixTileRanges() {
    updateDescriptorSet(tilePrefixDescriptorSet_, {5, 9});
    bindAndDispatch(tilePrefixPipeline_, tilePrefixDescriptorSet_, 1);
    shaderBufferBarrier({trainingBuffers_->tileRangesInfo(), trainingBuffers_->countersInfo()},
                        vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
}

void GaussianForwardRenderer::emitTileItems() {
    updateDescriptorSet(tileEmitDescriptorSet_, {3, 4, 5});
    bindAndDispatch(tileEmitPipeline_, tileEmitDescriptorSet_, ceilDiv(pushConstants_.gaussianCount, 256u));
    shaderBufferBarrier({trainingBuffers_->tileItemsInfo(), trainingBuffers_->tileRangesInfo()},
                        vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
}

void GaussianForwardRenderer::sortTileItems() {
    updateDescriptorSet(tileSortDescriptorSet_, {3, 4, 5});
    bindAndDispatch(tileSortPipeline_, tileSortDescriptorSet_, tileCount(extent_));
    shaderBufferBarrier({trainingBuffers_->tileItemsInfo()},
                        vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
}

void GaussianForwardRenderer::compositePixels() {
    updateDescriptorSet(forwardDescriptorSet_, {3, 4, 5, 6});
    bindAndDispatch(forwardPipeline_, forwardDescriptorSet_, ceilDiv(extent_.width, 16u), ceilDiv(extent_.height, 16u));
    shaderBufferBarrier({trainingBuffers_->renderedColorInfo()},
                        vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
}

void GaussianForwardRenderer::bindAndDispatch(ComputePipeline& pipeline,
                                              vk::DescriptorSet descriptorSet,
                                              uint32_t groupCountX,
                                              uint32_t groupCountY,
                                              uint32_t groupCountZ) {
    if (groupCountX == 0 || groupCountY == 0 || groupCountZ == 0) {
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
                                 sizeof(TrainingPushConstants),
                                 &pushConstants_);
    commandBuffer_.dispatch(groupCountX, groupCountY, groupCountZ);
}

void GaussianForwardRenderer::shaderBufferBarrier(std::initializer_list<vk::DescriptorBufferInfo> buffers,
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

    commandBuffer_.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                   vk::PipelineStageFlagBits::eComputeShader,
                                   vk::DependencyFlagBits{},
                                   0,
                                   nullptr,
                                   static_cast<uint32_t>(barriers.size()),
                                   barriers.data(),
                                   0,
                                   nullptr);
}

void GaussianForwardRenderer::updateDescriptorSet(vk::DescriptorSet descriptorSet,
                                                  std::initializer_list<uint32_t> bindings) {
    auto gaussianParamsInfo = trainingBuffers_->gaussianParamsInfo();
    auto gaussianGradsInfo = trainingBuffers_->gaussianGradsInfo();
    auto projectedInfo = trainingBuffers_->projectedInfo();
    auto tileItemsInfo = trainingBuffers_->tileItemsInfo();
    auto tileRangesInfo = trainingBuffers_->tileRangesInfo();
    auto renderedColorInfo = trainingBuffers_->renderedColorInfo();
    auto lossInfo = trainingBuffers_->lossInfo();
    auto countersInfo = trainingBuffers_->countersInfo();
    auto cameraInfo = trainingBuffers_->cameraInfo();

    std::array<vk::DescriptorBufferInfo, 12> infos{};
    infos[0] = gaussianParamsInfo;
    infos[1] = gaussianGradsInfo;
    infos[3] = projectedInfo;
    infos[4] = tileItemsInfo;
    infos[5] = tileRangesInfo;
    infos[6] = renderedColorInfo;
    infos[8] = lossInfo;
    infos[9] = countersInfo;
    infos[11] = cameraInfo;

    std::array<vk::WriteDescriptorSet, 9> writes{};
    uint32_t writeCount = 0;
    for (uint32_t binding : bindings) {
        if (!infos[binding].buffer) {
            continue;
        }
        writes[writeCount].setDstSet(descriptorSet)
                          .setDstBinding(binding)
                          .setDescriptorCount(1)
                          .setDescriptorType(binding == 11 ? vk::DescriptorType::eUniformBuffer
                                                           : vk::DescriptorType::eStorageBuffer)
                          .setPBufferInfo(&infos[binding]);
        ++writeCount;
    }

    device_.updateDescriptorSets(writeCount, writes.data(), 0, nullptr);
}

} // namespace vk_gs
