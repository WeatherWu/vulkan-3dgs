#include "training/core/backward/pixel_to_2dgs_dispatcher.hpp"

#include "utils/logger.hpp"

#include <array>
#include <stdexcept>
#include <vector>

namespace vulkan3DGS {

namespace {

PixelTo2DGSCapabilities queryCapabilities(vk::PhysicalDevice physicalDevice) {
    PixelTo2DGSCapabilities capabilities{};
    vk::PhysicalDeviceSubgroupProperties subgroupProperties{};
    vk::PhysicalDeviceProperties2 properties{};
    properties.pNext = &subgroupProperties;
    physicalDevice.getProperties2(&properties);
    const bool supportsCompute =
        static_cast<bool>(subgroupProperties.supportedStages & vk::ShaderStageFlagBits::eCompute);
    const bool supportsBasic = static_cast<bool>(subgroupProperties.supportedOperations &
                                                 vk::SubgroupFeatureFlagBits::eBasic);
    const bool supportsShuffle = static_cast<bool>(subgroupProperties.supportedOperations &
                                                   vk::SubgroupFeatureFlagBits::eShuffle);
    capabilities.subgroupSize = subgroupProperties.subgroupSize;
    capabilities.subgroup =
        capabilities.subgroupSize > 0u && supportsCompute && supportsBasic && supportsShuffle;

    constexpr uint32_t workgroupSize = 256u;
    constexpr uint32_t minimumSubgroupSize = 16u;
    const auto limits = physicalDevice.getProperties().limits;
    const bool powerOfTwo = capabilities.subgroupSize > 0u &&
                            (capabilities.subgroupSize & (capabilities.subgroupSize - 1u)) == 0u;
    capabilities.tileGaussian = capabilities.subgroup && powerOfTwo &&
                                capabilities.subgroupSize >= minimumSubgroupSize &&
                                capabilities.subgroupSize <= workgroupSize &&
                                workgroupSize % capabilities.subgroupSize == 0u &&
                                limits.maxComputeSharedMemorySize >= 32u * 1024u;
    capabilities.vkSplatPerSplat = capabilities.subgroup && capabilities.subgroupSize <= 128u &&
                                   limits.maxComputeSharedMemorySize >= 12u * 1024u;
    capabilities.vkSplatTensor = limits.maxComputeSharedMemorySize >= 45u * 1024u;
    return capabilities;
}

} // namespace

TrainingPixelTo2DGSMode
resolvePixelTo2DGSMode(TrainingPixelTo2DGSMode requested,
                       const PixelTo2DGSCapabilities& capabilities) noexcept {
    if (requested == TrainingPixelTo2DGSMode::Auto) {
        return capabilities.subgroup ? TrainingPixelTo2DGSMode::Auto
                                     : TrainingPixelTo2DGSMode::Direct;
    }
    if (requested == TrainingPixelTo2DGSMode::Subgroup && !capabilities.subgroup) {
        return TrainingPixelTo2DGSMode::Direct;
    }
    if (requested == TrainingPixelTo2DGSMode::TileGaussianAtomic && !capabilities.tileGaussian) {
        return TrainingPixelTo2DGSMode::Direct;
    }
    if (requested == TrainingPixelTo2DGSMode::VkSplatPerSplat && !capabilities.vkSplatPerSplat) {
        return TrainingPixelTo2DGSMode::Direct;
    }
    if (requested == TrainingPixelTo2DGSMode::VkSplatTensor && !capabilities.vkSplatTensor) {
        return TrainingPixelTo2DGSMode::Direct;
    }
    return requested;
}

void PixelTo2DGSDispatcher::initialize(vk::Device device, vk::PhysicalDevice physicalDevice) {
    cleanup();
    device_ = device;
    capabilities_ = queryCapabilities(physicalDevice);

    ComputePipelineConfig config{};
    config.descriptorBindings = {
        backwardStorageBinding(3),  backwardStorageBinding(4),  backwardStorageBinding(5),
        backwardStorageBinding(12), backwardStorageBinding(13), backwardStorageBinding(14),
    };
    config.pushConstantSize = sizeof(TrainingPushConstants);
    directPipeline_.initialize(device_, "shaders/train_backward_pixel_to_2dgs.comp.spv", config);
    workgroupPipeline_.initialize(
        device_, "shaders/train_backward_pixel_to_2dgs_workgroup.comp.spv", config);
    if (capabilities_.subgroup) {
        subgroupPipeline_.initialize(
            device_, "shaders/train_backward_pixel_to_2dgs_subgroup.comp.spv", config);
        adaptivePipeline_.initialize(
            device_, "shaders/train_backward_pixel_to_2dgs_adaptive.comp.spv", config);
    }
    if (capabilities_.tileGaussian) {
        tileGaussianPipeline_.initialize(
            device_, "shaders/train_backward_tile_gaussian_atomic.comp.spv", config);
    }
    if (capabilities_.vkSplatPerSplat) {
        vkSplatPerSplatPipeline_.initialize(
            device_, "shaders/train_backward_vksplat_per_splat.comp.spv", config);
    }
    if (capabilities_.vkSplatTensor) {
        vkSplatTensorPipeline_.initialize(device_, "shaders/train_backward_vksplat_tensor.comp.spv",
                                          config);
    }

    std::vector<vk::DescriptorSetLayout> layouts = {directPipeline_.getDescriptorSetLayout(),
                                                    workgroupPipeline_.getDescriptorSetLayout()};
    if (capabilities_.subgroup) {
        layouts.push_back(subgroupPipeline_.getDescriptorSetLayout());
        layouts.push_back(adaptivePipeline_.getDescriptorSetLayout());
    }
    if (capabilities_.tileGaussian) {
        layouts.push_back(tileGaussianPipeline_.getDescriptorSetLayout());
    }
    if (capabilities_.vkSplatPerSplat) {
        layouts.push_back(vkSplatPerSplatPipeline_.getDescriptorSetLayout());
    }
    if (capabilities_.vkSplatTensor) {
        layouts.push_back(vkSplatTensorPipeline_.getDescriptorSetLayout());
    }
    const std::vector<vk::DescriptorSet> sets =
        descriptorPool_.create(device_, static_cast<uint32_t>(layouts.size()) * 6u, 0u, layouts);
    size_t index = 0u;
    directDescriptorSet_ = sets[index++];
    workgroupDescriptorSet_ = sets[index++];
    if (capabilities_.subgroup) {
        subgroupDescriptorSet_ = sets[index++];
        adaptiveDescriptorSet_ = sets[index++];
    }
    if (capabilities_.tileGaussian) tileGaussianDescriptorSet_ = sets[index++];
    if (capabilities_.vkSplatPerSplat) vkSplatPerSplatDescriptorSet_ = sets[index++];
    if (capabilities_.vkSplatTensor) vkSplatTensorDescriptorSet_ = sets[index++];

    if (capabilities_.subgroup) {
        LOG_INFO("Pixel-to-2DGS subgroup and adaptive paths available (native subgroup size {})",
                 capabilities_.subgroupSize);
    } else {
        LOG_INFO("Pixel-to-2DGS subgroup path unavailable; Auto and Subgroup use Direct");
    }
    LOG_INFO("Tile-Gaussian atomic path {}",
             capabilities_.tileGaussian ? "available" : "unavailable; selected mode uses Direct");
    LOG_INFO("VkSplat per-splat path {}",
             capabilities_.vkSplatPerSplat ? "available" : "unavailable");
    LOG_INFO("VkSplat tensor path {}", capabilities_.vkSplatTensor
                                           ? "available with shared-memory reduction"
                                           : "unavailable");
}

void PixelTo2DGSDispatcher::cleanup() {
    descriptorPool_.reset();
    directDescriptorSet_ = nullptr;
    workgroupDescriptorSet_ = nullptr;
    subgroupDescriptorSet_ = nullptr;
    adaptiveDescriptorSet_ = nullptr;
    tileGaussianDescriptorSet_ = nullptr;
    vkSplatPerSplatDescriptorSet_ = nullptr;
    vkSplatTensorDescriptorSet_ = nullptr;
    vkSplatTensorPipeline_.cleanup();
    vkSplatPerSplatPipeline_.cleanup();
    tileGaussianPipeline_.cleanup();
    adaptivePipeline_.cleanup();
    subgroupPipeline_.cleanup();
    workgroupPipeline_.cleanup();
    directPipeline_.cleanup();
    capabilities_ = {};
    device_ = nullptr;
}

TrainingPixelTo2DGSMode PixelTo2DGSDispatcher::activeMode() const noexcept {
    return resolvePixelTo2DGSMode(requestedMode_, capabilities_);
}

void PixelTo2DGSDispatcher::record(const BackwardPassContext& context) {
    const TrainingPixelTo2DGSMode mode = activeMode();
    if (mode == TrainingPixelTo2DGSMode::TileGaussianAtomic) {
        recordTileGaussian(context);
        return;
    }
    ComputePipeline* pipeline = &directPipeline_;
    vk::DescriptorSet descriptorSet = directDescriptorSet_;
    if (mode == TrainingPixelTo2DGSMode::WorkgroupShared) {
        pipeline = &workgroupPipeline_;
        descriptorSet = workgroupDescriptorSet_;
    } else if (mode == TrainingPixelTo2DGSMode::Subgroup) {
        pipeline = &subgroupPipeline_;
        descriptorSet = subgroupDescriptorSet_;
    } else if (mode == TrainingPixelTo2DGSMode::Auto) {
        pipeline = &adaptivePipeline_;
        descriptorSet = adaptiveDescriptorSet_;
    } else if (mode == TrainingPixelTo2DGSMode::VkSplatPerSplat) {
        pipeline = &vkSplatPerSplatPipeline_;
        descriptorSet = vkSplatPerSplatDescriptorSet_;
    } else if (mode == TrainingPixelTo2DGSMode::VkSplatTensor) {
        pipeline = &vkSplatTensorPipeline_;
        descriptorSet = vkSplatTensorDescriptorSet_;
    }
    const auto projectedInfo = context.buffers.projectedInfo();
    const auto tileItemsInfo = context.buffers.tileItemsInfo();
    const auto tileRangesInfo = context.buffers.tileRangesInfo();
    const auto pixelGradsInfo = context.buffers.pixelGradsInfo();
    const auto projectedGradsInfo = context.buffers.projectedGradsInfo();
    const auto pixelBlendStatesInfo = context.buffers.pixelBlendStatesInfo();
    std::array<vk::DescriptorBufferInfo, 6> infos = {projectedInfo,      tileItemsInfo,
                                                     tileRangesInfo,     pixelGradsInfo,
                                                     projectedGradsInfo, pixelBlendStatesInfo};
    constexpr std::array<uint32_t, 6> bindings = {3u, 4u, 5u, 12u, 13u, 14u};
    std::array<vk::WriteDescriptorSet, 6> writes{};
    for (size_t index = 0; index < writes.size(); ++index) {
        writes[index]
            .setDstSet(descriptorSet)
            .setDstBinding(bindings[index])
            .setDescriptorCount(1)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer)
            .setPBufferInfo(&infos[index]);
    }
    device_.updateDescriptorSets(writes, {});
    context.commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline->getPipeline());
    context.commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                             pipeline->getPipelineLayout(), 0, descriptorSet, {});
    context.commandBuffer.pushConstants(pipeline->getPipelineLayout(),
                                        vk::ShaderStageFlagBits::eCompute, 0,
                                        sizeof(TrainingPushConstants), &context.pushConstants);
    if (context.pushConstants.width == 0u || context.pushConstants.height == 0u) return;

    const bool profileTileLocal = mode == TrainingPixelTo2DGSMode::VkSplatPerSplat ||
                                  mode == TrainingPixelTo2DGSMode::VkSplatTensor;
    if (profileTileLocal) {
        writeBackwardProfilingTimestamp(context.commandBuffer, context.profilingQueryPool,
                                        TrainingGpuProfileStage::TileLocalBackward, false);
    }
    context.commandBuffer.dispatch((context.pushConstants.width + 15u) / 16u,
                                   (context.pushConstants.height + 15u) / 16u, 1u);
    if (profileTileLocal) {
        writeBackwardProfilingTimestamp(context.commandBuffer, context.profilingQueryPool,
                                        TrainingGpuProfileStage::TileLocalBackward, true);
    }
    recordBackwardBufferBarrier(context.commandBuffer, {projectedGradsInfo},
                                vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
}

void PixelTo2DGSDispatcher::recordTileGaussian(const BackwardPassContext& context) {
    if (!tileGaussianDescriptorSet_) {
        throw std::runtime_error("Tile-Gaussian backward descriptor set is not initialized");
    }
    const auto projectedInfo = context.buffers.projectedInfo();
    const auto tileItemsInfo = context.buffers.tileItemsInfo();
    const auto tileRangesInfo = context.buffers.tileRangesInfo();
    const auto pixelGradsInfo = context.buffers.pixelGradsInfo();
    const auto projectedGradsInfo = context.buffers.projectedGradsInfo();
    const auto pixelBlendStatesInfo = context.buffers.pixelBlendStatesInfo();
    std::array<vk::DescriptorBufferInfo, 6> infos = {projectedInfo,      tileItemsInfo,
                                                     tileRangesInfo,     pixelGradsInfo,
                                                     projectedGradsInfo, pixelBlendStatesInfo};
    constexpr std::array<uint32_t, 6> bindings = {3u, 4u, 5u, 12u, 13u, 14u};
    std::array<vk::WriteDescriptorSet, 6> writes{};
    for (size_t index = 0; index < writes.size(); ++index) {
        writes[index]
            .setDstSet(tileGaussianDescriptorSet_)
            .setDstBinding(bindings[index])
            .setDescriptorCount(1)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer)
            .setPBufferInfo(&infos[index]);
    }
    device_.updateDescriptorSets(writes, {});
    context.commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute,
                                       tileGaussianPipeline_.getPipeline());
    context.commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                             tileGaussianPipeline_.getPipelineLayout(), 0,
                                             tileGaussianDescriptorSet_, {});
    context.commandBuffer.pushConstants(tileGaussianPipeline_.getPipelineLayout(),
                                        vk::ShaderStageFlagBits::eCompute, 0,
                                        sizeof(TrainingPushConstants), &context.pushConstants);
    if (context.pushConstants.width == 0u || context.pushConstants.height == 0u) return;
    writeBackwardProfilingTimestamp(context.commandBuffer, context.profilingQueryPool,
                                    TrainingGpuProfileStage::TileLocalBackward, false);
    context.commandBuffer.dispatch((context.pushConstants.width + 15u) / 16u,
                                   (context.pushConstants.height + 15u) / 16u, 1u);
    writeBackwardProfilingTimestamp(context.commandBuffer, context.profilingQueryPool,
                                    TrainingGpuProfileStage::TileLocalBackward, true);
    recordBackwardBufferBarrier(context.commandBuffer, {projectedGradsInfo},
                                vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
}

} // namespace vulkan3DGS
