#include "training/core/backward/pixel_to_2dgs_dispatcher.hpp"

#include "utils/logger.hpp"

#include <array>
#include <stdexcept>
#include <string>
#include <tuple>
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
    if (requested == TrainingPixelTo2DGSMode::CheckpointedWarp)
        return TrainingPixelTo2DGSMode::Direct; // checkpoint warp is selected by its explicit benchmark request.
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

bool supportsAdaptiveSpatialBackward(TrainingPixelTo2DGSMode requested,
                                     const PixelTo2DGSCapabilities& capabilities) noexcept {
    return (requested == TrainingPixelTo2DGSMode::VkSplatTensor && capabilities.vkSplatTensor) ||
           (requested == TrainingPixelTo2DGSMode::VkSplatPerSplat && capabilities.vkSplatPerSplat);
}

bool supportsTensorSubgroupReduction(const PixelTo2DGSCapabilities& capabilities) noexcept {
    return capabilities.vkSplatTensor && capabilities.subgroup && capabilities.subgroupSize == 32u;
}

bool useTensorSubgroupReduction(bool requested, TrainingPixelTo2DGSMode activeMode,
                                TrainingTileMode tileMode,
                                const PixelTo2DGSCapabilities& capabilities) noexcept {
    return requested && activeMode == TrainingPixelTo2DGSMode::VkSplatTensor &&
           tileMode == TrainingTileMode::Fixed16 && supportsTensorSubgroupReduction(capabilities);
}

bool useTensorGlobalPixelGrad(bool requested, bool subgroupRequested,
                              TrainingPixelTo2DGSMode activeMode, TrainingTileMode tileMode,
                              const PixelTo2DGSCapabilities& capabilities) noexcept {
    return requested && useTensorSubgroupReduction(subgroupRequested, activeMode, tileMode,
                                                    capabilities);
}

bool useTensorRgbLatencyHiding(bool requested, bool globalPixelGradRequested,
                               bool subgroupRequested, TrainingPixelTo2DGSMode activeMode,
                               TrainingTileMode tileMode,
                               const PixelTo2DGSCapabilities& capabilities) noexcept {
    return requested && useTensorGlobalPixelGrad(globalPixelGradRequested, subgroupRequested,
                                                  activeMode, tileMode, capabilities);
}

void PixelTo2DGSDispatcher::initializeVariant(TilePipelineVariant& variant, uint32_t tileSize,
                                              const ComputePipelineConfig& config) {
    variant.tileSize = tileSize;
    const bool tile8 = tileSize == kTrainingTileSize8;
    variant.directPipeline.initialize(device_,
                                      tile8 ? "shaders/train_backward_pixel_to_2dgs_tile8.comp.spv"
                                            : "shaders/train_backward_pixel_to_2dgs.comp.spv",
                                      config);
    variant.workgroupPipeline.initialize(
        device_,
        tile8 ? "shaders/train_backward_pixel_to_2dgs_workgroup_tile8.comp.spv"
              : "shaders/train_backward_pixel_to_2dgs_workgroup.comp.spv",
        config);
    if (capabilities_.subgroup) {
        variant.subgroupPipeline.initialize(
            device_,
            tile8 ? "shaders/train_backward_pixel_to_2dgs_subgroup_tile8.comp.spv"
                  : "shaders/train_backward_pixel_to_2dgs_subgroup.comp.spv",
            config);
        variant.adaptivePipeline.initialize(
            device_,
            tile8 ? "shaders/train_backward_pixel_to_2dgs_adaptive_tile8.comp.spv"
                  : "shaders/train_backward_pixel_to_2dgs_adaptive.comp.spv",
            config);
    }
    if (capabilities_.tileGaussian) {
        variant.tileGaussianPipeline.initialize(
            device_,
            tile8 ? "shaders/train_backward_tile_gaussian_atomic_tile8.comp.spv"
                  : "shaders/train_backward_tile_gaussian_atomic.comp.spv",
            config);
    }
    if (capabilities_.vkSplatPerSplat) {
        variant.vkSplatPerSplatPipeline.initialize(
            device_,
            tile8 ? "shaders/train_backward_vksplat_per_splat_tile8.comp.spv"
                  : "shaders/train_backward_vksplat_per_splat.comp.spv",
            config);
    }
    if (capabilities_.vkSplatTensor) {
        variant.vkSplatTensorPipeline.initialize(
            device_,
            tile8 ? "shaders/train_backward_vksplat_tensor_tile8.comp.spv"
                  : "shaders/train_backward_vksplat_tensor.comp.spv",
            config);
    }
}

void PixelTo2DGSDispatcher::cleanupVariant(TilePipelineVariant& variant) {
    variant.vkSplatTensorPipeline.cleanup();
    variant.vkSplatPerSplatPipeline.cleanup();
    variant.tileGaussianPipeline.cleanup();
    variant.adaptivePipeline.cleanup();
    variant.subgroupPipeline.cleanup();
    variant.workgroupPipeline.cleanup();
    variant.directPipeline.cleanup();
    variant.directDescriptorSet = nullptr;
    variant.workgroupDescriptorSet = nullptr;
    variant.subgroupDescriptorSet = nullptr;
    variant.adaptiveDescriptorSet = nullptr;
    variant.tileGaussianDescriptorSet = nullptr;
    variant.vkSplatPerSplatDescriptorSet = nullptr;
    variant.vkSplatTensorDescriptorSet = nullptr;
}

PixelTo2DGSDispatcher::TilePipelineVariant&
PixelTo2DGSDispatcher::variantForTileSize(uint32_t tileSize) {
    return tileSize == kTrainingTileSize8 ? tileVariants_[0] : tileVariants_[1];
}

void PixelTo2DGSDispatcher::initialize(vk::Device device, vk::PhysicalDevice physicalDevice) {
    cleanup();
    device_ = device;
    physicalDevice_ = physicalDevice;
    capabilities_ = queryCapabilities(physicalDevice);

    ComputePipelineConfig config{};
    config.descriptorBindings = {
        backwardStorageBinding(3),  backwardStorageBinding(4),  backwardStorageBinding(5),
        backwardStorageBinding(12), backwardStorageBinding(13), backwardStorageBinding(14),
    };
    config.pushConstantSize = sizeof(TrainingPushConstants);
    ComputePipelineConfig adaptiveSpatialConfig = config;
    adaptiveSpatialConfig.descriptorBindings.push_back(backwardStorageBinding(34));
    adaptiveSpatialConfig.descriptorBindings.push_back(backwardStorageBinding(40));
    adaptiveSpatialConfig.descriptorBindings.push_back(backwardStorageBinding(41));
    initializeVariant(tileVariants_[0], kTrainingTileSize8, config);
    initializeVariant(tileVariants_[1], kTrainingTileSize16, config);
    if (supportsTensorSubgroupReduction(capabilities_)) {
        tensorSubgroupPipeline_.initialize(
            device_, "shaders/train_backward_vksplat_tensor_subgroup.comp.spv", config);
        tensorGlobalPixelGradPipeline_.initialize(
            device_, "shaders/train_backward_vksplat_tensor_subgroup_global_grad.comp.spv", config);
        tensorRgbLatencyPipeline_.initialize(
            device_, "shaders/train_backward_vksplat_tensor_subgroup_global_grad_latency.comp.spv", config);
        // Existing >=45 KiB Tensor gate also covers method 1's 33604 bytes.
        try {
            tensorEarlyCullPipeline_.initialize(
                device_, "shaders/train_backward_vksplat_tensor_early_cull.comp.spv", config);
        } catch (const std::exception& error) {
            tensorEarlyCullPipeline_.cleanup();
            LOG_WARN("EarlyCull experiment unavailable; original Tensor retained: {}", error.what());
        }
    }
    constexpr std::array<uint32_t, 2> adaptiveTileSizes = {kTrainingTileSize8, kTrainingTileSize16};
    for (size_t index = 0; index < adaptiveTileSizes.size(); ++index) {
        const uint32_t tileSize = adaptiveTileSizes[index];
        if (capabilities_.vkSplatPerSplat) {
            adaptiveSpatialPerSplatPipelines_[index].initialize(
                device_,
                "shaders/train_backward_vksplat_per_splat_adaptive_tile" +
                    std::to_string(tileSize) + ".comp.spv",
                adaptiveSpatialConfig);
        }
        if (capabilities_.vkSplatTensor) {
            adaptiveSpatialTensorPipelines_[index].initialize(
                device_,
                "shaders/train_backward_vksplat_tensor_adaptive_tile" + std::to_string(tileSize) +
                    ".comp.spv",
                adaptiveSpatialConfig);
        }
    }

    if (capabilities_.vkSplatTensor) {
        for (size_t index = 0u; index < adaptiveTensorLargeChildPipelines_.size(); ++index) {
            const uint32_t batch = index == 0u ? 16u : 32u;
            adaptiveTensorLargeChildPipelines_[index].initialize(
                device_, "shaders/train_backward_vksplat_tensor_adaptive_tile8_batch" +
                             std::to_string(batch) + ".comp.spv", adaptiveSpatialConfig);
        }
    }

    std::vector<vk::DescriptorSetLayout> layouts;
    for (const TilePipelineVariant& variant : tileVariants_) {
        layouts.push_back(variant.directPipeline.getDescriptorSetLayout());
        layouts.push_back(variant.workgroupPipeline.getDescriptorSetLayout());
        if (capabilities_.subgroup) {
            layouts.push_back(variant.subgroupPipeline.getDescriptorSetLayout());
            layouts.push_back(variant.adaptivePipeline.getDescriptorSetLayout());
        }
        if (capabilities_.tileGaussian) {
            layouts.push_back(variant.tileGaussianPipeline.getDescriptorSetLayout());
        }
        if (capabilities_.vkSplatPerSplat) {
            layouts.push_back(variant.vkSplatPerSplatPipeline.getDescriptorSetLayout());
        }
        if (capabilities_.vkSplatTensor) {
            layouts.push_back(variant.vkSplatTensorPipeline.getDescriptorSetLayout());
        }
    }
    if (supportsTensorSubgroupReduction(capabilities_)) {
        layouts.push_back(tensorSubgroupPipeline_.getDescriptorSetLayout());
        layouts.push_back(tensorGlobalPixelGradPipeline_.getDescriptorSetLayout());
        layouts.push_back(tensorRgbLatencyPipeline_.getDescriptorSetLayout());
        if (tensorEarlyCullSupported())
            layouts.push_back(tensorEarlyCullPipeline_.getDescriptorSetLayout());
    }
    if (capabilities_.vkSplatPerSplat) {
        for (const ComputePipeline& pipeline : adaptiveSpatialPerSplatPipelines_) {
            layouts.push_back(pipeline.getDescriptorSetLayout());
        }
    }
    if (capabilities_.vkSplatTensor) {
        for (const ComputePipeline& pipeline : adaptiveSpatialTensorPipelines_) {
            layouts.push_back(pipeline.getDescriptorSetLayout());
        }
        for (const ComputePipeline& pipeline : adaptiveTensorLargeChildPipelines_) {
            layouts.push_back(pipeline.getDescriptorSetLayout());
        }
    }
    const std::vector<vk::DescriptorSet> sets =
        descriptorPool_.create(device_, static_cast<uint32_t>(layouts.size()) * 9u, 0u, layouts);
    size_t index = 0u;
    for (TilePipelineVariant& variant : tileVariants_) {
        variant.directDescriptorSet = sets[index++];
        variant.workgroupDescriptorSet = sets[index++];
        if (capabilities_.subgroup) {
            variant.subgroupDescriptorSet = sets[index++];
            variant.adaptiveDescriptorSet = sets[index++];
        }
        if (capabilities_.tileGaussian) variant.tileGaussianDescriptorSet = sets[index++];
        if (capabilities_.vkSplatPerSplat) variant.vkSplatPerSplatDescriptorSet = sets[index++];
        if (capabilities_.vkSplatTensor) variant.vkSplatTensorDescriptorSet = sets[index++];
    }
    if (supportsTensorSubgroupReduction(capabilities_)) {
        tensorSubgroupDescriptorSet_ = sets[index++];
        tensorGlobalPixelGradDescriptorSet_ = sets[index++];
        tensorRgbLatencyDescriptorSet_ = sets[index++];
        if (tensorEarlyCullSupported()) tensorEarlyCullDescriptorSet_ = sets[index++];
    }
    if (capabilities_.vkSplatPerSplat) {
        for (vk::DescriptorSet& descriptorSet : adaptiveSpatialPerSplatDescriptorSets_) {
            descriptorSet = sets[index++];
        }
    }
    if (capabilities_.vkSplatTensor) {
        for (vk::DescriptorSet& descriptorSet : adaptiveSpatialTensorDescriptorSets_) {
            descriptorSet = sets[index++];
        }
        for (vk::DescriptorSet& descriptorSet : adaptiveTensorLargeChildDescriptorSets_) {
            descriptorSet = sets[index++];
        }
    }

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
    LOG_INFO("Fixed16 tensor subgroup reduction {} (experimental, opt-in)",
             supportsTensorSubgroupReduction(capabilities_) ? "available" : "unavailable");
}

void PixelTo2DGSDispatcher::prepareCheckpointDiagnostics(TrainingBuffers& buffers, uint32_t itemCount,
                                                         bool warp) {
    if (!supportsTensorSubgroupReduction(capabilities_))
        throw std::runtime_error("Checkpoint P1 diagnostics require the subgroup32 Tensor baseline");
    if (!checkpointState_) {
        auto next = std::make_unique<CheckpointStatePass>();
        next->initialize(device_, physicalDevice_);
        checkpointState_ = std::move(next);
    }
    if (warp && !checkpointBackward_) {
        auto next = std::make_unique<CheckpointBackwardPass>();
        next->initialize(device_, physicalDevice_);
        checkpointBackward_ = std::move(next);
    }
    buffers.ensureCheckpointCapacity(itemCount);
}

void PixelTo2DGSDispatcher::recordCheckpointDiagnostics(const BackwardPassContext& context, bool resetRun) {
    if (!checkpointState_) throw std::runtime_error("Checkpoint P1 diagnostic pass is unavailable");
    checkpointState_->record(context.commandBuffer, context.pushConstants,
                             context.buffers.checkpointResources(), context.profilingQueryPool, resetRun);
}

void PixelTo2DGSDispatcher::cleanup() {
    if (checkpointBackward_) checkpointBackward_->cleanup();
    checkpointBackward_.reset();
    if (checkpointState_) checkpointState_->cleanup();
    checkpointState_.reset();
    physicalDevice_ = nullptr;
    descriptorPool_.reset();
    tensorSubgroupPipeline_.cleanup();
    tensorSubgroupDescriptorSet_ = nullptr;
    tensorGlobalPixelGradPipeline_.cleanup();
    tensorGlobalPixelGradDescriptorSet_ = nullptr;
    tensorRgbLatencyPipeline_.cleanup();
    tensorRgbLatencyDescriptorSet_ = nullptr;
    tensorEarlyCullPipeline_.cleanup();
    tensorEarlyCullDescriptorSet_ = nullptr;
    for (ComputePipeline& pipeline : adaptiveSpatialPerSplatPipelines_) {
        pipeline.cleanup();
    }
    for (ComputePipeline& pipeline : adaptiveSpatialTensorPipelines_) {
        pipeline.cleanup();
    }
    for (ComputePipeline& pipeline : adaptiveTensorLargeChildPipelines_) {
        pipeline.cleanup();
    }
    adaptiveSpatialPerSplatDescriptorSets_.fill(nullptr);
    adaptiveSpatialTensorDescriptorSets_.fill(nullptr);
    adaptiveTensorLargeChildDescriptorSets_.fill(nullptr);
    for (TilePipelineVariant& variant : tileVariants_) {
        cleanupVariant(variant);
    }
    capabilities_ = {};
    device_ = nullptr;
    adaptiveSpatialTiles_ = false;
}

TrainingPixelTo2DGSMode PixelTo2DGSDispatcher::activeMode() const noexcept {
    if (adaptiveSpatialTiles_) {
        return adaptiveSpatialModeSupported() ? requestedMode_ : TrainingPixelTo2DGSMode::Direct;
    }
    return resolvePixelTo2DGSMode(requestedMode_, capabilities_);
}

void PixelTo2DGSDispatcher::record(const BackwardPassContext& context) {
    if (context.tensorEarlyCull &&
        (context.checkpointWarp || !tensorEarlyCullSupported() ||
         context.pushConstants.tileSize != kTrainingTileSize16 || adaptiveSpatialTiles_ ||
         !tensorGlobalPixelGradActive(context.pushConstants.tileSize) ||
         tensorRgbLatencyHidingActive(context.pushConstants.tileSize)))
        throw std::runtime_error("EarlyCull requires the frozen Fixed16 Tensor Subgroup/Global Baseline path");
    if (!context.checkpointWarp) { recordSelected(context); return; }
    if (!checkpointBackward_) throw std::runtime_error("Checkpoint warp prototype is not prepared");
    const auto resources = context.buffers.checkpointResources();
    writeBackwardProfilingTimestamp(context.commandBuffer, context.profilingQueryPool,
                                    TrainingGpuProfileStage::TileLocalBackward, false);
    writeBackwardProfilingTimestamp(context.commandBuffer, context.profilingQueryPool,
                                    TrainingGpuProfileStage::CheckpointWarpBackward, false);
    checkpointBackward_->record(context.commandBuffer, context.pushConstants, resources,
                                 context.buffers.pixelGradsInfo(), context.buffers.projectedGradsInfo());
    vk::MemoryBarrier memory{vk::AccessFlagBits::eShaderWrite,
        vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite |
        vk::AccessFlagBits::eIndirectCommandRead | vk::AccessFlagBits::eTransferRead};
    context.commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eDrawIndirect |
        vk::PipelineStageFlagBits::eTransfer, {}, memory, {}, {});
    recordSelected(context, resources.descriptors[11].buffer,
                    offsetof(TrainingCheckpointControl, fallback));
    writeBackwardProfilingTimestamp(context.commandBuffer, context.profilingQueryPool,
                                    TrainingGpuProfileStage::CheckpointWarpBackward, true);
    writeBackwardProfilingTimestamp(context.commandBuffer, context.profilingQueryPool,
                                    TrainingGpuProfileStage::TileLocalBackward, true);
}

void PixelTo2DGSDispatcher::recordSelected(const BackwardPassContext& context, vk::Buffer indirect,
                                          vk::DeviceSize offset) {
    const TrainingPixelTo2DGSMode mode = activeMode();
    const bool adaptiveSpatial = context.pushConstants.tileSize == 32u && adaptiveSpatialTiles_;
    if (adaptiveSpatial) {
        if (!adaptiveSpatialModeSupported()) {
            throw std::logic_error(
                "Adaptive Spatial reached backward dispatch without optimized support");
        }
        recordAdaptiveSpatial(context, mode);
        return;
    }
    TilePipelineVariant& variant = variantForTileSize(context.pushConstants.tileSize);
    if (mode == TrainingPixelTo2DGSMode::TileGaussianAtomic) {
        recordTileGaussian(context, variant);
        return;
    }
    ComputePipeline* pipeline = &variant.directPipeline;
    vk::DescriptorSet descriptorSet = variant.directDescriptorSet;
    if (mode == TrainingPixelTo2DGSMode::WorkgroupShared) {
        pipeline = &variant.workgroupPipeline;
        descriptorSet = variant.workgroupDescriptorSet;
    } else if (mode == TrainingPixelTo2DGSMode::Subgroup) {
        pipeline = &variant.subgroupPipeline;
        descriptorSet = variant.subgroupDescriptorSet;
    } else if (mode == TrainingPixelTo2DGSMode::Auto) {
        pipeline = &variant.adaptivePipeline;
        descriptorSet = variant.adaptiveDescriptorSet;
    } else if (mode == TrainingPixelTo2DGSMode::VkSplatPerSplat) {
        pipeline = &variant.vkSplatPerSplatPipeline;
        descriptorSet = variant.vkSplatPerSplatDescriptorSet;
    } else if (mode == TrainingPixelTo2DGSMode::VkSplatTensor) {
        const bool useSubgroup = tensorSubgroupReductionActive(context.pushConstants.tileSize);
        pipeline = useSubgroup ? &tensorSubgroupPipeline_ : &variant.vkSplatTensorPipeline;
        descriptorSet = useSubgroup ? tensorSubgroupDescriptorSet_ : variant.vkSplatTensorDescriptorSet;
        if (tensorGlobalPixelGradActive(context.pushConstants.tileSize)) {
            pipeline = &tensorGlobalPixelGradPipeline_;
            descriptorSet = tensorGlobalPixelGradDescriptorSet_;
        }
        if (tensorRgbLatencyHidingActive(context.pushConstants.tileSize)) {
            pipeline = &tensorRgbLatencyPipeline_;
            descriptorSet = tensorRgbLatencyDescriptorSet_;
        }
        if (context.tensorEarlyCull) {
            pipeline = &tensorEarlyCullPipeline_;
            descriptorSet = tensorEarlyCullDescriptorSet_;
        }
    }
    const auto projectedInfo = context.buffers.projectedInfo();
    const auto tileItemsInfo = context.buffers.tileItemsInfo();
    const auto tileRangesInfo = context.buffers.tileRangesInfo();
    const auto pixelGradsInfo = context.buffers.pixelGradsInfo();
    const auto projectedGradsInfo = context.buffers.projectedGradsInfo();
    const auto pixelBlendStatesInfo = context.buffers.pixelBlendStatesInfo();
    const std::array<vk::DescriptorBufferInfo, 6> infos = {
        projectedInfo,  tileItemsInfo,      tileRangesInfo,
        pixelGradsInfo, projectedGradsInfo, pixelBlendStatesInfo};
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

    const bool profileTileLocal = !context.checkpointWarp &&
        (mode == TrainingPixelTo2DGSMode::VkSplatPerSplat || mode == TrainingPixelTo2DGSMode::VkSplatTensor);
    if (profileTileLocal) {
        writeBackwardProfilingTimestamp(context.commandBuffer, context.profilingQueryPool,
                                        TrainingGpuProfileStage::TileLocalBackward, false);
    }
    const uint32_t tileSize = variant.tileSize;
    if (indirect) context.commandBuffer.dispatchIndirect(indirect, offset);
    else context.commandBuffer.dispatch((context.pushConstants.width + tileSize - 1u) / tileSize,
                                         (context.pushConstants.height + tileSize - 1u) / tileSize, 1u);
    if (profileTileLocal) {
        writeBackwardProfilingTimestamp(context.commandBuffer, context.profilingQueryPool,
                                        TrainingGpuProfileStage::TileLocalBackward, true);
    }
    recordBackwardBufferBarrier(context.commandBuffer, {projectedGradsInfo},
                                vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
}

void PixelTo2DGSDispatcher::recordAdaptiveSpatial(const BackwardPassContext& context,
                                                  TrainingPixelTo2DGSMode mode) {
    std::array<ComputePipeline, 2>* pipelines = nullptr;
    std::array<vk::DescriptorSet, 2>* descriptorSets = nullptr;
    if (mode == TrainingPixelTo2DGSMode::VkSplatTensor) {
        pipelines = &adaptiveSpatialTensorPipelines_;
        descriptorSets = &adaptiveSpatialTensorDescriptorSets_;
    } else if (mode == TrainingPixelTo2DGSMode::VkSplatPerSplat) {
        pipelines = &adaptiveSpatialPerSplatPipelines_;
        descriptorSets = &adaptiveSpatialPerSplatDescriptorSets_;
    } else {
        throw std::logic_error("Unsupported optimized adaptive Pixel-to-2DGS mode");
    }

    if (context.pushConstants.width == 0u || context.pushConstants.height == 0u) return;
    writeBackwardProfilingTimestamp(context.commandBuffer, context.profilingQueryPool,
                                    TrainingGpuProfileStage::TileLocalBackward, false);

    // Coarse parents are handled first. Refined parents are skipped by this
    // shader; the following 8x8 dispatch handles only their child leaves.
    writeBackwardProfilingTimestamp(context.commandBuffer, context.profilingQueryPool,
                                    TrainingGpuProfileStage::AdaptiveParentBackward, false);
    recordAdaptiveSpatialDispatch(context, (*pipelines)[1], (*descriptorSets)[1],
                                  kTrainingTileSize16);
    writeBackwardProfilingTimestamp(context.commandBuffer, context.profilingQueryPool,
                                    TrainingGpuProfileStage::AdaptiveParentBackward, true);
    writeBackwardProfilingTimestamp(context.commandBuffer, context.profilingQueryPool,
                                    TrainingGpuProfileStage::AdaptiveChildBackward, false);
    if (mode == TrainingPixelTo2DGSMode::VkSplatTensor && adaptiveTensorChildBatch_ != 8u) {
        const size_t index = adaptiveTensorChildBatch_ == 16u ? 0u : 1u;
        recordAdaptiveSpatialDispatch(context, adaptiveTensorLargeChildPipelines_[index],
                                      adaptiveTensorLargeChildDescriptorSets_[index], kTrainingTileSize8);
    } else {
        recordAdaptiveSpatialDispatch(context, (*pipelines)[0], (*descriptorSets)[0], kTrainingTileSize8);
    }
    writeBackwardProfilingTimestamp(context.commandBuffer, context.profilingQueryPool,
                                    TrainingGpuProfileStage::AdaptiveChildBackward, true);

    writeBackwardProfilingTimestamp(context.commandBuffer, context.profilingQueryPool,
                                    TrainingGpuProfileStage::TileLocalBackward, true);
}

void PixelTo2DGSDispatcher::recordAdaptiveSpatialDispatch(const BackwardPassContext& context,
                                                          ComputePipeline& pipeline,
                                                          vk::DescriptorSet descriptorSet,
                                                          uint32_t tileSize) {
    const auto projectedInfo = context.buffers.projectedInfo();
    const auto tileItemsInfo = context.buffers.tileItemsInfo();
    const auto tileRangesInfo = context.buffers.tileRangesInfo();
    const auto pixelGradsInfo = context.buffers.pixelGradsInfo();
    const auto projectedGradsInfo = context.buffers.projectedGradsInfo();
    const auto pixelBlendStatesInfo = context.buffers.pixelBlendStatesInfo();
    const auto adaptiveSplitFlagsInfo = context.buffers.adaptiveTileSplitFlagsInfo();
    const auto activeParentsInfo = context.buffers.adaptiveActiveParentsInfo();
    const auto childDispatchInfo = context.buffers.adaptiveChildDispatchInfo();
    const std::array<vk::DescriptorBufferInfo, 9> infos = {
        projectedInfo,      tileItemsInfo,        tileRangesInfo,        pixelGradsInfo,
        projectedGradsInfo, pixelBlendStatesInfo, adaptiveSplitFlagsInfo,
        activeParentsInfo, childDispatchInfo};
    constexpr std::array<uint32_t, 9> bindings = {3u, 4u, 5u, 12u, 13u, 14u, 34u, 40u, 41u};
    static_assert(std::tuple_size_v<decltype(infos)> == bindings.size());
    std::array<vk::WriteDescriptorSet, bindings.size()> writes{};
    for (size_t index = 0; index < writes.size(); ++index) {
        writes[index]
            .setDstSet(descriptorSet)
            .setDstBinding(bindings[index])
            .setDescriptorCount(1)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer)
            .setPBufferInfo(&infos[index]);
    }
    device_.updateDescriptorSets(writes, {});
    context.commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.getPipeline());
    context.commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                             pipeline.getPipelineLayout(), 0, descriptorSet, {});
    context.commandBuffer.pushConstants(pipeline.getPipelineLayout(),
                                        vk::ShaderStageFlagBits::eCompute, 0,
                                        sizeof(TrainingPushConstants), &context.pushConstants);
    if (tileSize == kTrainingTileSize8) {
        context.commandBuffer.dispatchIndirect(context.buffers.adaptiveChildDispatchBuffer(), 0u);
    } else {
        context.commandBuffer.dispatch((context.pushConstants.width + tileSize - 1u) / tileSize,
                                       (context.pushConstants.height + tileSize - 1u) / tileSize, 1u);
    }
    recordBackwardBufferBarrier(context.commandBuffer, {projectedGradsInfo},
                                vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
}

void PixelTo2DGSDispatcher::recordTileGaussian(const BackwardPassContext& context,
                                               TilePipelineVariant& variant) {
    if (!variant.tileGaussianDescriptorSet) {
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
            .setDstSet(variant.tileGaussianDescriptorSet)
            .setDstBinding(bindings[index])
            .setDescriptorCount(1)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer)
            .setPBufferInfo(&infos[index]);
    }
    device_.updateDescriptorSets(writes, {});
    context.commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute,
                                       variant.tileGaussianPipeline.getPipeline());
    context.commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                             variant.tileGaussianPipeline.getPipelineLayout(), 0,
                                             variant.tileGaussianDescriptorSet, {});
    context.commandBuffer.pushConstants(variant.tileGaussianPipeline.getPipelineLayout(),
                                        vk::ShaderStageFlagBits::eCompute, 0,
                                        sizeof(TrainingPushConstants), &context.pushConstants);
    if (context.pushConstants.width == 0u || context.pushConstants.height == 0u) return;
    writeBackwardProfilingTimestamp(context.commandBuffer, context.profilingQueryPool,
                                    TrainingGpuProfileStage::TileLocalBackward, false);
    context.commandBuffer.dispatch(
        (context.pushConstants.width + variant.tileSize - 1u) / variant.tileSize,
        (context.pushConstants.height + variant.tileSize - 1u) / variant.tileSize, 1u);
    writeBackwardProfilingTimestamp(context.commandBuffer, context.profilingQueryPool,
                                    TrainingGpuProfileStage::TileLocalBackward, true);
    recordBackwardBufferBarrier(context.commandBuffer, {projectedGradsInfo},
                                vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
}

} // namespace vulkan3DGS
