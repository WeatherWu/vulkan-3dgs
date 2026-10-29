#include "training/core/backward/projection_optimizer_pass.hpp"

#include "utils/logger.hpp"
#include "utils/file_utils.hpp"
#include "training/core/training_optimizer_capture_gpu.hpp"

#include <array>
#include <cstdlib>
#include <cstring>

namespace vulkan3DGS {

namespace {

bool forceSeparateProjectionOptimizer() {
#ifdef _WIN32
    char* value = nullptr;
    size_t length = 0;
    if (_dupenv_s(&value, &length, "VULKAN_3DGS_SEPARATE_PROJECTION_OPTIMIZER") != 0 || !value) {
        return false;
    }
    const bool enabled = std::strcmp(value, "1") == 0;
    std::free(value);
    return enabled;
#else
    const char* value = std::getenv("VULKAN_3DGS_SEPARATE_PROJECTION_OPTIMIZER");
    return value && std::strcmp(value, "1") == 0;
#endif
}

} // namespace

void ProjectionOptimizerPass::initialize(vk::Device device, const std::filesystem::path& shaderDirectory) {
    cleanup();
    device_ = device;
    fusedEnabled_ = !forceSeparateProjectionOptimizer();

    ComputePipelineConfig projectionConfig{};
    projectionConfig.descriptorBindings = {
        backwardStorageBinding(0),  backwardStorageBinding(1),  backwardStorageBinding(3),
        backwardStorageBinding(13), backwardStorageBinding(16), backwardStorageBinding(32),
        backwardUniformBinding(11),
    };
    projectionConfig.pushConstantSize = sizeof(TrainingPushConstants);
    ComputePipelineConfig fusedConfig{};
    fusedConfig.descriptorBindings = {
        backwardStorageBinding(0),  backwardStorageBinding(2),  backwardStorageBinding(3),
        backwardStorageBinding(13), backwardStorageBinding(16), backwardStorageBinding(32),
        backwardUniformBinding(11),
    };
    fusedConfig.pushConstantSize = sizeof(TrainingPushConstants);
    ComputePipelineConfig optimizerConfig{};
    optimizerConfig.descriptorBindings = {
        backwardStorageBinding(0),
        backwardStorageBinding(1),
        backwardStorageBinding(2),
    };
    optimizerConfig.pushConstantSize = sizeof(TrainingPushConstants);

    const auto shaderPath = [&](const char* name) {
        return FileUtils::resolvePath((shaderDirectory / name).string());
    };
    projectionPipeline_.initialize(device_, shaderPath("train_backward_2dgs_to_3dgs.comp.spv"),
                                   projectionConfig);
    fusedPipeline_.initialize(device_, shaderPath("train_backward_2dgs_to_3dgs_optimizer.comp.spv"),
                              fusedConfig);
    fusedShaderHash_ = optimizerShaderHash(shaderPath("train_backward_2dgs_to_3dgs_optimizer.comp.spv"));
    optimizerPipeline_.initialize(device_, shaderPath("train_optimizer.comp.spv"), optimizerConfig);
    auto directProjectionConfig = projectionConfig;
    auto directFusedConfig = fusedConfig;
    directProjectionConfig.descriptorBindings.push_back(backwardUniformBinding(64));
    directFusedConfig.descriptorBindings.push_back(backwardUniformBinding(64));
    directPipelines_[0].initialize(
        device_, shaderPath("train_backward_2dgs_to_3dgs_direct_intrinsics.comp.spv"),
        directProjectionConfig);
    directPipelines_[1].initialize(
        device_, shaderPath("train_backward_2dgs_to_3dgs_optimizer_direct_intrinsics.comp.spv"),
        directFusedConfig);
    const std::vector<vk::DescriptorSet> sets = descriptorPool_.create(
        device_, 27u, 6u,
        {projectionPipeline_.getDescriptorSetLayout(), fusedPipeline_.getDescriptorSetLayout(),
         optimizerPipeline_.getDescriptorSetLayout(), directPipelines_[0].getDescriptorSetLayout(),
         directPipelines_[1].getDescriptorSetLayout()});
    projectionDescriptorSet_ = sets[0];
    fusedDescriptorSet_ = sets[1];
    optimizerDescriptorSet_ = sets[2];
    directSets_ = {sets[3], sets[4]};

    if (fusedEnabled_) {
        LOG_INFO("Fused projection backward and optimizer enabled");
    } else {
        LOG_WARN("Using separate projection backward and optimizer because "
                 "VULKAN_3DGS_SEPARATE_PROJECTION_OPTIMIZER=1");
    }
}

void ProjectionOptimizerPass::cleanup() {
    descriptorPool_.reset();
    directSets_ = {};
    for (auto& pipeline : directPipelines_)
        pipeline.cleanup();
    projectionDescriptorSet_ = nullptr;
    fusedDescriptorSet_ = nullptr;
    optimizerDescriptorSet_ = nullptr;
    optimizerPipeline_.cleanup();
    fusedPipeline_.cleanup();
    projectionPipeline_.cleanup();
    fusedEnabled_ = true;
    device_ = nullptr;
}

void ProjectionOptimizerPass::recordProjection(const BackwardPassContext& context) {
    if (context.directIntrinsics) {
        recordDirect(context, false);
        return;
    }
    const auto gaussianParamsInfo = context.buffers.gaussianParamsInfo();
    const auto gaussianGradsInfo = context.buffers.gaussianGradsInfo();
    const auto projectedInfo = context.buffers.projectedInfo();
    const auto projectedGradsInfo = context.buffers.projectedGradsInfo();
    const auto densificationStatesInfo = context.buffers.densificationStatesInfo();
    const auto gaussianTileRangesInfo = context.buffers.gaussianTileRangesInfo();
    const auto cameraInfo = context.buffers.cameraInfo();
    std::array<vk::DescriptorBufferInfo, 7> infos = {
        gaussianParamsInfo,      gaussianGradsInfo,      projectedInfo, projectedGradsInfo,
        densificationStatesInfo, gaussianTileRangesInfo, cameraInfo};
    constexpr std::array<uint32_t, 7> bindings = {0u, 1u, 3u, 13u, 16u, 32u, 11u};
    constexpr std::array<vk::DescriptorType, 7> types = {
        vk::DescriptorType::eStorageBuffer, vk::DescriptorType::eStorageBuffer,
        vk::DescriptorType::eStorageBuffer, vk::DescriptorType::eStorageBuffer,
        vk::DescriptorType::eStorageBuffer, vk::DescriptorType::eStorageBuffer,
        vk::DescriptorType::eUniformBuffer};
    std::array<vk::WriteDescriptorSet, 7> writes{};
    for (size_t index = 0; index < writes.size(); ++index) {
        writes[index]
            .setDstSet(projectionDescriptorSet_)
            .setDstBinding(bindings[index])
            .setDescriptorCount(1)
            .setDescriptorType(types[index])
            .setPBufferInfo(&infos[index]);
    }
    device_.updateDescriptorSets(writes, {});
    context.commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute,
                                       projectionPipeline_.getPipeline());
    context.commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                             projectionPipeline_.getPipelineLayout(), 0,
                                             projectionDescriptorSet_, {});
    context.commandBuffer.pushConstants(projectionPipeline_.getPipelineLayout(),
                                        vk::ShaderStageFlagBits::eCompute, 0,
                                        sizeof(TrainingPushConstants), &context.pushConstants);
    if (context.pushConstants.gaussianCount == 0u) return;
    context.commandBuffer.dispatch((context.pushConstants.gaussianCount + 255u) / 256u, 1u, 1u);
    recordBackwardBufferBarrier(context.commandBuffer, {gaussianGradsInfo, densificationStatesInfo},
                                vk::AccessFlagBits::eShaderRead |
                                    vk::AccessFlagBits::eTransferRead);
}

void ProjectionOptimizerPass::recordProjectionAndOptimize(const BackwardPassContext& context) {
    if (context.directIntrinsics) {
        recordDirect(context, true);
        return;
    }
    const auto gaussianParamsInfo = context.buffers.gaussianParamsInfo();
    const auto adamStatesInfo = context.buffers.adamStatesInfo();
    const auto projectedInfo = context.buffers.projectedInfo();
    const auto projectedGradsInfo = context.buffers.projectedGradsInfo();
    const auto densificationStatesInfo = context.buffers.densificationStatesInfo();
    const auto gaussianTileRangesInfo = context.buffers.gaussianTileRangesInfo();
    const auto cameraInfo = context.buffers.cameraInfo();
    std::array<vk::DescriptorBufferInfo, 7> infos = {
        gaussianParamsInfo,      adamStatesInfo,         projectedInfo, projectedGradsInfo,
        densificationStatesInfo, gaussianTileRangesInfo, cameraInfo};
    if (context.optimizerCapture)
        context.optimizerCapture->recordPre(context.commandBuffer, infos, context.pushConstants,
                                             context.buffers.uploadedCamera(), fusedShaderHash_);
    constexpr std::array<uint32_t, 7> bindings = {0u, 2u, 3u, 13u, 16u, 32u, 11u};
    constexpr std::array<vk::DescriptorType, 7> types = {
        vk::DescriptorType::eStorageBuffer, vk::DescriptorType::eStorageBuffer,
        vk::DescriptorType::eStorageBuffer, vk::DescriptorType::eStorageBuffer,
        vk::DescriptorType::eStorageBuffer, vk::DescriptorType::eStorageBuffer,
        vk::DescriptorType::eUniformBuffer};
    std::array<vk::WriteDescriptorSet, 7> writes{};
    for (size_t index = 0; index < writes.size(); ++index) {
        writes[index]
            .setDstSet(fusedDescriptorSet_)
            .setDstBinding(bindings[index])
            .setDescriptorCount(1)
            .setDescriptorType(types[index])
            .setPBufferInfo(&infos[index]);
    }
    device_.updateDescriptorSets(writes, {});
    context.commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute,
                                       fusedPipeline_.getPipeline());
    context.commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                             fusedPipeline_.getPipelineLayout(), 0,
                                             fusedDescriptorSet_, {});
    context.commandBuffer.pushConstants(fusedPipeline_.getPipelineLayout(),
                                        vk::ShaderStageFlagBits::eCompute, 0,
                                        sizeof(TrainingPushConstants), &context.pushConstants);
    if (context.pushConstants.gaussianCount == 0u) return;
    context.commandBuffer.dispatch((context.pushConstants.gaussianCount + 255u) / 256u, 1u, 1u);
    recordBackwardBufferBarrier(context.commandBuffer,
                                {gaussianParamsInfo, adamStatesInfo, densificationStatesInfo},
                                vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite |
                                    vk::AccessFlagBits::eTransferRead);
}

void ProjectionOptimizerPass::recordDirect(const BackwardPassContext& context, bool fused) {
    if (!context.buffers.hasExactIntrinsics() || context.optimizerCapture)
        throw std::runtime_error("Direct projection requires exact intrinsics and cannot write "
                                 "legacy optimizer captures");
    auto& pipeline = directPipelines_[fused ? 1 : 0];
    const auto set = directSets_[fused ? 1 : 0];
    const std::array<vk::DescriptorBufferInfo, 8> infos{context.buffers.gaussianParamsInfo(),
                                                        fused ? context.buffers.adamStatesInfo()
                                                              : context.buffers.gaussianGradsInfo(),
                                                        context.buffers.projectedInfo(),
                                                        context.buffers.projectedGradsInfo(),
                                                        context.buffers.densificationStatesInfo(),
                                                        context.buffers.gaussianTileRangesInfo(),
                                                        context.buffers.cameraInfo(),
                                                        context.buffers.cameraIntrinsicsInfo()};
    const std::array<uint32_t, 8> bindings{0, fused ? 2u : 1u, 3, 13, 16, 32, 11, 64};
    std::array<vk::WriteDescriptorSet, 8> writes;
    for (size_t i = 0; i < writes.size(); ++i)
        writes[i]
            .setDstSet(set)
            .setDstBinding(bindings[i])
            .setDescriptorCount(1)
            .setDescriptorType(i >= 6 ? vk::DescriptorType::eUniformBuffer
                                      : vk::DescriptorType::eStorageBuffer)
            .setPBufferInfo(&infos[i]);
    device_.updateDescriptorSets(writes, {});
    context.commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.getPipeline());
    context.commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                             pipeline.getPipelineLayout(), 0, set, {});
    context.commandBuffer.pushConstants(pipeline.getPipelineLayout(),
                                        vk::ShaderStageFlagBits::eCompute, 0,
                                        sizeof(TrainingPushConstants), &context.pushConstants);
    if (!context.pushConstants.gaussianCount) return;
    context.commandBuffer.dispatch((context.pushConstants.gaussianCount + 255u) / 256u, 1, 1);
    recordBackwardBufferBarrier(context.commandBuffer, {infos[0], infos[1], infos[4]},
                                vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite |
                                    vk::AccessFlagBits::eTransferRead);
}

void ProjectionOptimizerPass::recordOptimizer(const BackwardPassContext& context) {
    const auto gaussianParamsInfo = context.buffers.gaussianParamsInfo();
    const auto gaussianGradsInfo = context.buffers.gaussianGradsInfo();
    const auto adamStatesInfo = context.buffers.adamStatesInfo();
    std::array<vk::DescriptorBufferInfo, 3> infos = {gaussianParamsInfo, gaussianGradsInfo,
                                                     adamStatesInfo};
    constexpr std::array<uint32_t, 3> bindings = {0u, 1u, 2u};
    std::array<vk::WriteDescriptorSet, 3> writes{};
    for (size_t index = 0; index < writes.size(); ++index) {
        writes[index]
            .setDstSet(optimizerDescriptorSet_)
            .setDstBinding(bindings[index])
            .setDescriptorCount(1)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer)
            .setPBufferInfo(&infos[index]);
    }
    device_.updateDescriptorSets(writes, {});
    context.commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute,
                                       optimizerPipeline_.getPipeline());
    context.commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                             optimizerPipeline_.getPipelineLayout(), 0,
                                             optimizerDescriptorSet_, {});
    context.commandBuffer.pushConstants(optimizerPipeline_.getPipelineLayout(),
                                        vk::ShaderStageFlagBits::eCompute, 0,
                                        sizeof(TrainingPushConstants), &context.pushConstants);
    if (context.pushConstants.gaussianCount == 0u) return;
    context.commandBuffer.dispatch((context.pushConstants.gaussianCount + 255u) / 256u, 1u, 1u);
    recordBackwardBufferBarrier(context.commandBuffer, {gaussianParamsInfo, adamStatesInfo},
                                vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite |
                                    vk::AccessFlagBits::eTransferRead);
}

} // namespace vulkan3DGS
