#include "training/core/backward/checkpoint_backward_pass.hpp"
#include <stdexcept>

namespace vulkan3DGS {
void CheckpointBackwardPass::initialize(vk::Device device, vk::PhysicalDevice physical,
                                        const std::string& shaderDirectory,
                                        uint32_t requiredSubgroupSize, bool fullSubgroups) {
    cleanup();
    vk::PhysicalDeviceSubgroupProperties subgroup{};
    vk::PhysicalDeviceProperties2 properties{};
    properties.pNext = &subgroup;
    physical.getProperties2(&properties);
    const auto& limits = properties.properties.limits;
    const auto operations =
        vk::SubgroupFeatureFlagBits::eBasic | vk::SubgroupFeatureFlagBits::eShuffle;
    if ((requiredSubgroupSize == 0u ? subgroup.subgroupSize : requiredSubgroupSize) != 32u ||
        !(subgroup.supportedStages & vk::ShaderStageFlagBits::eCompute) ||
        (subgroup.supportedOperations & operations) != operations ||
        limits.maxComputeWorkGroupInvocations < 128u || limits.maxComputeWorkGroupSize[0] < 128u ||
        limits.maxPerStageDescriptorStorageBuffers < 12u ||
        limits.maxDescriptorSetStorageBuffers < 12u)
        throw std::runtime_error("Checkpoint warp backward requires supported compute subgroup32");
    device_ = device;
    ComputePipelineConfig config{};
    for (uint32_t binding : {3u, 4u, 12u, 13u, 14u, 43u, 45u, 46u, 47u, 48u, 49u, 50u})
        config.descriptorBindings.push_back(backwardStorageBinding(binding));
    config.pushConstantSize = sizeof(TrainingPushConstants);
    config.requiredSubgroupSize = requiredSubgroupSize;
    config.requireFullSubgroups = fullSubgroups;
    pipeline_.initialize(device_, shaderDirectory + "/train_backward_checkpoint_warp.comp.spv",
                         config);
    set_ = pool_.create(device_, 12u, 0u, {pipeline_.getDescriptorSetLayout()})[0];
}

void CheckpointBackwardPass::cleanup() {
    pool_.reset();
    pipeline_.cleanup();
    set_ = nullptr;
    device_ = nullptr;
}

void CheckpointBackwardPass::record(vk::CommandBuffer command, const TrainingPushConstants& params,
                                    const TrainingCheckpointResources& resources,
                                    vk::DescriptorBufferInfo pixelGrads,
                                    vk::DescriptorBufferInfo projectedGrads) {
    if (!device_) throw std::runtime_error("Checkpoint warp backward is not initialized");
    constexpr std::array<uint32_t, 12> bindings = {3, 4, 12, 13, 14, 43, 45, 46, 47, 48, 49, 50};
    const auto& r = resources.descriptors;
    const std::array<vk::DescriptorBufferInfo, 12> infos = {
        r[0], r[1], pixelGrads, projectedGrads, r[3], r[4], r[6], r[7], r[8], r[9], r[10], r[11]};
    std::array<vk::WriteDescriptorSet, 12> writes{};
    for (size_t i = 0; i < writes.size(); ++i)
        writes[i]
            .setDstSet(set_)
            .setDstBinding(bindings[i])
            .setDescriptorCount(1)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer)
            .setPBufferInfo(&infos[i]);
    device_.updateDescriptorSets(writes, {});
    command.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline_.getPipeline());
    command.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipeline_.getPipelineLayout(), 0,
                               set_, {});
    command.pushConstants(pipeline_.getPipelineLayout(), vk::ShaderStageFlagBits::eCompute, 0,
                          sizeof(params), &params);
    command.dispatchIndirect(r[11].buffer, offsetof(TrainingCheckpointControl, dispatch));
}
} // namespace vulkan3DGS
