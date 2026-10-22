#include "training/core/backward/checkpoint_state_pass.hpp"

#include "utils/logger.hpp"
#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <vector>

namespace vulkan3DGS {

void CheckpointStatePass::initialize(vk::Device device, vk::PhysicalDevice physicalDevice,
                                     const std::string& shaderDirectory) {
    const auto setupStart = std::chrono::steady_clock::now();
    cleanup();
    device_ = device;
    limits_ = physicalDevice.getProperties().limits;
    if (limits_.maxPerStageDescriptorStorageBuffers < kCheckpointBindings.size() ||
        limits_.maxDescriptorSetStorageBuffers < kCheckpointBindings.size() ||
        limits_.maxComputeWorkGroupInvocations < 256u ||
        limits_.maxComputeWorkGroupSize[0] < 256u || limits_.maxComputeWorkGroupSize[1] < 16u ||
        limits_.maxComputeSharedMemorySize < 14u * 1024u + 4u)
        throw std::runtime_error("Checkpoint diagnostic device limits are insufficient");
    ComputePipelineConfig config{};
    for (uint32_t binding : kCheckpointBindings)
        config.descriptorBindings.push_back(backwardStorageBinding(binding));
    constexpr std::array<const char*, 4> paths = {
        "train_checkpoint_layout.comp.spv", "train_checkpoint_prefix.comp.spv",
        "train_checkpoint_state.comp.spv", "train_checkpoint_tasks.comp.spv"};
    std::vector<vk::DescriptorSetLayout> layouts;
    for (size_t i = 0; i < pipelines_.size(); ++i) {
        config.pushConstantSize =
            i == 1u ? sizeof(TrainingCheckpointPrefixPush) : sizeof(TrainingPushConstants);
        pipelines_[i].initialize(device_, shaderDirectory + "/" + paths[i], config);
        layouts.push_back(pipelines_[i].getDescriptorSetLayout());
    }
    const auto sets = pool_.create(device_, 48u, 0u, layouts);
    std::copy(sets.begin(), sets.end(), sets_.begin());
    LOG_INFO("Checkpoint P1 pipeline/descriptor setup: {} ms CPU",
             std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - setupStart)
                 .count());
}

void CheckpointStatePass::cleanup() {
    pool_.reset();
    for (auto& pipeline : pipelines_)
        pipeline.cleanup();
    sets_ = {};
    device_ = nullptr;
    lastControl_ = nullptr;
}

void CheckpointStatePass::record(vk::CommandBuffer cmd, const TrainingPushConstants& params,
                                 const TrainingCheckpointResources& resources,
                                 vk::QueryPool queries, bool resetRun) {
    if (!device_) throw std::runtime_error("Checkpoint diagnostic pipelines are not initialized");
    const uint32_t tilesX = (params.width + 15u) / 16u;
    const uint32_t tilesY = (params.height + 15u) / 16u;
    const uint32_t tileCount = static_cast<uint32_t>(resources.capacity.tileCount);
    if (params.tileSize != 16u || tileCount != uint64_t(tilesX) * tilesY || tileCount == 0u ||
        tilesX > limits_.maxComputeWorkGroupCount[0] ||
        tilesY > limits_.maxComputeWorkGroupCount[1] ||
        (uint64_t(tileCount) + 255u) / 256u > limits_.maxComputeWorkGroupCount[0])
        throw std::runtime_error("Checkpoint diagnostic grid is unsupported");
    for (size_t i = 0; i < pipelines_.size(); ++i) {
        std::array<vk::WriteDescriptorSet, kCheckpointBindings.size()> writes{};
        for (size_t j = 0; j < writes.size(); ++j)
            writes[j]
                .setDstSet(sets_[i])
                .setDstBinding(kCheckpointBindings[j])
                .setDescriptorType(vk::DescriptorType::eStorageBuffer)
                .setDescriptorCount(1)
                .setPBufferInfo(&resources.descriptors[j]);
        device_.updateDescriptorSets(writes, {});
    }
    auto barrier = [&](vk::PipelineStageFlags source, vk::AccessFlags srcAccess,
                       vk::PipelineStageFlags destination, vk::AccessFlags dstAccess) {
        vk::MemoryBarrier memory{srcAccess, dstAccess};
        cmd.pipelineBarrier(source, destination, {}, memory, {}, {});
    };
    writeBackwardProfilingTimestamp(cmd, queries, TrainingGpuProfileStage::CheckpointLayout, false);
    // This also covers previous-step reads before updateBuffer overwrites control.
    barrier(vk::PipelineStageFlagBits::eAllCommands,
            vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
            vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferWrite);
    TrainingCheckpointControl control{};
    control.capacity = {static_cast<uint32_t>(resources.capacity.slotCapacity),
                        static_cast<uint32_t>(resources.capacity.taskCapacity), tileCount, 2u};
    control.fallback = {tilesX, tilesY, 1u, 0u};
    const bool reset = resetRun || lastControl_ != resources.descriptors[11].buffer;
    cmd.updateBuffer(resources.descriptors[11].buffer, 0,
                     reset ? sizeof(control) : offsetof(TrainingCheckpointControl, run), &control);
    lastControl_ = resources.descriptors[11].buffer;
    barrier(vk::PipelineStageFlagBits::eTransfer | vk::PipelineStageFlagBits::eComputeShader,
            vk::AccessFlagBits::eTransferWrite | vk::AccessFlagBits::eShaderWrite,
            vk::PipelineStageFlagBits::eComputeShader,
            vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
    auto computeBarrier = [&] {
        barrier(vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eShaderWrite,
                vk::PipelineStageFlagBits::eComputeShader,
                vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
    };
    auto bind = [&](size_t i) {
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipelines_[i].getPipeline());
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelines_[i].getPipelineLayout(),
                               0, sets_[i], {});
    };
    auto fullDispatch = [&](size_t i) {
        bind(i);
        cmd.pushConstants(pipelines_[i].getPipelineLayout(), vk::ShaderStageFlagBits::eCompute, 0,
                          sizeof(params), &params);
        cmd.dispatch(tilesX, tilesY, 1);
        computeBarrier();
    };
    fullDispatch(0);
    bind(1);
    auto prefixDispatch = [&](TrainingCheckpointPrefixPush push) {
        push.maxGroupsX = limits_.maxComputeWorkGroupCount[0];
        push.maxGroupsY = limits_.maxComputeWorkGroupCount[1];
        cmd.pushConstants(pipelines_[1].getPipelineLayout(), vk::ShaderStageFlagBits::eCompute, 0,
                          sizeof(push), &push);
        cmd.dispatch((push.count + 255u) / 256u, 1, 1);
        computeBarrier();
    };
    struct Level {
        uint32_t count, offset;
    };
    std::vector<Level> levels{{(tileCount + 255u) / 256u, 0u}};
    while (levels.back().count > 256u) {
        const auto last = levels.back();
        levels.push_back({(last.count + 255u) / 256u, last.offset + last.count});
    }
    prefixDispatch({0u, tileCount, 0u, 0u});
    for (size_t i = 0; i + 1u < levels.size(); ++i)
        prefixDispatch({1u, levels[i].count, levels[i].offset, levels[i + 1u].offset});
    prefixDispatch({2u, levels.back().count, levels.back().offset});
    for (size_t i = levels.size() - 1u; i > 0u; --i)
        prefixDispatch({3u, levels[i - 1u].count, levels[i - 1u].offset, 0u, levels[i].offset});
    prefixDispatch({4u, tileCount, 0u, 0u, 0u});
    writeBackwardProfilingTimestamp(cmd, queries, TrainingGpuProfileStage::CheckpointLayout, true);
    writeBackwardProfilingTimestamp(cmd, queries, TrainingGpuProfileStage::CheckpointStateBuild,
                                    false);
    fullDispatch(2);
    writeBackwardProfilingTimestamp(cmd, queries, TrainingGpuProfileStage::CheckpointStateBuild,
                                    true);
    writeBackwardProfilingTimestamp(cmd, queries, TrainingGpuProfileStage::CheckpointTasks, false);
    fullDispatch(3);
    bind(1);
    prefixDispatch({5u, 1u});
    barrier(vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eShaderWrite,
            vk::PipelineStageFlagBits::eDrawIndirect | vk::PipelineStageFlagBits::eComputeShader |
                vk::PipelineStageFlagBits::eTransfer,
            vk::AccessFlagBits::eIndirectCommandRead | vk::AccessFlagBits::eShaderRead |
                vk::AccessFlagBits::eTransferRead);
    writeBackwardProfilingTimestamp(cmd, queries, TrainingGpuProfileStage::CheckpointTasks, true);
}

} // namespace vulkan3DGS
