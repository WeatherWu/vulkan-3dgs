#pragma once

#include "training/core/backward/backward_command_utils.hpp"
#include "training_pixel_culling_reference.hpp"
#include "vulkan/buffer.hpp"
#include "vulkan/command_pool.hpp"
#include "vulkan/compute_pipeline.hpp"

#include <fstream>
#include <iostream>
#include <stdexcept>

inline void verifyPixelCullingShader(const std::string& name) {
    const auto path = std::string(CHECKPOINT_RUNTIME_DIR) + "/shaders/" + name + ".comp.spv";
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() < 20) throw std::runtime_error("Missing EarlyCull SPIR-V: " + path);
    const auto bytes = static_cast<size_t>(file.tellg());
    if (bytes % 4u != 0u) throw std::runtime_error("Invalid EarlyCull SPIR-V length");
    std::vector<uint32_t> words(bytes / 4u);
    file.seekg(0);
    file.read(reinterpret_cast<char*>(words.data()), static_cast<std::streamsize>(bytes));
    if (!file || words[0] != 0x07230203u) throw std::runtime_error("Invalid EarlyCull SPIR-V");
    bool exp = false, log = false;
    for (size_t offset = 5u; offset < words.size();) {
        const uint32_t count = words[offset] >> 16u, op = words[offset] & 0xffffu;
        if (count == 0u || count > words.size() - offset)
            throw std::runtime_error("Invalid EarlyCull SPIR-V instruction");
        if ((op == 71u && count >= 3u && words[offset + 2u] == 0u) ||
            (op == 72u && count >= 4u && words[offset + 3u] == 0u))
            throw std::runtime_error("EarlyCull safety bound forbids RelaxedPrecision");
        if (op == 22u && count >= 3u && words[offset + 2u] != 32u)
            throw std::runtime_error("EarlyCull safety bound requires FP32");
        if (op == 12u && count >= 5u) {
            exp |= words[offset + 4u] == 27u;
            log |= words[offset + 4u] == 28u;
        }
        offset += count;
    }
    if (!exp || !log)
        throw std::runtime_error("EarlyCull SPIR-V lowering changed; re-audit error bound");
}

// Included by the explicit GPU test executable, sharing its device/validation
// setup. Does not require a Tensor-capable shared-memory size or subgroup32.
inline void runPixelCullingProbe(vk::Device device, vk::PhysicalDevice physical, vk::Queue queue,
                                 uint32_t family) {
    namespace ref = pixel_culling_reference;
    verifyPixelCullingShader("train_pixel_culling_probe");
    verifyPixelCullingShader("train_backward_vksplat_tensor_early_cull");
    vk::PhysicalDeviceSubgroupProperties subgroup{};
    vk::PhysicalDeviceProperties2 properties{};
    properties.pNext = &subgroup;
    physical.getProperties2(&properties);
    if (!(subgroup.supportedStages & vk::ShaderStageFlagBits::eCompute) ||
        !(subgroup.supportedOperations & vk::SubgroupFeatureFlagBits::eVote))
        throw std::runtime_error("S3 synthetic probe requires compute subgroup VOTE");
    const auto inputs = ref::cases();
    std::vector<std::array<float, 4>> outputs(inputs.size());
    std::vector<std::array<uint32_t, 4>> decisions(inputs.size());
    std::array<vulkan3DGS::Buffer, 3> buffers;
    for (size_t i = 0; i < buffers.size(); ++i)
        buffers[i].create(
            device, physical, queue, family, i == 0 ? inputs.data() : nullptr, inputs.size() * 16u,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc |
                vk::BufferUsageFlagBits::eTransferDst,
            vk::MemoryPropertyFlagBits::eDeviceLocal, {family});
    vulkan3DGS::ComputePipeline pipeline;
    vulkan3DGS::ComputePipelineConfig config;
    for (uint32_t binding = 0; binding < 3; ++binding)
        config.descriptorBindings.push_back(vulkan3DGS::backwardStorageBinding(binding));
    config.pushConstantSize = sizeof(vulkan3DGS::TrainingPushConstants);
    pipeline.initialize(
        device, std::string(CHECKPOINT_RUNTIME_DIR) + "/shaders/train_pixel_culling_probe.comp.spv",
        config);
    vulkan3DGS::BackwardDescriptorPool descriptors;
    const auto set = descriptors.create(device, 3, 0, {pipeline.getDescriptorSetLayout()})[0];
    std::array<vk::DescriptorBufferInfo, 3> infos;
    std::array<vk::WriteDescriptorSet, 3> writes;
    for (uint32_t i = 0; i < 3; ++i) {
        infos[i] = buffers[i].getDescriptorInfo();
        writes[i]
            .setDstSet(set)
            .setDstBinding(i)
            .setDescriptorCount(1)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer)
            .setPBufferInfo(&infos[i]);
    }
    device.updateDescriptorSets(writes, {});
    vulkan3DGS::CommandPool pool;
    pool.create(device, family);
    const auto command = pool.allocateCommandBuffer();
    command.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    vk::MemoryBarrier uploaded{vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead};
    command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                            vk::PipelineStageFlagBits::eComputeShader, {}, uploaded, {}, {});
    command.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.getPipeline());
    command.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipeline.getPipelineLayout(), 0,
                               set, {});
    vulkan3DGS::TrainingPushConstants params{};
    params.pixelCount = static_cast<uint32_t>(inputs.size());
    command.pushConstants(pipeline.getPipelineLayout(), vk::ShaderStageFlagBits::eCompute, 0,
                          sizeof(params), &params);
    command.dispatch((params.pixelCount + 127u) / 128u, 1, 1);
    vk::MemoryBarrier completed{vk::AccessFlagBits::eShaderWrite,
                                vk::AccessFlagBits::eTransferRead};
    command.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                            vk::PipelineStageFlagBits::eTransfer, {}, completed, {}, {});
    command.end();
    vk::SubmitInfo submit;
    submit.setCommandBuffers(command);
    queue.submit(submit);
    queue.waitIdle();
    buffers[1].download(outputs.data(), outputs.size() * 16u);
    buffers[2].download(decisions.data(), decisions.size() * 16u);
    uint64_t eligible = 0, exp = 0, rejected = 0, skippedWaves = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        const auto& d = decisions[i];
        eligible += d[0];
        exp += d[1];
        rejected += d[2];
        skippedWaves += outputs[i][1] == 1.0f ? 1u : 0u;
        if (d[2] && d[3]) throw std::runtime_error("GPU EarlyCull rejected an old contributor");
        const float cutoff = ref::threshold(inputs[i][0]);
        if (cutoff == ref::disabled && outputs[i][0] != ref::disabled)
            throw std::runtime_error("GPU EarlyCull fallback domain mismatch");
        if (cutoff != ref::disabled &&
            !(double(outputs[i][0]) <
              std::log(double(ref::alphaMin) / double(inputs[i][0])) - double(ref::guard) * 0.9))
            throw std::runtime_error(
                "GPU threshold did not preserve its conservative boundary band");
        if (std::isfinite(inputs[i][0]) && std::isfinite(outputs[i][2]) && d[3] &&
            outputs[i][2] != outputs[i][3])
            throw std::runtime_error("GPU EarlyCull changed retained alpha");
        if (d[2] && !(inputs[i][1] >= -80.0f && inputs[i][1] < outputs[i][0]))
            throw std::runtime_error("GPU EarlyCull classification/threshold mismatch");
    }
    if (rejected == 0u) throw std::runtime_error("GPU EarlyCull did not exercise rejection");
    pool.freeCommandBuffer(command);
    std::cout << "S3-0 GPU probe: " << inputs.size() << " cases; eligible=" << eligible
              << " old-exp=" << exp << " rejected=" << rejected
              << " fully-skipped-native-waves=" << skippedWaves
              << "; false rejections=0 (synthetic, not target-scene speedup)\n";
}
