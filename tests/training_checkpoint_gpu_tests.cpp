#include "training/core/backward/checkpoint_backward_pass.hpp"
#include "training/core/backward/checkpoint_state_pass.hpp"
#include "training/core/training_buffers.hpp"
#include "training/core/training_step_executor.hpp"
#include "training_checkpoint_reference.hpp"
#include "vulkan/buffer.hpp"
#include "vulkan/command_pool.hpp"

#include <atomic>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <optional>
#include "training_pixel_culling_gpu_probe.hpp"
#include "training_scanline_gpu_diagnostic.hpp"
#include "training_scanline_reference.hpp"
#include "training_row_empty_gpu_checks.hpp"
#include "training_row_empty_kernel_checks.hpp"
#include "training_row_empty_prototype_report.hpp"

using namespace vulkan3DGS;
namespace reference = checkpoint_reference;
namespace {
std::atomic<uint32_t> validationErrors = 0;
VKAPI_ATTR vk::Bool32 VKAPI_CALL onMessage(vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
                                           vk::DebugUtilsMessageTypeFlagsEXT,
                                           const vk::DebugUtilsMessengerCallbackDataEXT* data,
                                           void*) {
    if (severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eError) {
        ++validationErrors;
        std::cerr << "Vulkan validation: " << data->pMessage << '\n';
    }
    return VK_FALSE;
}
void check(bool valid, const char* message) {
    if (!valid) throw std::runtime_error(message);
}
bool near(float a, float b) {
    return std::isfinite(a) && std::isfinite(b) &&
           std::abs(a - b) <= 1e-5f + 1e-3f * std::max(std::abs(a), std::abs(b));
}

void checkWorkspaceLifecycle(vk::Device device, vk::PhysicalDevice physical, vk::Queue queue,
                             uint32_t family) {
    TrainingBuffers buffers;
    buffers.initialize(device, physical, queue, family, family);
    try {
        buffers.resize(65u, {17u, 19u});
        buffers.ensureCheckpointCapacity(65u);
        const auto first = buffers.checkpointResources();
        buffers.ensureCheckpointCapacity(33u);
        const auto reused = buffers.checkpointResources();
        check(first.descriptors[6].buffer == reused.descriptors[6].buffer &&
                  first.capacity.slotCapacity == reused.capacity.slotCapacity,
              "Checkpoint high-water workspace was not reused");
        buffers.ensureCheckpointCapacity(129u);
        const auto grown = buffers.checkpointResources();
        check(grown.capacity.slotCapacity > first.capacity.slotCapacity &&
                  grown.descriptors[6].buffer != first.descriptors[6].buffer,
              "Checkpoint workspace did not replace resources on growth");
        bool rejected = false;
        try {
            buffers.ensureCheckpointCapacity(0xffffffffu);
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        check(rejected && buffers.checkpointResources().descriptors[6].buffer ==
                              grown.descriptors[6].buffer,
              "Rejected checkpoint growth damaged the previous workspace");
        buffers.resize(65u, {1u, 1u});
        rejected = false;
        try {
            (void)buffers.checkpointResources();
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        check(rejected, "Checkpoint workspace survived an extent change");
        buffers.ensureCheckpointCapacity(0u);
        buffers.releaseCheckpointWorkspace();
    } catch (...) {
        buffers.cleanup();
        throw;
    }
    buffers.cleanup();
    std::cout << "GPU workspace growth/reuse/rejection/resize/release passed\n";
}

void runFixture(vk::Device device, vk::PhysicalDevice physical, vk::Queue queue, uint32_t family,
                CheckpointStatePass& statePass, uint32_t width, uint32_t height,
                const std::vector<uint32_t>& counts, uint32_t pattern, bool overflow = false,
                CheckpointBackwardPass* warpPass = nullptr, uint32_t requiredSubgroupSize = 0u,
                bool fullSubgroups = false, bool pixelCull = false) {
    const uint32_t tilesX = (width + 15u) / 16u;
    check(counts.size() == uint64_t(tilesX) * ((height + 15u) / 16u),
          "Fixture tile count mismatch");
    std::vector<reference::Scene<float>> scenes(counts.size());
    std::vector<ProjectedGaussian> projected;
    std::vector<uint32_t> items;
    std::vector<glm::uvec4> ranges;
    for (uint32_t tile = 0; tile < counts.size(); ++tile) {
        const uint32_t base = static_cast<uint32_t>(projected.size());
        ranges.emplace_back(static_cast<uint32_t>(items.size()), counts[tile], 0u, 0u);
        auto& scene = scenes[tile];
        for (uint32_t i = 0; i < counts[tile]; ++i) {
            reference::Gaussian<float> g{};
            g.centerX = float((tile % tilesX) * 16u);
            g.centerY = float((tile / tilesX) * 16u);
            g.opacity = pattern == 3u ? 0.001f : pattern == 4u ? 0.005f : 0.01f;
            g.color = {float(i % 17u) / 16.0f, 0.3f, 0.1f};
            if (pattern == 5u) {
                g.centerX = g.centerY = 8.0f;
                g.a = g.c = 0.001f;
                g.b = 0.0f;
                g.opacity = 0.005f;
            }
            if (pattern == 2u) {
                g.opacity = i < 32u ? 0.005f : i == 32u ? 0.99f : i == 33u ? 0.985f : 0.01f;
                g.color = {i <= 32u ? 64.0f : i == 33u ? 0.0002f : 0.0f, 0, 0};
            }
            if (pattern == 6u || pattern == 7u) {
                g.centerX += float(i * 7u % 16u) + 0.25f;
                g.centerY += float(i * 11u % 16u) + 0.5f;
                g.a = i % 3u == 0u ? 0.02f : 20.0f;
                g.c = i % 3u == 0u ? 20.0f : 0.02f;
                g.b = 0.5f;
                g.opacity = pixel_culling_reference::alphaMin + float(i % 11u) / 32.0f;
                g.color = {i % 2u == 0u ? 64.0f : 0.0002f, 0.3f, 0.1f};
                if (pattern == 7u && i % 2u == 0u) g.a = -1.0f; // Fallback: indefinite conic.
            }
            if (pattern == 8u) g.opacity = 1.0f / 255.0f;
            if (pattern == 9u) {
                const bool tiny = i + 1u < counts[tile];
                g.centerX += tiny ? 8.5f : 8.0f;
                g.centerY += tiny ? 8.5f : 8.0f;
                g.a = g.c = tiny ? 1000.0f : 0.02f;
                g.b = 0.0f;
            }
            if (pattern >= 10u && pattern <= 14u) {
                g.centerX += 8.0f;
                g.centerY += ((pattern == 10u ? i / 16u : i) % 2u) ? 4.0f : 12.0f;
                g.a = 0.03f; g.b = 0.0f; g.c = 0.18f; g.opacity = 0.015f;
                if (pattern == 12u && i < 2u) g.opacity = 0.99f;
                if (pattern == 13u && i % 3u == 0u) g.a = -0.1f;
                if (pattern == 14u) {
                    g.centerY = float((tile / tilesX) * 16u) + 8.0f;
                    g.a = 1.0f; g.c = 2.0f; g.opacity = 0.1f;
                }
                if (pattern != 12u && i + 1u == counts[tile]) {
                    g.a = g.c = 0.001f; g.opacity = 0.005f; // Keep the full processed prefix alive.
                }
            }
            if (pattern == 1u && i == 64u) g.opacity = 0.001f;
            scene.gaussians.push_back(g);
            ProjectedGaussian gpu{};
            gpu.centerRadius = {g.centerX, g.centerY, 1, 4};
            gpu.conicOpacity = {g.a, g.b, g.c, g.opacity};
            gpu.color = {g.color[0], g.color[1], g.color[2], 1};
            projected.push_back(gpu);
            const uint32_t id = pattern == 5u               ? i % 2u
                                : pattern == 1u && i == 32u ? 0xffffffffu
                                : pattern == 4u && i == 64u ? 1u
                                                            : i;
            scene.items.push_back(id);
            items.push_back(id == 0xffffffffu ? id : pattern == 5u ? id : base + id);
        }
    }
    TrainingPushConstants params{};
    params.gaussianCount = static_cast<uint32_t>(projected.size());
    params.width = width;
    params.height = height;
    params.pixelCount = width * height;
    params.tileItemCount = static_cast<uint32_t>(items.size());
    params.tileSize = 16u;
    TrainingCheckpointResources resources{};
    resources.capacity = makeTrainingCheckpointCapacity(params.tileItemCount, width, height);
    std::array<Buffer, 12> buffers;
    auto create = [&](Buffer& buffer, uint64_t bytes, const void* data = nullptr) {
        buffer.create(
            device, physical, queue, family, data, std::max<uint64_t>(bytes, 16u),
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc |
                vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndirectBuffer,
            vk::MemoryPropertyFlagBits::eDeviceLocal, {family});
    };
    create(buffers[0], projected.size() * sizeof(ProjectedGaussian));
    if (!projected.empty())
        buffers[0].upload(projected.data(), projected.size() * sizeof(ProjectedGaussian));
    create(buffers[1], items.size() * sizeof(uint32_t));
    if (!items.empty()) buffers[1].upload(items.data(), items.size() * sizeof(uint32_t));
    create(buffers[2], ranges.size() * sizeof(glm::uvec4), ranges.data());
    create(buffers[3], uint64_t(width) * height * sizeof(PixelBlendState));
    create(buffers[4], resources.capacity.metadataBytes);
    create(buffers[5],
           uint64_t(checkpointScratchElements(static_cast<uint32_t>(counts.size()))) * 8u);
    for (size_t i = 6; i < 10; ++i)
        create(buffers[i], resources.capacity.bankBytes);
    create(buffers[10], resources.capacity.taskBytes);
    create(buffers[11], sizeof(TrainingCheckpointControl));
    for (size_t i = 0; i < buffers.size(); ++i)
        resources.descriptors[i] = buffers[i].getDescriptorInfo();
    if (overflow) resources.capacity.slotCapacity = 0u;
    Buffer rendered;
    create(rendered, uint64_t(width) * height * sizeof(glm::vec4));
    Buffer pixelGradientBuffer, gradientBuffer;
    std::vector<PixelGrad> inputGradients(width * height);
    for (auto& gradient : inputGradients)
        gradient.color = {0.3f, -0.2f, 0.1f, 0.0f};
    create(pixelGradientBuffer, uint64_t(width) * height * sizeof(PixelGrad),
           inputGradients.data());
    create(gradientBuffer,
           uint64_t(std::max(params.gaussianCount, 1u)) * sizeof(ProjectedGaussianGrad));
    ComputePipeline tensor, culled;
    BackwardDescriptorPool tensorPool;
    vk::DescriptorSet tensorSet = nullptr;
    if (warpPass || pixelCull) {
        vk::PhysicalDeviceSubgroupProperties subgroup{};
        vk::PhysicalDeviceProperties2 properties{};
        properties.pNext = &subgroup;
        physical.getProperties2(&properties);
        const bool tensorReference =
            subgroup.subgroupSize == 32u &&
            properties.properties.limits.maxComputeSharedMemorySize >= 45u * 1024u;
        ComputePipelineConfig tensorConfig{};
        for (uint32_t binding : {3u, 4u, 5u, 12u, 13u, 14u})
            tensorConfig.descriptorBindings.push_back(backwardStorageBinding(binding));
        tensorConfig.pushConstantSize = sizeof(params);
        (void)requiredSubgroupSize;
        (void)fullSubgroups;
        tensor.initialize(
            device,
            std::string(CHECKPOINT_RUNTIME_DIR) +
                (tensorReference
                     ? "/shaders/train_backward_vksplat_tensor_subgroup_global_grad.comp.spv"
                     : "/shaders/train_backward_pixel_to_2dgs.comp.spv"),
            tensorConfig);
        if (pixelCull) {
            culled.initialize(device, std::string(CHECKPOINT_RUNTIME_DIR) +
                (tensorReference ? "/shaders/train_backward_vksplat_tensor_early_cull.comp.spv"
                                 : "/shaders/train_backward_pixel_culling_test.comp.spv"),
                tensorConfig);
            std::cout << "S3 gradient candidate=" << (tensorReference ? "Tensor" : "Direct test-only")
                      << '\n';
        }
        tensorSet = tensorPool.create(device, 6u, 0u, {tensor.getDescriptorSetLayout()})[0];
        const std::array<vk::DescriptorBufferInfo, 6> infos = {
            resources.descriptors[0],           resources.descriptors[1],
            resources.descriptors[2],           pixelGradientBuffer.getDescriptorInfo(),
            gradientBuffer.getDescriptorInfo(), resources.descriptors[3]};
        constexpr std::array<uint32_t, 6> bindings = {3, 4, 5, 12, 13, 14};
        std::array<vk::WriteDescriptorSet, 6> writes{};
        for (size_t i = 0; i < writes.size(); ++i)
            writes[i]
                .setDstSet(tensorSet)
                .setDstBinding(bindings[i])
                .setDescriptorCount(1)
                .setDescriptorType(vk::DescriptorType::eStorageBuffer)
                .setPBufferInfo(&infos[i]);
        device.updateDescriptorSets(writes, {});
    }
    ComputePipeline forward;
    ComputePipelineConfig config{};
    for (uint32_t binding : {3u, 4u, 5u, 6u, 14u})
        config.descriptorBindings.push_back(backwardStorageBinding(binding));
    config.pushConstantSize = sizeof(params);
    forward.initialize(
        device, std::string(CHECKPOINT_RUNTIME_DIR) + "/shaders/train_forward_workgroup.comp.spv",
        config);
    BackwardDescriptorPool descriptorPool;
    const auto descriptor =
        descriptorPool.create(device, 5u, 0u, {forward.getDescriptorSetLayout()})[0];
    const std::array<vk::DescriptorBufferInfo, 5> infos = {
        buffers[0].getDescriptorInfo(), buffers[1].getDescriptorInfo(),
        buffers[2].getDescriptorInfo(), rendered.getDescriptorInfo(),
        buffers[3].getDescriptorInfo()};
    constexpr std::array<uint32_t, 5> bindings = {3, 4, 5, 6, 14};
    std::array<vk::WriteDescriptorSet, 5> writes{};
    for (size_t i = 0; i < writes.size(); ++i)
        writes[i]
            .setDstSet(descriptor)
            .setDstBinding(bindings[i])
            .setDescriptorCount(1)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer)
            .setPBufferInfo(&infos[i]);
    device.updateDescriptorSets(writes, {});
    CommandPool pool;
    pool.create(device, family, vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
    const auto command = pool.allocateCommandBuffer();
    auto submit = [&](bool diagnostic, bool resetRun = true) {
        command.reset();
        command.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        command.fillBuffer(gradientBuffer.getBuffer(), 0, VK_WHOLE_SIZE, 0u);
        if (diagnostic && !pixelCull) {
            for (size_t i = 6; i < 10; ++i)
                command.fillBuffer(buffers[i].getBuffer(), 0, VK_WHOLE_SIZE, 0x7fc00000u);
        }
        vk::MemoryBarrier inputBarrier{
            vk::AccessFlagBits::eTransferWrite | vk::AccessFlagBits::eShaderWrite,
            vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite};
        command.pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer | vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eComputeShader, {}, inputBarrier, {}, {});
        command.bindPipeline(vk::PipelineBindPoint::eCompute, forward.getPipeline());
        command.bindDescriptorSets(vk::PipelineBindPoint::eCompute, forward.getPipelineLayout(), 0,
                                   descriptor, {});
        command.pushConstants(forward.getPipelineLayout(), vk::ShaderStageFlagBits::eCompute, 0,
                              sizeof(params), &params);
        command.dispatch(tilesX, (height + 15u) / 16u, 1);
        if (diagnostic && !pixelCull) statePass.record(command, params, resources, nullptr, resetRun);
        if (warpPass || pixelCull) {
            vk::MemoryBarrier ready{
                vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite,
                vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite |
                    vk::AccessFlagBits::eIndirectCommandRead};
            command.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader |
                                        vk::PipelineStageFlagBits::eTransfer,
                                    vk::PipelineStageFlagBits::eComputeShader |
                                        vk::PipelineStageFlagBits::eDrawIndirect,
                                    {}, ready, {}, {});
            if (diagnostic && !pixelCull) {
                warpPass->record(command, params, resources,
                                 pixelGradientBuffer.getDescriptorInfo(),
                                 gradientBuffer.getDescriptorInfo());
                vk::MemoryBarrier between{vk::AccessFlagBits::eShaderWrite,
                                          vk::AccessFlagBits::eShaderRead |
                                              vk::AccessFlagBits::eShaderWrite |
                                              vk::AccessFlagBits::eIndirectCommandRead |
                                              vk::AccessFlagBits::eTransferRead};
                command.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                        vk::PipelineStageFlagBits::eComputeShader |
                                            vk::PipelineStageFlagBits::eDrawIndirect |
                                            vk::PipelineStageFlagBits::eTransfer,
                                        {}, between, {}, {});
            }
            auto& selected = pixelCull && diagnostic ? culled : tensor;
            command.bindPipeline(vk::PipelineBindPoint::eCompute, selected.getPipeline());
            command.bindDescriptorSets(vk::PipelineBindPoint::eCompute, selected.getPipelineLayout(),
                                       0, tensorSet, {});
            command.pushConstants(selected.getPipelineLayout(), vk::ShaderStageFlagBits::eCompute, 0,
                                  sizeof(params), &params);
            if (diagnostic && !pixelCull)
                command.dispatchIndirect(resources.descriptors[11].buffer,
                                         offsetof(TrainingCheckpointControl, fallback));
            else
                command.dispatch(tilesX, (height + 15u) / 16u, 1u);
        }
        vk::MemoryBarrier barrier{vk::AccessFlagBits::eShaderWrite,
                                  vk::AccessFlagBits::eTransferRead};
        command.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                vk::PipelineStageFlagBits::eTransfer, {}, barrier, {}, {});
        command.end();
        auto fence = device.createFenceUnique({});
        vk::SubmitInfo info{};
        info.setCommandBuffers(command);
        queue.submit(info, *fence);
        (void)device.waitForFences(*fence, VK_TRUE, UINT64_MAX);
    };
    std::vector<PixelBlendState> baselineBlend(width * height), blend(width * height);
    std::vector<glm::vec4> baselineColor(width * height), color(width * height);
    std::vector<ProjectedGaussianGrad> baselineGrad(params.gaussianCount),
        actualGrad(params.gaussianCount);
    std::vector<reference::Gradient<float>> cpuGrad(params.gaussianCount);
    submit(false);
    if ((warpPass || pixelCull) && !baselineGrad.empty())
        gradientBuffer.download(baselineGrad.data(),
                                baselineGrad.size() * sizeof(ProjectedGaussianGrad));
    buffers[3].download(baselineBlend.data(), baselineBlend.size() * sizeof(PixelBlendState));
    rendered.download(baselineColor.data(), baselineColor.size() * sizeof(glm::vec4));
    submit(true);
    if (width == 17u && pattern == 1u) submit(true); // Same buffers: counters must reset per step.
    buffers[3].download(blend.data(), blend.size() * sizeof(PixelBlendState));
    rendered.download(color.data(), color.size() * sizeof(glm::vec4));
    check(std::memcmp(blend.data(), baselineBlend.data(), blend.size() * sizeof(PixelBlendState)) ==
                  0 &&
              std::memcmp(color.data(), baselineColor.data(), color.size() * sizeof(glm::vec4)) ==
                  0,
          "P1 state construction changed the original forward output");
    TrainingCheckpointControl control{};
    if (!pixelCull) buffers[11].download(&control, sizeof(control));
    if ((warpPass || pixelCull) && !actualGrad.empty()) {
        gradientBuffer.download(actualGrad.data(),
                                actualGrad.size() * sizeof(ProjectedGaussianGrad));
        for (size_t id = 0; id < actualGrad.size(); ++id) {
            std::array<float, 12> expected{}, actual{};
            std::memcpy(expected.data(), &baselineGrad[id], sizeof(expected));
            std::memcpy(actual.data(), &actualGrad[id], sizeof(actual));
            for (size_t component = 0; component < 12; ++component)
                if (!near(expected[component], actual[component])) {
                    std::cerr << "P2/GPU reference mismatch Gaussian=" << id
                              << " component=" << component << " expected=" << expected[component]
                              << " actual=" << actual[component] << '\n';
                    throw std::runtime_error(
                        "Checkpoint warp gradient differs from original GPU reference");
                }
        }
    }
    if (pixelCull) {
        TrainingScanlineCapture capture;
        auto& h = capture.header;
        h.gaussianCount = params.gaussianCount; h.width = width; h.height = height;
        h.itemCount = params.tileItemCount; h.tileCount = static_cast<uint32_t>(ranges.size());
        capture.projected = projected; capture.items = items; capture.ranges = ranges;
        capture.blends = baselineBlend; capture.pixelGrads = inputGradients;
        const auto census = runScanlineGpuDiagnostic(device, physical, queue, family, capture);
        const auto oracle = scanline_reference::oracle(capture);
        const auto rowCensus = runScanlineGpuDiagnostic(device, physical, queue, family, capture, true);
        checkRowKernelFixture(device,physical,queue,family,capture);
        for (size_t counter : {0u,1u,2u,3u,4u,5u,6u,7u,8u,9u,10u,11u,12u,13u,25u,28u,29u,31u})
            if (census.counts[counter] != oracle[counter] || rowCensus.counts[counter] != oracle[counter]) {
                std::cerr << "S3-2 oracle mismatch " << kScanlineCounterNames[counter]
                          << " GPU=" << census.counts[counter] << " CPU=" << oracle[counter] << '\n';
                throw std::runtime_error("S3-2 GPU census differs from independent scalar oracle");
            }
        check(census.counts[24] == 0u && census.counts[26] == 0u,
              "S3-2 span rejected an original contributor or accessed an invalid range");
        check(rowCensus.counts[24] == 0u && rowCensus.counts[26] == 0u &&
              rowCensus.counts[14] == rowCensus.counts[17] &&
              rowCensus.rowEmptyCounts[1] == rowCensus.counts[14] &&
              rowCensus.rowEmptyCounts[5] <= rowCensus.rowEmptyCounts[4],
              "S3-3 R0 mask safety or wave/counter accounting failed");
        check(census.counts[14] <= census.counts[1] && census.counts[15] <= census.counts[3] &&
              census.counts[16] <= census.counts[6] && census.counts[20] <= census.counts[12],
              "S3-2 predicted savings exceed the oracle opportunity");
        std::cout << "S3-2 census: eligible=" << census.counts[1] << " oracle-empty-rows="
                  << census.counts[6] << " oracle-empty-batches=" << census.counts[12]
                  << " span-empty-batches=" << census.counts[20] << " false-drops=" << census.counts[24] << '\n';
        pool.freeCommandBuffer(command);
        std::cout << "S3 GPU forward/BlendState/12-component gradient fixture passed: "
                  << width << 'x' << height << " pattern=" << pattern << '\n';
        return;
    }
    if (overflow) {
        check((control.totals[2] & 2u) != 0u && control.diagnostics[0] == 0u &&
                  control.dispatch[0] == 0u && control.fallback[0] == tilesX &&
                  control.fallback[1] == (height + 15u) / 16u && control.fallback[3] == 1u &&
                  control.run[0] == 1u,
              "GPU checkpoint overflow did not suppress state/tasks/indirect dispatch");
        resources.capacity.slotCapacity = checkpointBucketCount(params.tileItemCount);
        submit(true, false);
        buffers[11].download(&control, sizeof(control));
        check(control.totals[2] == 0u && control.run[0] == 1u && (control.run[1] & 2u) != 0u &&
                  control.run[2] == 2u && control.fallback[0] == 0u,
              "GPU fallback history was lost between steps");
        submit(true, true);
        buffers[11].download(&control, sizeof(control));
        check(control.run[0] == 0u && control.run[1] == 0u && control.run[2] == 1u,
              "GPU measured-interval fallback reset failed");
    } else {
        check(control.totals[2] == 0u, "GPU checkpoint diagnostic flags are nonzero");
        std::vector<TrainingCheckpointTileMetadata> metadata(counts.size());
        buffers[4].download(metadata.data(), metadata.size() * sizeof(metadata[0]));
        std::vector<TrainingCheckpointTask> tasks(control.totals[1]);
        if (!tasks.empty()) buffers[10].download(tasks.data(), tasks.size() * sizeof(tasks[0]));
        const size_t bankElements = static_cast<size_t>(resources.capacity.bankElements);
        std::array<std::vector<glm::vec4>, 4> banks;
        for (size_t bank = 0; bank < 4; ++bank) {
            banks[bank].resize(std::max<size_t>(bankElements, 1u));
            buffers[6 + bank].download(banks[bank].data(), banks[bank].size() * sizeof(glm::vec4));
        }
        uint32_t slotBase = 0, taskBase = 0, pixelStates = 0;
        for (uint32_t tile = 0; tile < counts.size(); ++tile) {
            uint32_t maximum = 0;
            const auto& scene = scenes[tile];
            for (uint32_t rank = 0; rank < 256; ++rank) {
                const uint32_t x = (tile % tilesX) * 16u + rank % 16u;
                const uint32_t y = (tile / tilesX) * 16u + rank / 16u;
                const bool valid = x < width && y < height;
                reference::Forward<float> state{};
                if (valid) {
                    const auto& gpu = blend[y * width + x];
                    state.processedCount = std::min(gpu.processedCount, counts[tile]);
                    state.finalState.transmittance = gpu.finalTransmittance;
                    if (warpPass) {
                        const auto gradients =
                            reference::reverse(scene, state, float(x), float(y),
                                               reference::Color<float>{0.3f, -0.2f, 0.1f});
                        for (size_t id = 0; id < gradients.size(); ++id) {
                            const size_t global =
                                pattern == 5u && id < 2u ? id : ranges[tile].x + id;
                            for (size_t component = 0; component < 9; ++component)
                                cpuGrad[global][component] += gradients[id][component];
                        }
                    }
                }
                maximum = std::max(maximum, state.processedCount);
                const auto expected =
                    reference::buildReverseCheckpoints(scene, state, float(x), float(y), valid);
                for (uint32_t boundary = 0; boundary < expected.boundaries.size(); ++boundary) {
                    const auto address = trainingCheckpointAddress(slotBase + boundary, rank,
                                                                   resources.capacity.slotCapacity);
                    check(address.has_value(), "GPU fixture checkpoint address invalid");
                    const auto actual = banks[address->bank][address->element];
                    if (expected.boundaries[boundary]) {
                        const auto& value = *expected.boundaries[boundary];
                        for (uint32_t channel = 0; channel < 3; ++channel)
                            check(near(value.suffix[channel], actual[channel]),
                                  "GPU suffix boundary mismatch");
                        check(near(value.transmittance, actual.w),
                              "GPU transmittance boundary mismatch");
                        ++pixelStates;
                    } else {
                        for (uint32_t channel = 0; channel < 4; ++channel)
                            check(std::isnan(actual[channel]),
                                  "Unreachable checkpoint state was overwritten");
                    }
                }
            }
            const uint32_t active = checkpointBucketCount(maximum);
            check(metadata[tile].range ==
                          std::array<uint32_t, 4>{ranges[tile].x, counts[tile], slotBase,
                                                  checkpointStoredCount(counts[tile])} &&
                      metadata[tile].activity ==
                          std::array<uint32_t, 4>{maximum, active, taskBase, 0u},
                  "GPU checkpoint prefix/metadata mismatch");
            for (uint32_t bucket = 0; bucket < active; ++bucket) {
                const auto expected = makeReverseCheckpointTask(
                    tile, counts[tile], slotBase, bucket, resources.capacity.slotCapacity);
                check(expected && std::memcmp(&tasks[taskBase + bucket], &*expected,
                                              sizeof(*expected)) == 0,
                      "GPU active bucket task mismatch");
            }
            slotBase += checkpointStoredCount(counts[tile]);
            taskBase += active;
        }
        check(control.totals[0] == slotBase && control.totals[1] == taskBase &&
                  control.totals[3] == pixelStates && control.diagnostics[0] == pixelStates,
              "GPU checkpoint totals or written-state count mismatch");
        if (warpPass) {
            check(taskBase == 0u || control.run[3] == 32u, "GPU warp size is not 32");
            constexpr std::array<uint32_t, 9> active = {0, 1, 4, 5, 6, 7, 8, 9, 10};
            for (size_t id = 0; id < actualGrad.size(); ++id) {
                std::array<float, 12> actual{};
                std::memcpy(actual.data(), &actualGrad[id], sizeof(actual));
                for (size_t component = 0; component < 9; ++component)
                    check(near(cpuGrad[id][component], actual[active[component]]),
                          "GPU warp gradient differs from independent CPU reverse");
            }
        }
    }
    std::cout << "GPU fixture " << width << 'x' << height << " tiles=" << counts.size()
              << " pattern=" << pattern << " overflow=" << overflow << " passed\n";
    pool.freeCommandBuffer(command);
    forward.cleanup();
}
} // namespace

int main(int argc, char** argv) {
    try {
        const bool cullingOnly = argc == 2 && std::strcmp(argv[1], "--pixel-culling-only") == 0;
        const bool scanlineOnly = argc == 2 && std::strcmp(argv[1], "--scanline-only") == 0;
        const bool scanlineCapture = argc == 3 && std::strcmp(argv[1], "--scanline-capture") == 0;
        const bool rowEmptyCapture = argc == 3 && std::strcmp(argv[1], "--row-empty-r0-capture") == 0;
        const bool rowPrototype = (argc == 3 || (argc == 5 && std::strcmp(argv[3], "--timing") == 0 && std::strcmp(argv[4], "--resources-reviewed") == 0)) && std::strcmp(argv[1], "--row-empty-capture") == 0;
        const bool longRowTiming = rowPrototype && argc == 5;
        check(argc == 1 || cullingOnly || scanlineOnly || scanlineCapture || rowEmptyCapture || rowPrototype,
              "Usage: training_checkpoint_gpu_tests [--pixel-culling-only | --scanline-only | --scanline-capture INPUT.bin | --row-empty-r0-capture INPUT.bin | --row-empty-capture INPUT.bin [--timing --resources-reviewed]]");
        std::optional<TrainingScanlineCapture> offlineCapture;
        if (scanlineCapture || rowEmptyCapture || rowPrototype) offlineCapture = readTrainingScanlineCapture(std::filesystem::path(argv[2]));
        if (!scanlineCapture) verifyRowEmptyShader();
        if (rowPrototype || !offlineCapture) checkRowShaderAudit();
        VULKAN_HPP_DEFAULT_DISPATCHER.init();
        bool validation = false;
        for (const auto& layer : vk::enumerateInstanceLayerProperties())
            validation |= std::strcmp(layer.layerName.data(), "VK_LAYER_KHRONOS_validation") == 0;
        const char* layer = "VK_LAYER_KHRONOS_validation";
        std::vector<const char*> extensions;
        bool synchronizationValidation = false;
        vk::ValidationFeatureEnableEXT syncFeature =
            vk::ValidationFeatureEnableEXT::eSynchronizationValidation;
        vk::ValidationFeaturesEXT features{};
        vk::ApplicationInfo app{"checkpoint-p1", 1, "vulkan-3dgs", 1, VK_API_VERSION_1_2};
        vk::InstanceCreateInfo create{};
        create.setPApplicationInfo(&app);
        if (validation) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
            uint32_t extensionCount = 0;
            (void)vkEnumerateInstanceExtensionProperties(layer, &extensionCount, nullptr);
            std::vector<VkExtensionProperties> available(extensionCount);
            (void)vkEnumerateInstanceExtensionProperties(layer, &extensionCount, available.data());
            for (const auto& extension : available)
                synchronizationValidation |=
                    std::strcmp(extension.extensionName,
                                VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME) == 0;
            if (synchronizationValidation) {
                extensions.push_back(VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME);
                features.setEnabledValidationFeatures(syncFeature);
                create.setPNext(&features);
            }
            create.setPEnabledLayerNames(layer).setPEnabledExtensionNames(extensions);
        }
        auto instance = vk::createInstanceUnique(create);
        VULKAN_HPP_DEFAULT_DISPATCHER.init(*instance);
        vk::UniqueDebugUtilsMessengerEXT messenger;
        if (validation) {
            vk::DebugUtilsMessengerCreateInfoEXT info{};
            info.setMessageSeverity(vk::DebugUtilsMessageSeverityFlagBitsEXT::eError)
                .setMessageType(vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
                                vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance);
            info.pfnUserCallback = onMessage;
            messenger = instance->createDebugUtilsMessengerEXTUnique(info);
        }
        const auto devices = instance->enumeratePhysicalDevices();
        check(!devices.empty(), "No Vulkan GPU available");
        vk::PhysicalDevice physical = devices.front();
        for (const auto candidate : devices)
            if (candidate.getProperties().deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
                physical = candidate;
                break;
            }
        if (offlineCapture) {
            for (const auto candidate : devices) {
                const auto info = candidate.getProperties();
                if (info.vendorID == offlineCapture->header.vendorID && info.deviceID == offlineCapture->header.deviceID) {
                    physical = candidate;
                    break;
                }
            }
        }
        uint32_t family = 0xffffffffu;
        const auto queues = physical.getQueueFamilyProperties();
        for (uint32_t i = 0; i < queues.size(); ++i)
            if (queues[i].queueFlags & vk::QueueFlagBits::eCompute) {
                family = i;
                break;
            }
        check(family != 0xffffffffu, "No compute queue available");
        bool atomicExtension = false, sizeExtension = false;
        for (const auto& extension : physical.enumerateDeviceExtensionProperties()) {
            atomicExtension |= std::strcmp(extension.extensionName.data(),
                                           VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME) == 0;
            sizeExtension |= std::strcmp(extension.extensionName.data(),
                                         VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME) == 0;
        }
        check(atomicExtension, "P2 GPU checks require buffer float32 atomic add");
        vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT atomicSupported{};
        vk::PhysicalDeviceSubgroupSizeControlFeaturesEXT sizeSupported{};
        vk::PhysicalDeviceFeatures2 supported{};
        supported.pNext = &atomicSupported;
        if (sizeExtension) atomicSupported.pNext = &sizeSupported;
        physical.getFeatures2(&supported);
        check(atomicSupported.shaderBufferFloat32AtomicAdd, "P2 float32 atomic add is unsupported");
        vk::PhysicalDeviceSubgroupProperties subgroup{};
        vk::PhysicalDeviceSubgroupSizeControlPropertiesEXT sizes{};
        vk::PhysicalDeviceProperties2 properties{};
        properties.pNext = &subgroup;
        if (sizeExtension) subgroup.pNext = &sizes;
        physical.getProperties2(&properties);
        const uint32_t requiredWidth = subgroup.subgroupSize == 32u || cullingOnly || scanlineOnly || scanlineCapture || rowEmptyCapture || rowPrototype ? 0u : 32u;
        if (requiredWidth != 0u)
            check(sizeExtension && sizeSupported.subgroupSizeControl &&
                      sizes.minSubgroupSize <= 32u && sizes.maxSubgroupSize >= 32u &&
                      (sizes.requiredSubgroupSizeStages & vk::ShaderStageFlagBits::eCompute),
                  "P2 GPU checks need native subgroup32 or enabled subgroup-size control");
        const bool fullSubgroups = requiredWidth != 0u && sizeSupported.computeFullSubgroups;
        vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT atomicEnabled{};
        atomicEnabled.shaderBufferFloat32AtomicAdd = true;
        vk::PhysicalDeviceSubgroupSizeControlFeaturesEXT sizeEnabled{};
        std::vector<const char*> deviceExtensions{VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME};
        if (requiredWidth != 0u) {
            sizeEnabled.subgroupSizeControl = true;
            sizeEnabled.computeFullSubgroups = fullSubgroups;
            atomicEnabled.pNext = &sizeEnabled;
            deviceExtensions.push_back(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);
        }
        const float priority = 1;
        vk::DeviceQueueCreateInfo queueInfo{{}, family, 1, &priority};
        vk::DeviceCreateInfo deviceInfo{};
        deviceInfo.setQueueCreateInfos(queueInfo);
        deviceInfo.setPNext(&atomicEnabled).setPEnabledExtensionNames(deviceExtensions);
        auto device = physical.createDeviceUnique(deviceInfo);
        VULKAN_HPP_DEFAULT_DISPATCHER.init(*device);
        const auto queue = device->getQueue(family, 0);
        if (rowPrototype) {
            auto result=runRowPrototype(*device,physical,queue,family,*offlineCapture,longRowTiming,&validationErrors);
            if(validationErrors) {result.verdict="STOP_VULKAN_VALIDATION_ERRORS"; result.error="Vulkan validation reported errors";}
            writeRowPrototypeReport(std::filesystem::path(argv[2]),physical,*offlineCapture,result,longRowTiming,validation,synchronizationValidation);
            check(validationErrors==0u && result.error.empty(),"RowEmpty evaluation failed; see R2 report");
            check(!result.gradientsRun || (result.gradients.passed() && result.repeat.passed() && result.inputsSame),"RowEmpty gradients or readonly inputs changed; see STOP report");
            check(result.verdict!="STOP_R0_CLASSIFICATION_OR_INPUT","RowEmpty R0 rejected input; see STOP report");
            return 0;
        }
        if (rowEmptyCapture) {
            const auto report = runScanlineGpuDiagnostic(*device, physical, queue, family, *offlineCapture, true);
            check(validationErrors == 0u, "S3-3 R0 Vulkan validation reported errors");
            writeRowEmptyGateReport(std::filesystem::path(argv[2]), physical, *offlineCapture, report);
            check(report.counts[24] == 0u && report.counts[25] == 0u && report.counts[26] == 0u && report.counts[31] == 0u,
                  "S3-3 R0 classification or input rejected; see STOP report");
            return 0;
        }
        if (scanlineOnly) {
            TrainingStepExecutor downloadSync;
            downloadSync.initialize(*device, queue, family);
            downloadSync.synchronizeDiagnosticDownloads(queue);
            downloadSync.cleanup();
        }
        if (scanlineCapture) {
            const auto& capture = *offlineCapture;
            const auto report = runScanlineGpuDiagnostic(*device, physical, queue, family, capture);
            check(validationErrors == 0u, "S3-2 Vulkan validation reported errors");
            auto reportPath = std::filesystem::path(argv[2]); reportPath.replace_extension(".json");
            writeScanlineGpuReport(reportPath, physical, capture, report);
            check(report.counts[24] == 0u && report.counts[26] == 0u, "S3-2 candidate or input rejected; see STOP report");
            return 0;
        }
        runRowEmptyGpuChecks(*device, physical, queue, family);
        if (!scanlineOnly) runPixelCullingProbe(*device, physical, queue, family);
        std::cout << "GPU=" << physical.getProperties().deviceName.data()
                  << " validation=" << validation << " syncValidation=" << synchronizationValidation
                  << " nativeSubgroup=" << subgroup.subgroupSize
                  << " requiredSubgroup=" << requiredWidth << '\n';
        std::cout << "Gradient GPU reference="
                  << (subgroup.subgroupSize == 32u &&
                              physical.getProperties().limits.maxComputeSharedMemorySize >=
                                  45u * 1024u
                          ? "Tensor Subgroup/Global"
                          : "Direct (Tensor capability gate unavailable)")
                  << '\n';
        checkWorkspaceLifecycle(*device, physical, queue, family);
        CheckpointStatePass state;
        state.initialize(*device, physical, std::string(CHECKPOINT_RUNTIME_DIR) + "/shaders");
        for (uint32_t length : {0u, 1u, 15u, 16u, 17u, 31u, 32u, 33u, 1025u})
            runFixture(*device, physical, queue, family, state, 1u, 1u, {length}, 4u,
                       false, nullptr, 0u, false, true);
        for (uint32_t pattern : {1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u, 13u, 14u})
            runFixture(*device, physical, queue, family, state, 17u, 19u, {33u, 65u, 65u, 63u},
                       pattern, false, nullptr, 0u, false, true);
        runFixture(*device, physical, queue, family, state, 16u, 16u, {1500u}, 0u,
                   false, nullptr, 0u, false, true);
        if (cullingOnly || scanlineOnly) {
            check(validationErrors == 0u, "GPU EarlyCull validation reported errors");
            return 0;
        }
        CheckpointBackwardPass warp;
        warp.initialize(*device, physical, std::string(CHECKPOINT_RUNTIME_DIR) + "/shaders",
                        requiredWidth, fullSubgroups);
        const auto fixture = [&](uint32_t width, uint32_t height,
                                 const std::vector<uint32_t>& counts, uint32_t pattern,
                                 bool overflow = false) {
            runFixture(*device, physical, queue, family, state, width, height, counts, pattern,
                       overflow, &warp, requiredWidth, fullSubgroups);
        };
        fixture(1, 1, {0}, 0);
        for (uint32_t length : {1u, 31u, 32u, 33u, 63u, 64u, 65u, 1025u})
            fixture(1, 1, {length}, 4);
        fixture(17, 19, {0, 33, 65, 63}, 1);
        fixture(17, 19, {65, 63, 33, 0}, 2);
        fixture(1, 1, {65}, 3);
        fixture(16, 16, {1500}, 0);
        fixture(17, 19, {33, 65, 65, 63}, 5);
        std::vector<uint32_t> counts(513);
        for (uint32_t i = 0; i < counts.size(); ++i)
            counts[i] = i * 17u % 97u;
        fixture(16u * 513u, 16, counts, 1);
        fixture(1, 1, {65}, 4, true);
        warp.cleanup();
        state.cleanup();
        device->waitIdle();
        check(validationErrors == 0u, "GPU checkpoint validation reported errors");
        std::cout << "P2 GPU gradients/state/tasks/fallback checks passed; target-scene cost "
                     "remains unmeasured.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Training GPU diagnostics failed: " << error.what() << '\n';
        return 1;
    }
}
