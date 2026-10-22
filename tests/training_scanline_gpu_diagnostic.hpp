#pragma once
#include "training/core/backward/backward_command_utils.hpp"
#include "training/core/training_scanline_capture.hpp"
#include "vulkan/buffer.hpp"
#include "vulkan/command_pool.hpp"
#include "vulkan/compute_pipeline.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>

struct ScanlineGpuResult {
    std::array<uint64_t, 32> counts{};
    std::array<uint64_t, 8> rowEmptyCounts{};
    bool tensorReference = false;
    bool timingAvailable = false;
    std::vector<double> boundsMs, tensorBeforeMs, tensorAfterMs;
};
inline double scanlineMedian(std::vector<double> values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    return (values[(values.size() - 1u) / 2u] + values[values.size() / 2u]) * 0.5;
}

inline ScanlineGpuResult runScanlineGpuDiagnostic(vk::Device device, vk::PhysicalDevice physical,
                                                  vk::Queue queue, uint32_t family,
                                                  const vulkan3DGS::TrainingScanlineCapture& data,
                                                  bool rowEmptyR0 = false) {
    using namespace vulkan3DGS;
    validateTrainingScanlineCapture(data);
    const auto& h = data.header;
    const auto properties = physical.getProperties();
    const auto& limits = properties.limits;
    const uint32_t tx = static_cast<uint32_t>((uint64_t(h.width) + 15u) / 16u);
    const uint32_t ty = static_cast<uint32_t>((uint64_t(h.height) + 15u) / 16u);
    if (tx > limits.maxComputeWorkGroupCount[0] || ty > limits.maxComputeWorkGroupCount[1] ||
        limits.maxComputeWorkGroupInvocations < 256u || limits.maxComputeWorkGroupSize[0] < 16u ||
        limits.maxComputeWorkGroupSize[1] < 16u || limits.maxComputeSharedMemorySize < 12u * 1024u)
        throw std::runtime_error("GPU cannot execute the S3-2 diagnostic workgroup/dispatch");
    std::array<Buffer, 7> buffers;
    const std::array<uint64_t, 7> sizes = {data.projected.size() * sizeof(ProjectedGaussian),
                                           data.items.size() * sizeof(uint32_t),
                                           data.ranges.size() * sizeof(glm::uvec4),
                                           data.blends.size() * sizeof(PixelBlendState),
                                           uint64_t(h.tileCount) *
                                               (rowEmptyR0 ? 40u : 32u) * sizeof(uint32_t),
                                           data.pixelGrads.size() * sizeof(PixelGrad),
                                           data.projected.size() * sizeof(ProjectedGaussianGrad)};
    const std::array<const void*, 7> sources = {data.projected.data(),
                                                data.items.data(),
                                                data.ranges.data(),
                                                data.blends.data(),
                                                nullptr,
                                                data.pixelGrads.data(),
                                                nullptr};
    for (size_t i = 0; i < buffers.size(); ++i) {
        if (sizes[i] > limits.maxStorageBufferRange)
            throw std::runtime_error("S3-2 input exceeds device storage-buffer range");
        buffers[i].create(device, physical, queue, family, sizes[i] ? sources[i] : nullptr,
                          std::max<uint64_t>(sizes[i], 16u),
                          vk::BufferUsageFlagBits::eStorageBuffer |
                              vk::BufferUsageFlagBits::eTransferSrc |
                              vk::BufferUsageFlagBits::eTransferDst,
                          vk::MemoryPropertyFlagBits::eDeviceLocal, {family});
    }
    vk::PhysicalDeviceSubgroupProperties subgroup{};
    vk::PhysicalDeviceProperties2 props{};
    props.pNext = &subgroup;
    physical.getProperties2(&props);
    if (rowEmptyR0 && (!(subgroup.supportedStages & vk::ShaderStageFlagBits::eCompute) ||
                      !(subgroup.supportedOperations & vk::SubgroupFeatureFlagBits::eBasic) ||
                      !(subgroup.supportedOperations & vk::SubgroupFeatureFlagBits::eVote)))
        throw std::runtime_error("S3-3 R0 native-wave statistics require compute BASIC/VOTE");
    ScanlineGpuResult result;
    result.tensorReference = !rowEmptyR0 &&
        subgroup.subgroupSize == 32u &&
        (subgroup.supportedStages & vk::ShaderStageFlagBits::eCompute) &&
        (subgroup.supportedOperations & vk::SubgroupFeatureFlagBits::eBasic) &&
        (subgroup.supportedOperations & vk::SubgroupFeatureFlagBits::eShuffle) &&
        limits.maxComputeSharedMemorySize >= 45u * 1024u;
    const uint32_t timestampBits = physical.getQueueFamilyProperties()[family].timestampValidBits;
    result.timingAvailable = !rowEmptyR0 && timestampBits != 0u;
    vk::UniqueQueryPool queries;
    if (result.timingAvailable)
        queries = device.createQueryPoolUnique({{}, vk::QueryType::eTimestamp, 2u});
    std::array<ComputePipeline, 3> pipelines;
    ComputePipelineConfig config;
    for (uint32_t binding : {3u, 4u, 5u, 14u, 51u})
        config.descriptorBindings.push_back(backwardStorageBinding(binding));
    config.pushConstantSize = sizeof(TrainingPushConstants);
    const auto shaderRoot = std::string(CHECKPOINT_RUNTIME_DIR) + "/shaders/";
    pipelines[0].initialize(device, shaderRoot + (rowEmptyR0 ? "train_row_empty_census.comp.spv" : "train_scanline_census.comp.spv"), config);
    if (!rowEmptyR0)
        pipelines[1].initialize(device, shaderRoot + "train_scanline_bounds_cost.comp.spv", config);
    if (result.tensorReference) {
        config.descriptorBindings.clear();
        for (uint32_t binding : {3u, 4u, 5u, 12u, 13u, 14u})
            config.descriptorBindings.push_back(backwardStorageBinding(binding));
        pipelines[2].initialize(
            device, shaderRoot + "train_backward_vksplat_tensor_subgroup_global_grad.comp.spv",
            config);
    }
    std::vector<vk::DescriptorSetLayout> layouts = {pipelines[0].getDescriptorSetLayout()};
    if (!rowEmptyR0) layouts.push_back(pipelines[1].getDescriptorSetLayout());
    if (result.tensorReference) layouts.push_back(pipelines[2].getDescriptorSetLayout());
    BackwardDescriptorPool descriptors;
    const auto sets = descriptors.create(device, 16u, 0u, layouts);
    std::array<vk::DescriptorBufferInfo, 7> infos;
    for (size_t i = 0; i < buffers.size(); ++i)
        infos[i] = buffers[i].getDescriptorInfo();
    for (size_t p = 0; p < layouts.size(); ++p) {
        const std::vector<uint32_t> bindings = p < 2u ? std::vector<uint32_t>{3, 4, 5, 14, 51}
                                                      : std::vector<uint32_t>{3, 4, 5, 12, 13, 14};
        const std::vector<size_t> indices =
            p < 2u ? std::vector<size_t>{0, 1, 2, 3, 4} : std::vector<size_t>{0, 1, 2, 5, 6, 3};
        std::vector<vk::WriteDescriptorSet> writes(bindings.size());
        for (size_t i = 0; i < bindings.size(); ++i)
            writes[i]
                .setDstSet(sets[p])
                .setDstBinding(bindings[i])
                .setDescriptorCount(1)
                .setDescriptorType(vk::DescriptorType::eStorageBuffer)
                .setPBufferInfo(&infos[indices[i]]);
        device.updateDescriptorSets(writes, {});
    }
    CommandPool pool;
    pool.create(device, family, vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
    const auto command = pool.allocateCommandBuffer();
    auto fence = device.createFenceUnique({});
    TrainingPushConstants params{};
    params.gaussianCount = h.gaussianCount;
    params.width = h.width;
    params.height = h.height;
    params.pixelCount = h.width * h.height;
    params.tileSize = 16u;
    params.tileItemCount = h.itemCount;
    auto dispatch = [&](size_t p) {
        command.reset();
        command.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        vk::MemoryBarrier initial{
            vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite |
                vk::AccessFlagBits::eTransferWrite,
            vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite |
                vk::AccessFlagBits::eTransferWrite};
        command.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                                vk::PipelineStageFlagBits::eAllCommands, {}, initial, {}, {});
        if (p == 2u) {
            command.fillBuffer(buffers[6].getBuffer(), 0u, VK_WHOLE_SIZE, 0u);
            vk::MemoryBarrier cleared{vk::AccessFlagBits::eTransferWrite,
                                      vk::AccessFlagBits::eShaderRead |
                                          vk::AccessFlagBits::eShaderWrite};
            command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                    vk::PipelineStageFlagBits::eComputeShader, {}, cleared, {}, {});
        }
        if (queries) {
            command.resetQueryPool(*queries, 0, 2);
            command.writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, *queries, 0);
        }
        command.bindPipeline(vk::PipelineBindPoint::eCompute, pipelines[p].getPipeline());
        command.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                   pipelines[p].getPipelineLayout(), 0, sets[p], {});
        command.pushConstants(pipelines[p].getPipelineLayout(), vk::ShaderStageFlagBits::eCompute,
                              0, sizeof(params), &params);
        command.dispatch(tx, ty, 1u);
        recordBackwardBufferBarrier(command, {infos[p == 2u ? 6u : 4u]},
                                    vk::AccessFlagBits::eShaderRead |
                                        vk::AccessFlagBits::eShaderWrite |
                                        vk::AccessFlagBits::eTransferRead);
        if (queries) command.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, *queries, 1);
        command.end();
        device.resetFences(*fence);
        vk::SubmitInfo submit;
        submit.setCommandBuffers(command);
        queue.submit(submit, *fence);
        (void)device.waitForFences(*fence, VK_TRUE, UINT64_MAX);
        if (!queries) return 0.0;
        std::array<uint64_t, 2> ticks{};
        const auto status = device.getQueryPoolResults(
            *queries, 0, 2, sizeof(ticks), ticks.data(), sizeof(uint64_t),
            vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);
        if (status != vk::Result::eSuccess) throw std::runtime_error("S3-2 timestamp read failed");
        const uint64_t mask =
            timestampBits == 64u ? UINT64_MAX : (uint64_t(1) << timestampBits) - 1u;
        return double((ticks[1] - ticks[0]) & mask) * double(limits.timestampPeriod) / 1e6;
    };
    // Census time is deliberately not used as a candidate performance metric.
    (void)dispatch(0);
    const size_t stride = rowEmptyR0 ? 40u : 32u;
    std::vector<uint32_t> tiles(size_t(h.tileCount) * stride);
    buffers[4].download(tiles.data(), tiles.size() * sizeof(tiles[0]));
    for (size_t tile = 0; tile < h.tileCount; ++tile) {
        for (size_t i = 0; i < result.counts.size(); ++i)
            result.counts[i] += tiles[tile * stride + i];
        if (rowEmptyR0)
            for (size_t i = 0; i < result.rowEmptyCounts.size(); ++i)
                result.rowEmptyCounts[i] += tiles[tile * stride + 32u + i];
    }
    const auto sample = [&](size_t p, std::vector<double>& samples) {
        for (uint32_t i = 0; i < 10u; ++i) {
            const auto ms = dispatch(p);
            if (i >= 2u && result.timingAvailable) samples.push_back(ms);
        }
    };
    if (result.tensorReference) sample(2u, result.tensorBeforeMs);
    if (!rowEmptyR0) sample(1u, result.boundsMs);
    if (result.tensorReference) sample(2u, result.tensorAfterMs);
    pool.freeCommandBuffer(command);
    return result;
}

inline void writeScanlineGpuReport(const std::filesystem::path& requested,
                                   vk::PhysicalDevice physical,
                                   const vulkan3DGS::TrainingScanlineCapture& data,
                                   const ScanlineGpuResult& result) {
    using namespace vulkan3DGS;
    auto path = requested;
    for (uint32_t suffix = 2; std::filesystem::exists(path); ++suffix)
        path = requested.parent_path() / (requested.stem().string() + "_" + std::to_string(suffix) +
                                          requested.extension().string());
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    const auto& c = result.counts;
    const auto ratio = [&](size_t numerator, size_t denominator) {
        return c[denominator] ? double(c[numerator]) / double(c[denominator]) : 0.0;
    };
    const double before = scanlineMedian(result.tensorBeforeMs),
                 after = scanlineMedian(result.tensorAfterMs);
    const double bounds = scanlineMedian(result.boundsMs);
    const auto props = physical.getProperties();
    const bool sameDevice = data.header.vendorID != 0u && data.header.deviceID != 0u &&
                            data.header.vendorID == props.vendorID &&
                            data.header.deviceID == props.deviceID;
    std::string verdict = "OPPORTUNITY_ONLY_REQUIRES_INTEGRATED_PROTOTYPE";
    if (c[24] || c[26])
        verdict = "STOP_INVALID_SPAN_OR_INPUT";
    else if (!c[1])
        verdict = "STOP_NO_ELIGIBLE_WORK";
    else if (!c[14] && !c[17] && !c[21])
        verdict = "STOP_CANDIDATE_RECOGNIZES_NO_WORK";
    else if (ratio(17, 1) < 0.10 && ratio(15, 3) < 0.10 && ratio(21, 1) < 0.05)
        verdict = "STOP_LOW_OPPORTUNITY_SCREEN";
    else if (!result.tensorReference || !result.timingAvailable || !sameDevice)
        verdict = "INCONCLUSIVE_NO_MATCHED_TENSOR_COST";
    else if (std::abs(before - after) > 0.15 * std::max(before, after))
        verdict = "INCONCLUSIVE_TIMING_DRIFT";
    else if (bounds > 0.25 * (before + after) * 0.5)
        verdict = "STOP_EXPENSIVE_BOUNDS_PROXY_SCREEN";
    std::ofstream file(path);
    if (!file) throw std::runtime_error("Cannot write S3-2 JSON report");
    file << std::setprecision(10)
         << "{\n  \"version\": 1,\n  \"method\": \"diagnostic_padded_spd_row_span_v1\",\n"
         << "  \"gpu\": " << std::quoted(std::string(props.deviceName.data())) << ",\n"
         << "  \"frame\": " << data.header.frame << ", \"iteration\": " << data.header.iteration
         << ",\n"
         << "  \"width\": " << data.header.width << ", \"height\": " << data.header.height << ",\n"
         << "  \"gaussians\": " << data.header.gaussianCount
         << ", \"items\": " << data.header.itemCount << ",\n"
         << "  \"source_vendor_id\": " << data.header.vendorID
         << ", \"source_device_id\": " << data.header.deviceID << ",\n"
         << "  \"source_driver\": " << data.header.driverVersion
         << ", \"current_driver\": " << props.driverVersion << ",\n"
         << "  \"tensor_reference\": " << (result.tensorReference ? "true" : "false") << ",\n"
         << "  \"same_device_ids\": " << (sameDevice ? "true" : "false") << ",\n"
         << "  \"timing_available\": " << (result.timingAvailable ? "true" : "false") << ",\n"
         << "  \"counters\": {\n";
    for (size_t i = 0; i < c.size(); ++i)
        file << "    " << std::quoted(kScanlineCounterNames[i]) << ": " << c[i]
             << (i + 1u < c.size() ? ",\n" : "\n");
    file << "  },\n  \"ratios\": {\"oracle_empty_row_pairs\": " << ratio(7, 1)
         << ", \"oracle_empty_gaussian_pairs\": " << ratio(10, 1)
         << ", \"oracle_empty_batch_pairs\": " << ratio(13, 1)
         << ", \"span_rejected_eligible_pairs\": " << ratio(14, 1)
         << ", \"span_rejected_old_exp_pairs\": " << ratio(15, 3)
         << ", \"span_empty_row_pairs\": " << ratio(17, 1)
         << ", \"span_empty_gaussian_pairs\": " << ratio(19, 1)
         << ", \"span_empty_batch_pairs\": " << ratio(21, 1) << "},\n";
    const auto timing = [&](const char* name, const std::vector<double>& samples) {
        file << "  " << std::quoted(std::string(name) + "_median_ms") << ": ";
        if (samples.empty())
            file << "null";
        else
            file << scanlineMedian(samples);
        file << ",\n  " << std::quoted(std::string(name) + "_samples_ms") << ": [";
        for (size_t i = 0; i < samples.size(); ++i)
            file << (i ? "," : "") << samples[i];
        file << "],\n";
    };
    timing("bounds_proxy_not_net_overhead", result.boundsMs);
    timing("tensor_before", result.tensorBeforeMs);
    timing("tensor_after", result.tensorAfterMs);
    file << "  \"timing_warmup\": 2, \"timing_samples\": 8,\n"
         << "  \"integration_resource_screen\": "
            "{\"reported_tensor_shared_bytes\":33540,\"naive_span_table_bytes\":1024,\"naive_"
            "coefficients_bytes\":256,\"naive_integrated_shared_bytes\":34820,\"three_blocks_raw_"
            "bytes\":104460,\"ampere_86_sm_shared_capacity_model_bytes\":102400,\"actual_vulkan_"
            "residency_measured\":false},\n"
         << "  \"screening_heuristics\": "
            "{\"min_row_or_exp_fraction\":0.10,\"min_batch_fraction\":0.05,\"max_proxy_to_tensor_"
            "fraction\":0.25,\"max_baseline_drift_fraction\":0.15},\n"
         << "  \"verdict\": " << std::quoted(verdict) << ",\n"
         << "  \"production_kernel_implemented\": false,\n  \"estimated_speedup\": null,\n"
         << "  \"limitations\": \"Screens are heuristics, not a speedup proof. Bounds timing "
            "includes standalone staging/barriers/hash; excludes mask application. Oracle and span "
            "savings overlap and must not be added. Naive shared-span caching risks losing "
            "residency; resource fields are an arithmetic screen, not a native allocation "
            "measurement. No training gradients are modified.\"\n}\n";
    file.flush();
    if (!file) throw std::runtime_error("Failed writing S3-2 report");
    std::cout << "S3-2 " << verdict << ": eligible=" << c[1]
              << " oracle-row-pair-fraction=" << ratio(7, 1)
              << " oracle-batch-pair-fraction=" << ratio(13, 1) << " false-drops=" << c[24]
              << " report=" << path.string() << '\n';
}
