#pragma once
#include "training_full_sort_session.hpp"
#include "training_sort_downstream_reference.hpp"

namespace full_sort {
inline bool supportsDownstreamTensor(vk::PhysicalDevice physical) {
    vk::PhysicalDeviceSubgroupProperties subgroup;
    vk::PhysicalDeviceProperties2 properties;
    properties.pNext = &subgroup;
    physical.getProperties2(&properties);
    const auto& limits = properties.properties.limits;
    return subgroup.subgroupSize == 32 &&
           (subgroup.supportedStages & vk::ShaderStageFlagBits::eCompute) &&
           (subgroup.supportedOperations & vk::SubgroupFeatureFlagBits::eBasic) &&
           (subgroup.supportedOperations & vk::SubgroupFeatureFlagBits::eShuffle) &&
           limits.maxComputeSharedMemorySize >= 45u * 1024u &&
           limits.maxComputeWorkGroupInvocations >= 256 &&
           limits.maxComputeWorkGroupSize[0] >= 16 && limits.maxComputeWorkGroupSize[1] >= 16;
}
// Consumes the producer's live GPU outputs in the same command buffer; owns only downstream
// workspace.
class DownstreamSession {
public:
    DownstreamSession(vk::Device device, vk::PhysicalDevice physical, vk::Queue queue,
                      uint32_t family, const OutputView& view,
                      const TrainingScanlineCapture& source, bool portable)
        : device_(device), source_(source), transfer_(device, physical, queue, family) {
        const auto& h = source.header;
        const uint64_t pixels = uint64_t(h.width) * h.height;
        check(pixels && pixels <= (1u << 22) && source.pixelGrads.size() == pixels,
              "Downstream pixel-gradient dimensions/budget");
        check(portable || supportsDownstreamTensor(physical),
              "Downstream Tensor requires native32 and original capability gate");
        const auto limits = physical.getProperties().limits;
        check((h.width + 15ull) / 16 <= limits.maxComputeWorkGroupCount[0] &&
                  (h.height + 15ull) / 16 <= limits.maxComputeWorkGroupCount[1] &&
                  limits.maxComputeWorkGroupInvocations >= 256 &&
                  limits.maxComputeWorkGroupSize[0] >= 16 &&
                  limits.maxComputeWorkGroupSize[1] >= 16 &&
                  limits.maxComputeSharedMemorySize >= 16u * 1024u,
              "Downstream workgroup/device limits");
        sizes_ = {pixels * 16, pixels * 16, uint64_t(std::max(h.gaussianCount, 1u)) * 48,
                  pixels * 16};
        for (auto bytes : sizes_) {
            check(bytes + 64 <= limits.maxStorageBufferRange, "Downstream descriptor range");
            logicalBytes_ += bytes + 64;
        }
        check(availableSystemMemoryBytes() >= logicalBytes_ * 3 + (256ull << 20),
              "Downstream host memory budget");
        checkOptimizerCaptureBudget(physical, 0, logicalBytes_ + (1ull << 20));
        for (const auto& pixel : source.pixelGrads)
            for (int component = 0; component < 4; ++component)
                check(std::isfinite(pixel.color[component]),
                      "Non-finite captured pixel gradient; no cleaning allowed");
        for (size_t i = 0; i < buffers_.size(); ++i)
            buffers_[i].create(device, physical, queue, family, nullptr, sizes_[i] + 64,
                               vk::BufferUsageFlagBits::eStorageBuffer |
                                   vk::BufferUsageFlagBits::eTransferSrc |
                                   vk::BufferUsageFlagBits::eTransferDst,
                               vk::MemoryPropertyFlagBits::eDeviceLocal, {family});
        std::vector<std::byte> pixelsSeed(sizes_[3] + 64, std::byte{0xcd});
        std::memcpy(pixelsSeed.data(), source.pixelGrads.data(), sizes_[3]);
        uploadSortReplay(transfer_, buffers_[3].getDescriptorInfo(), pixelsSeed);
        const std::array<std::vector<uint32_t>, 2> bindings{
            {{3, 4, 5, 6, 14}, {3, 4, 5, 12, 13, 14}}};
        const std::array<const char*, 2> names{
            "train_forward_workgroup", portable
                                           ? "train_backward_pixel_to_2dgs"
                                           : "train_backward_vksplat_tensor_subgroup_global_grad"};
        std::vector<vk::DescriptorSetLayout> layouts;
        for (size_t i = 0; i < 2; ++i) {
            ComputePipelineConfig config;
            config.pushConstantSize = sizeof(TrainingPushConstants);
            for (auto binding : bindings[i])
                config.descriptorBindings.push_back(backwardStorageBinding(binding));
            pipelines_[i].initialize(device, std::string("shaders/") + names[i] + ".comp.spv",
                                     config);
            layouts.push_back(pipelines_[i].getDescriptorSetLayout());
        }
        const auto sets = descriptors_.create(device, 11, 0, layouts);
        std::copy(sets.begin(), sets.end(), sets_.begin());
        const auto outputInfo = [&](size_t i) {
            auto info = buffers_[i].getDescriptorInfo();
            info.range = sizes_[i];
            return info;
        };
        const std::array<std::vector<vk::DescriptorBufferInfo>, 2> infos{
            {{view.projected, view.items, view.ranges, outputInfo(0), outputInfo(1)},
             {view.projected, view.items, view.ranges, outputInfo(3), outputInfo(2),
              outputInfo(1)}}};
        for (size_t i = 0; i < 2; ++i) {
            std::vector<vk::WriteDescriptorSet> writes(bindings[i].size());
            for (size_t j = 0; j < writes.size(); ++j)
                writes[j]
                    .setDstSet(sets_[i])
                    .setDstBinding(bindings[i][j])
                    .setDescriptorCount(1)
                    .setDescriptorType(vk::DescriptorType::eStorageBuffer)
                    .setPBufferInfo(&infos[i][j]);
            device.updateDescriptorSets(writes, {});
        }
        bound_ = view;
    }
    uint64_t logicalBufferBytes() const {
        return logicalBytes_;
    }
    void record(vk::CommandBuffer command, const OutputView& view) {
        check(view.projected == bound_.projected && view.items == bound_.items &&
                  view.ranges == bound_.ranges,
              "Downstream borrowed producer buffers changed");
        const auto& h = source_.header;
        check(view.push.gaussianCount == h.gaussianCount &&
                  view.push.tileItemCount == h.itemCount && view.push.width == h.width &&
                  view.push.height == h.height,
              "Downstream producer push mismatch");
        for (size_t i = 0; i < 3; ++i) {
            command.fillBuffer(buffers_[i].getBuffer(), 0, sizes_[i], i == 2 ? 0u : 0x7fc00000u);
            command.fillBuffer(buffers_[i].getBuffer(), sizes_[i], 64, canary);
        }
        vk::MemoryBarrier ready{vk::AccessFlagBits::eTransferWrite |
                                    vk::AccessFlagBits::eShaderWrite,
                                vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite};
        command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer |
                                    vk::PipelineStageFlagBits::eComputeShader,
                                vk::PipelineStageFlagBits::eComputeShader, {}, ready, {}, {});
        auto push = view.push;
        push.trainingIteration = h.iteration;
        push.optimizerEnabled = 0;
        for (size_t i = 0; i < 2; ++i) {
            auto& pipeline = pipelines_[i];
            command.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.getPipeline());
            command.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                       pipeline.getPipelineLayout(), 0, sets_[i], {});
            command.pushConstants(pipeline.getPipelineLayout(), vk::ShaderStageFlagBits::eCompute,
                                  0, sizeof(push), &push);
            command.dispatch((h.width + 15) / 16, (h.height + 15) / 16, 1);
            vk::MemoryBarrier dependency{
                vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
                vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite |
                    vk::AccessFlagBits::eTransferRead};
            command.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                    vk::PipelineStageFlagBits::eComputeShader |
                                        vk::PipelineStageFlagBits::eTransfer,
                                    {}, dependency, {}, {});
        }
    }
    DownstreamSnapshot snapshot() {
        DownstreamSnapshot result;
        result.color.resize(source_.pixelGrads.size());
        result.blends.resize(source_.pixelGrads.size());
        result.gradients.resize(source_.header.gaussianCount);
        downloadSortReplay(transfer_, buffers_[0].getDescriptorInfo(),
                           std::as_writable_bytes(std::span(result.color)));
        downloadSortReplay(transfer_, buffers_[1].getDescriptorInfo(),
                           std::as_writable_bytes(std::span(result.blends)));
        downloadSortReplay(transfer_, buffers_[2].getDescriptorInfo(),
                           std::as_writable_bytes(std::span(result.gradients)));
        for (size_t i = 0; i < 4; ++i) {
            std::array<uint32_t, guardWords> words{};
            transfer_.download(buffers_[i].getDescriptorInfo(), sizes_[i],
                               std::as_writable_bytes(std::span(words)));
            check(std::all_of(words.begin(), words.end(), [](auto v) { return v == canary; }),
                  "Downstream canary changed");
        }
        std::vector<PixelGrad> pixels(source_.pixelGrads.size());
        downloadSortReplay(transfer_, buffers_[3].getDescriptorInfo(),
                           std::as_writable_bytes(std::span(pixels)));
        check(std::memcmp(pixels.data(), source_.pixelGrads.data(), sizes_[3]) == 0,
              "Readonly pixel gradients changed");
        for (const auto& blend : result.blends)
            check(std::isfinite(blend.finalTransmittance) && blend.finalTransmittance >= 0 &&
                      blend.finalTransmittance <= 1 && std::isfinite(blend.accumulatedAlpha) &&
                      blend.processedCount <= source_.header.itemCount &&
                      blend.contributionCount <= blend.processedCount,
                  "Downstream uninitialized/invalid blend state");
        const auto self = compareDownstream(result, result);
        check(self.passed(), "Downstream non-finite gradient or nonzero gradient padding");
        return result;
    }

private:
    vk::Device device_;
    const TrainingScanlineCapture& source_;
    OptimizerCaptureTransfer transfer_;
    std::array<Buffer, 4> buffers_;
    std::array<uint64_t, 4> sizes_{};
    uint64_t logicalBytes_ = 0;
    std::array<ComputePipeline, 2> pipelines_;
    BackwardDescriptorPool descriptors_;
    std::array<vk::DescriptorSet, 2> sets_{};
    OutputView bound_;
};
} // namespace full_sort
