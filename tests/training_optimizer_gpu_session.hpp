#pragma once
#include "training/core/backward/backward_command_utils.hpp"
#include "training/core/training_optimizer_capture_gpu.hpp"
#include "vulkan/compute_pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <vector>

using namespace vulkan3DGS;
namespace optimizer_gpu {
inline void check(bool okay, const char* message) {
    if (!okay) throw std::runtime_error(message);
}
struct Sample {
    double kernelMs = 0, restoreMs = 0, cpuSubmitWaitMs = 0, restoreCpuMs = 0;
};
enum class Variant : size_t { Baseline = 0, Candidate = 1 };
struct Difference {
    uint64_t words = 0, nonFinite = 0, parameterUpdates = 0;
    std::array<uint64_t, 4> fields{}; // params, m, v, densification (including padding)
    uint64_t firstBlock = UINT64_MAX, firstWord = UINT64_MAX;
    uint32_t firstActualBits = 0, firstReferenceBits = 0;
    double maxParameterUpdate = 0;
};
inline void requireMatch(const Difference& difference, const char* label) {
    if (!difference.words && !difference.nonFinite) return;
    throw std::runtime_error(
        std::string(label) + ": params=" + std::to_string(difference.fields[0]) + ", m=" +
        std::to_string(difference.fields[1]) + ", v=" + std::to_string(difference.fields[2]) +
        ", densification=" + std::to_string(difference.fields[3]) +
        ", nonfinite=" + std::to_string(difference.nonFinite) + ", first block/word=" +
        std::to_string(difference.firstBlock) + "/" + std::to_string(difference.firstWord) +
        ", actual/reference uint32=" + std::to_string(difference.firstActualBits) + "/" +
        std::to_string(difference.firstReferenceBits));
}
// Only owns private seed/work buffers. Every invocation restores all three mutable arrays.
class Session {
public:
    Session(vk::Device device, vk::PhysicalDevice physical, vk::Queue queue, uint32_t family,
            OptimizerCaptureHeader header, const OptimizerCaptureSource& source,
            const std::filesystem::path& shader, const std::filesystem::path& candidate = {})
        : device_(device), queue_(queue), h_(header), transfer_(device, physical, queue, family) {
        const auto limits = physical.getProperties().limits;
        checkOptimizerCaptureBudget(physical, header.gaussianCount,
                                    uint64_t(header.gaussianCount) * 1576);
        const auto usage = vk::BufferUsageFlagBits::eStorageBuffer |
                           vk::BufferUsageFlagBits::eTransferSrc |
                           vk::BufferUsageFlagBits::eTransferDst;
        for (size_t i = 0; i < seed_.size(); ++i)
            seed_[i].create(
                device, physical, queue, family, nullptr,
                std::max<uint64_t>(16, uint64_t(header.gaussianCount) * header.strides[i]), usage,
                vk::MemoryPropertyFlagBits::eDeviceLocal, {family});
        constexpr std::array<size_t, 3> mutableSeeds{0, 1, 4};
        for (size_t i = 0; i < work_.size(); ++i)
            work_[i].create(device, physical, queue, family, nullptr,
                            seed_[mutableSeeds[i]].getSize(), usage,
                            vk::MemoryPropertyFlagBits::eDeviceLocal, {family});
        camera_.create(device, physical, queue, family, &header.camera, sizeof(header.camera),
                       vk::BufferUsageFlagBits::eUniformBuffer,
                       vk::MemoryPropertyFlagBits::eHostVisible |
                           vk::MemoryPropertyFlagBits::eHostCoherent,
                       {family});
        std::vector<std::byte> chunk(optimizerCaptureChunkBytes);
        for (size_t block = 0; block < seed_.size(); ++block) {
            const uint64_t length = uint64_t(header.gaussianCount) * header.strides[block];
            for (uint64_t offset = 0; offset < length;) {
                auto dest = std::span(chunk).first(
                    size_t(std::min<uint64_t>(chunk.size(), length - offset)));
                source(block, offset, dest);
                transfer_.upload(seed_[block].getDescriptorInfo(), offset, dest);
                offset += dest.size();
            }
        }
        infos_ = {work_[0].getDescriptorInfo(), work_[1].getDescriptorInfo(),
                  seed_[2].getDescriptorInfo(), seed_[3].getDescriptorInfo(),
                  work_[2].getDescriptorInfo(), seed_[5].getDescriptorInfo(),
                  camera_.getDescriptorInfo()};
        ComputePipelineConfig config;
        for (auto binding : optimizerCaptureBindings)
            config.descriptorBindings.push_back(backwardStorageBinding(binding));
        config.descriptorBindings.push_back(backwardUniformBinding(11));
        config.pushConstantSize = sizeof(TrainingPushConstants);
        pipeline_[0].initialize(device, shader.string(), config);
        std::vector<vk::DescriptorSetLayout> layouts{pipeline_[0].getDescriptorSetLayout()};
        if (!candidate.empty()) {
            pipeline_[1].initialize(device, candidate.string(), config);
            layouts.push_back(pipeline_[1].getDescriptorSetLayout());
        }
        sets_ = descriptors_.create(device, uint32_t(layouts.size()) * 6, uint32_t(layouts.size()),
                                    layouts);
        for (auto set : sets_) {
            std::array<vk::WriteDescriptorSet, 7> writes;
            for (size_t i = 0; i < writes.size(); ++i)
                writes[i]
                    .setDstSet(set)
                    .setDstBinding(i == 6 ? 11 : optimizerCaptureBindings[i])
                    .setDescriptorCount(1)
                    .setDescriptorType(i == 6 ? vk::DescriptorType::eUniformBuffer
                                              : vk::DescriptorType::eStorageBuffer)
                    .setPBufferInfo(&infos_[i]);
            device.updateDescriptorSets(writes, {});
        }
        pool_.create(device, family, vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
        command_ = pool_.allocateCommandBuffer();
        fence_ = device.createFenceUnique({});
        bits_ = physical.getQueueFamilyProperties()[family].timestampValidBits;
        period_ = limits.timestampPeriod;
        if (bits_) queries_ = device.createQueryPoolUnique({{}, vk::QueryType::eTimestamp, 4});
    }
    bool timestampsAvailable() const {
        return bits_ != 0;
    }
    Sample dispatch(bool restoreMoments = true, Variant variant = Variant::Baseline) {
        const size_t selected = size_t(variant);
        check(selected < sets_.size(), "S4 candidate pipeline was not requested");
        command_.reset();
        command_.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        barrier(vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite);
        if (queries_) {
            command_.resetQueryPool(*queries_, 0, 4);
            command_.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, *queries_, 0);
        }
        constexpr std::array<size_t, 3> input{0, 1, 4};
        for (size_t i = 0; i < 3; ++i) {
            if (!restoreMoments && i == 1)
                continue; // Deliberate negative control, never timed baseline.
            const uint64_t size = uint64_t(h_.gaussianCount) * h_.strides[input[i]];
            if (size)
                command_.copyBuffer(seed_[input[i]].getBuffer(), work_[i].getBuffer(),
                                    vk::BufferCopy{0, 0, size});
        }
        barrier(vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
        if (queries_)
            command_.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, *queries_, 1);
        command_.end();
        Sample sample;
        // Fence the restore separately: TOP timestamps of a later dispatch in the
        // same command buffer could otherwise overlap the transfer stage.
        sample.restoreCpuMs = submitAndWait();
        command_.reset();
        command_.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        command_.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline_[selected].getPipeline());
        command_.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                    pipeline_[selected].getPipelineLayout(), 0, sets_[selected],
                                    {});
        command_.pushConstants(pipeline_[selected].getPipelineLayout(),
                               vk::ShaderStageFlagBits::eCompute, 0, sizeof(h_.push), &h_.push);
        if (queries_) command_.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, *queries_, 2);
        if (h_.gaussianCount) command_.dispatch((h_.gaussianCount + 255u) / 256u, 1, 1);
        if (queries_)
            command_.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, *queries_, 3);
        barrier(vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eShaderRead |
                vk::AccessFlagBits::eShaderWrite);
        command_.end();
        sample.cpuSubmitWaitMs = submitAndWait();
        if (queries_) {
            std::array<uint64_t, 4> ticks{};
            check(device_.getQueryPoolResults(
                      *queries_, 0, 4, sizeof(ticks), ticks.data(), sizeof(uint64_t),
                      vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait) ==
                      vk::Result::eSuccess,
                  "S4 timestamp read failed");
            const uint64_t mask = bits_ >= 64 ? UINT64_MAX : (uint64_t(1) << bits_) - 1;
            sample.restoreMs = double((ticks[1] - ticks[0]) & mask) * period_ / 1e6;
            sample.kernelMs = double((ticks[3] - ticks[2]) & mask) * period_ / 1e6;
        }
        return sample;
    }
    void readOutput(size_t block, uint64_t offset, std::span<std::byte> dest) {
        check(block >= 6 && block <= 8, "S4 output block index");
        transfer_.download(work_[block - 6].getDescriptorInfo(), offset, dest);
    }
    Difference compare(const OptimizerCaptureSource& reference) {
        Difference result;
        std::vector<std::byte> actual(optimizerCaptureChunkBytes), expected(actual.size()),
            before(actual.size());
        for (size_t block = 6; block < 9; ++block) {
            const uint64_t bytes = uint64_t(h_.gaussianCount) * h_.strides[block];
            for (uint64_t offset = 0; offset < bytes;) {
                const size_t n = size_t(std::min<uint64_t>(actual.size(), bytes - offset));
                auto got = std::span(actual).first(n), want = std::span(expected).first(n);
                readOutput(block, offset, got);
                reference(block, offset, want);
                if (block == 6)
                    transfer_.download(seed_[0].getDescriptorInfo(), offset,
                                       std::span(before).first(n));
                for (size_t i = 0; i < n; i += 4) {
                    if (std::memcmp(got.data() + i, want.data() + i, 4)) {
                        ++result.words;
                        const size_t field = block == 6   ? 0
                                             : block == 8 ? 3
                                                          : ((offset + i) % 480 < 240 ? 1 : 2);
                        ++result.fields[field];
                        if (result.firstBlock == UINT64_MAX) {
                            result.firstBlock = block;
                            result.firstWord = (offset + i) / 4;
                            std::memcpy(&result.firstActualBits, got.data() + i, 4);
                            std::memcpy(&result.firstReferenceBits, want.data() + i, 4);
                        }
                    }
                    // Densification contains uint fields. Only its float lanes are finite-checked.
                    const bool floating =
                        block != 8 || (offset + i) % 16 == 0 || (offset + i) % 16 == 8;
                    float a, b;
                    std::memcpy(&a, got.data() + i, 4);
                    std::memcpy(&b, want.data() + i, 4);
                    if (floating && (!std::isfinite(a) || !std::isfinite(b))) ++result.nonFinite;
                    if (block == 6) {
                        float old;
                        std::memcpy(&old, before.data() + i, 4);
                        if (a != old) ++result.parameterUpdates;
                        result.maxParameterUpdate =
                            std::max(result.maxParameterUpdate, std::abs(double(a) - old));
                    }
                }
                offset += n;
            }
        }
        return result;
    }
    void verifyReadonly() {
        // Hash immutable GPU seeds after all runs, including params/m/v/densification seeds.
        std::vector<std::byte> bytes(optimizerCaptureChunkBytes);
        for (size_t block = 0; block < 6; ++block) {
            uint64_t hash = optimizerHashSeed;
            const uint64_t length = uint64_t(h_.gaussianCount) * h_.strides[block];
            for (uint64_t offset = 0; offset < length;) {
                auto chunk = std::span(bytes).first(
                    size_t(std::min<uint64_t>(bytes.size(), length - offset)));
                transfer_.download(seed_[block].getDescriptorInfo(), offset, chunk);
                hash = optimizerHash(chunk, hash);
                offset += chunk.size();
            }
            check(hash == h_.blocks[block].checksum, "S4 readonly seed/input modified");
        }
    }

private:
    double submitAndWait() {
        device_.resetFences(*fence_);
        vk::SubmitInfo submit;
        submit.setCommandBuffers(command_);
        const auto start = std::chrono::steady_clock::now();
        queue_.submit(submit, *fence_);
        (void)device_.waitForFences(*fence_, true, UINT64_MAX);
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
            .count();
    }
    void barrier(vk::AccessFlags destination) {
        vk::MemoryBarrier b{vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite |
                                vk::AccessFlagBits::eTransferRead |
                                vk::AccessFlagBits::eTransferWrite | vk::AccessFlagBits::eHostWrite,
                            destination};
        command_.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands |
                                     vk::PipelineStageFlagBits::eHost,
                                 vk::PipelineStageFlagBits::eAllCommands, {}, b, {}, {});
    }
    vk::Device device_;
    vk::Queue queue_;
    OptimizerCaptureHeader h_;
    OptimizerCaptureTransfer transfer_;
    std::array<Buffer, 6> seed_;
    std::array<Buffer, 3> work_;
    Buffer camera_;
    std::array<vk::DescriptorBufferInfo, 7> infos_;
    std::array<ComputePipeline, 2> pipeline_;
    BackwardDescriptorPool descriptors_;
    std::vector<vk::DescriptorSet> sets_;
    CommandPool pool_;
    vk::CommandBuffer command_;
    vk::UniqueFence fence_;
    vk::UniqueQueryPool queries_;
    uint32_t bits_ = 0;
    float period_ = 0;
};
} // namespace optimizer_gpu
