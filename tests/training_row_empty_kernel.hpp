#pragma once
#include "training_row_empty_metrics.hpp"
#include "training_row_empty_spirv.hpp"
#include "training_scanline_gpu_diagnostic.hpp"
#include <atomic>

inline bool supportsRowEmptyTensor(vk::PhysicalDevice physical) {
    vk::PhysicalDeviceSubgroupProperties subgroup{};
    vk::PhysicalDeviceProperties2 properties{};
    properties.pNext = &subgroup;
    physical.getProperties2(&properties);
    const auto& limits = properties.properties.limits;
    return subgroup.subgroupSize == 32u &&
           (subgroup.supportedStages & vk::ShaderStageFlagBits::eCompute) &&
           (subgroup.supportedOperations & vk::SubgroupFeatureFlagBits::eBasic) &&
           (subgroup.supportedOperations & vk::SubgroupFeatureFlagBits::eShuffle) &&
           limits.maxComputeSharedMemorySize >= 45u * 1024u &&
           limits.maxComputeWorkGroupInvocations >= 256u &&
           limits.maxComputeWorkGroupSize[0] >= 16u && limits.maxComputeWorkGroupSize[1] >= 16u;
}

// Private offline buffers only. portable=true is a batch4/Shared fixture,
// explicitly NOT a fallback or a production-size Tensor performance path.
class RowEmptyKernelSession {
public:
    RowEmptyKernelSession(vk::Device device, vk::PhysicalDevice physical, vk::Queue queue,
                          uint32_t family, const vulkan3DGS::TrainingScanlineCapture& capture,
                          bool portable = false, const std::atomic<uint32_t>* errors = nullptr)
        : device_(device), queue_(queue), data_(capture), errors_(errors) {
        using namespace vulkan3DGS;
        validateTrainingScanlineCapture(capture);
        if (!portable && !supportsRowEmptyTensor(physical))
            throw std::runtime_error(
                "RowEmpty Tensor requires native32 and the original capability gate");
        const auto limits = physical.getProperties().limits;
        const std::string baseline = portable
                                         ? "train_backward_tensor_batch4_test"
                                         : "train_backward_vksplat_tensor_subgroup_global_grad";
        const std::string candidate = portable ? "train_backward_tensor_row_empty_test"
                                               : "train_backward_vksplat_tensor_row_empty";
        referenceShared = rowShaderSharedBytes(baseline, false);
        candidateShared = rowShaderSharedBytes(candidate, true);
        if (candidateShared > referenceShared + 320u || candidateShared < referenceShared ||
            candidateShared > limits.maxComputeSharedMemorySize ||
            referenceShared > limits.maxComputeSharedMemorySize)
            throw std::runtime_error(
                "RowEmpty declared shared storage exceeds audited/device budget");
        const auto& h = capture.header;
        tx_ = static_cast<uint32_t>((uint64_t(h.width) + 15u) / 16u);
        ty_ = static_cast<uint32_t>((uint64_t(h.height) + 15u) / 16u);
        if (tx_ > limits.maxComputeWorkGroupCount[0] || ty_ > limits.maxComputeWorkGroupCount[1] ||
            limits.maxComputeWorkGroupInvocations < 256u ||
            limits.maxComputeWorkGroupSize[0] < 16u || limits.maxComputeWorkGroupSize[1] < 16u)
            throw std::runtime_error("RowEmpty dispatch exceeds device limits");
        const std::array<uint64_t, 6> sizes = {capture.projected.size() * sizeof(ProjectedGaussian),
                                               capture.items.size() * sizeof(uint32_t),
                                               capture.ranges.size() * sizeof(glm::uvec4),
                                               capture.pixelGrads.size() * sizeof(PixelGrad),
                                               capture.projected.size() *
                                                   sizeof(ProjectedGaussianGrad),
                                               capture.blends.size() * sizeof(PixelBlendState)};
        const std::array<const void*, 6> sources = {capture.projected.data(),
                                                    capture.items.data(),
                                                    capture.ranges.data(),
                                                    capture.pixelGrads.data(),
                                                    nullptr,
                                                    capture.blends.data()};
        for (size_t i = 0; i < buffers_.size(); ++i) {
            if (sizes[i] > limits.maxStorageBufferRange)
                throw std::runtime_error("RowEmpty input exceeds storage-buffer range");
            buffers_[i].create(device, physical, queue, family, sizes[i] ? sources[i] : nullptr,
                               std::max<uint64_t>(sizes[i], 16u),
                               vk::BufferUsageFlagBits::eStorageBuffer |
                                   vk::BufferUsageFlagBits::eTransferSrc |
                                   vk::BufferUsageFlagBits::eTransferDst,
                               vk::MemoryPropertyFlagBits::eDeviceLocal, {family});
            infos_[i] = buffers_[i].getDescriptorInfo();
        }
        ComputePipelineConfig config;
        constexpr std::array<uint32_t, 6> bindings = {3, 4, 5, 12, 13, 14};
        for (auto binding : bindings)
            config.descriptorBindings.push_back(backwardStorageBinding(binding));
        config.pushConstantSize = sizeof(TrainingPushConstants);
        const auto root = std::string(CHECKPOINT_RUNTIME_DIR) + "/shaders/";
        pipelines_[0].initialize(device, root + baseline + ".comp.spv", config);
        pipelines_[1].initialize(device, root + candidate + ".comp.spv", config);
        sets_ = descriptors_.create(
            device, 12, 0,
            {pipelines_[0].getDescriptorSetLayout(), pipelines_[1].getDescriptorSetLayout()});
        for (size_t p = 0; p < 2; ++p) {
            std::array<vk::WriteDescriptorSet, 6> writes{};
            for (size_t i = 0; i < 6; ++i)
                writes[i]
                    .setDstSet(sets_[p])
                    .setDstBinding(bindings[i])
                    .setDescriptorCount(1)
                    .setDescriptorType(vk::DescriptorType::eStorageBuffer)
                    .setPBufferInfo(&infos_[i]);
            device.updateDescriptorSets(writes, {});
        }
        pool_.create(device, family, vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
        command_ = pool_.allocateCommandBuffer();
        fence_ = device.createFenceUnique({});
        timestampBits_ = physical.getQueueFamilyProperties()[family].timestampValidBits;
        period_ = limits.timestampPeriod;
        if (timestampBits_)
            queries_ = device.createQueryPoolUnique({{}, vk::QueryType::eTimestamp, 2});
        params_.gaussianCount = h.gaussianCount;
        params_.width = h.width;
        params_.height = h.height;
        params_.pixelCount = h.width * h.height;
        params_.tileSize = 16;
        params_.tileItemCount = h.itemCount;
    }
    double dispatch(bool candidate) {
        using namespace vulkan3DGS;
        checkValidation();
        command_.reset();
        command_.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        vk::MemoryBarrier reuse{vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite |
                                    vk::AccessFlagBits::eTransferRead |
                                    vk::AccessFlagBits::eTransferWrite,
                                vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite |
                                    vk::AccessFlagBits::eTransferWrite};
        command_.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                                 vk::PipelineStageFlagBits::eAllCommands, {}, reuse, {}, {});
        command_.fillBuffer(buffers_[4].getBuffer(), 0, VK_WHOLE_SIZE, 0);
        vk::MemoryBarrier clear{vk::AccessFlagBits::eTransferWrite,
                                vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite};
        command_.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                 vk::PipelineStageFlagBits::eComputeShader, {}, clear, {}, {});
        if (queries_) {
            command_.resetQueryPool(*queries_, 0, 2);
            command_.writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, *queries_, 0);
        }
        const auto p = candidate ? 1u : 0u;
        command_.bindPipeline(vk::PipelineBindPoint::eCompute, pipelines_[p].getPipeline());
        command_.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                    pipelines_[p].getPipelineLayout(), 0, sets_[p], {});
        command_.pushConstants(pipelines_[p].getPipelineLayout(), vk::ShaderStageFlagBits::eCompute,
                               0, sizeof(params_), &params_);
        command_.dispatch(tx_, ty_, 1);
        recordBackwardBufferBarrier(command_, {infos_[4]},
                                    vk::AccessFlagBits::eShaderRead |
                                        vk::AccessFlagBits::eShaderWrite |
                                        vk::AccessFlagBits::eTransferRead);
        if (queries_)
            command_.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, *queries_, 1);
        command_.end();
        submit();
        if (!queries_) return 0;
        std::array<uint64_t, 2> ticks{};
        const auto status = device_.getQueryPoolResults(
            *queries_, 0, 2, sizeof(ticks), ticks.data(), sizeof(uint64_t),
            vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);
        if (status != vk::Result::eSuccess)
            throw std::runtime_error("RowEmpty timestamp read failed");
        const uint64_t mask =
            timestampBits_ == 64u ? UINT64_MAX : (uint64_t(1) << timestampBits_) - 1u;
        return double((ticks[1] - ticks[0]) & mask) * period_ / 1e6;
    }
    std::vector<vulkan3DGS::ProjectedGaussianGrad> gradients() {
        std::vector<vulkan3DGS::ProjectedGaussianGrad> out(data_.header.gaussianCount);
        if (!out.empty()) buffers_[4].download(out.data(), out.size() * sizeof(out[0]));
        return out;
    }
    bool inputsUnchanged() {
        command_.reset();
        command_.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        vk::MemoryBarrier ready{vk::AccessFlagBits::eShaderWrite |
                                    vk::AccessFlagBits::eTransferWrite,
                                vk::AccessFlagBits::eTransferRead};
        command_.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                                 vk::PipelineStageFlagBits::eTransfer, {}, ready, {}, {});
        command_.end();
        submit();
        std::vector<vulkan3DGS::PixelGrad> pixels(data_.pixelGrads.size());
        std::vector<vulkan3DGS::PixelBlendState> blends(data_.blends.size());
        buffers_[3].download(pixels.data(), pixels.size() * sizeof(pixels[0]));
        buffers_[5].download(blends.data(), blends.size() * sizeof(blends[0]));
        return std::memcmp(pixels.data(), data_.pixelGrads.data(),
                           pixels.size() * sizeof(pixels[0])) == 0 &&
               std::memcmp(blends.data(), data_.blends.data(), blends.size() * sizeof(blends[0])) ==
                   0;
    }
    bool timingAvailable() const {
        return bool(queries_);
    }
    void sample(bool candidate, uint32_t warmup, uint32_t measured, std::vector<double>& out) {
        for (uint32_t i = 0; i < warmup + measured; ++i) {
            const auto ms = dispatch(candidate);
            if (i >= warmup && queries_) out.push_back(ms);
        }
    }
    uint64_t referenceShared = 0, candidateShared = 0;

private:
    void submit() {
        device_.resetFences(*fence_);
        vk::SubmitInfo info;
        info.setCommandBuffers(command_);
        queue_.submit(info, *fence_);
        (void)device_.waitForFences(*fence_, VK_TRUE, UINT64_MAX);
        checkValidation();
    }
    void checkValidation() const {
        if (errors_ && errors_->load() != 0u)
            throw std::runtime_error("RowEmpty Vulkan validation failed; timing aborted");
    }
    vk::Device device_;
    vk::Queue queue_;
    const vulkan3DGS::TrainingScanlineCapture& data_;
    const std::atomic<uint32_t>* errors_;
    std::array<vulkan3DGS::Buffer, 6> buffers_;
    std::array<vk::DescriptorBufferInfo, 6> infos_{};
    std::array<vulkan3DGS::ComputePipeline, 2> pipelines_;
    vulkan3DGS::BackwardDescriptorPool descriptors_;
    std::vector<vk::DescriptorSet> sets_;
    vulkan3DGS::CommandPool pool_;
    vk::UniqueFence fence_;
    vk::UniqueQueryPool queries_;
    vk::CommandBuffer command_;
    vulkan3DGS::TrainingPushConstants params_{};
    uint32_t tx_ = 0, ty_ = 0, timestampBits_ = 0;
    double period_ = 0;
};

struct RowPrototypeResult {
    std::string verdict = "NOT_RUN", error;
    bool sourceMatches = false, capable = false, r0Run = false, gradientsRun = false,
         inputsSame = false, longRun = false;
    uint64_t referenceShared = 0, candidateShared = 0;
    ScanlineGpuResult r0;
    RowGradientComparison gradients, repeat;
    std::vector<double> shortBefore, shortCandidate, shortAfter, longBefore, longCandidate,
        longAfter;
};
inline RowPrototypeResult runRowPrototype(vk::Device device, vk::PhysicalDevice physical,
                                          vk::Queue queue, uint32_t family,
                                          const vulkan3DGS::TrainingScanlineCapture& data,
                                          bool requestLong = false,
                                          const std::atomic<uint32_t>* errors = nullptr) {
    RowPrototypeResult out;
    const auto props = physical.getProperties();
    const auto& h = data.header;
    out.sourceMatches = h.vendorID && h.deviceID && h.vendorID == props.vendorID &&
                        h.deviceID == props.deviceID && h.driverVersion == props.driverVersion;
    out.capable = supportsRowEmptyTensor(physical);
    try {
        out.referenceShared =
            rowShaderSharedBytes("train_backward_vksplat_tensor_subgroup_global_grad", false);
        out.candidateShared = rowShaderSharedBytes("train_backward_vksplat_tensor_row_empty", true);
        if (out.candidateShared < out.referenceShared ||
            out.candidateShared > out.referenceShared + 320u)
            throw std::runtime_error("RowEmpty declared shared budget changed");
    } catch (const std::exception& error) {
        out.verdict = "STOP_SHADER_RESOURCE_OR_NUMERIC_CONTRACT";
        out.error = error.what();
        return out;
    }
    if (!out.capable) {
        out.verdict = "INCONCLUSIVE_TENSOR_CAPABILITY_UNAVAILABLE";
        return out;
    }
    if (!out.sourceMatches) {
        out.verdict = "INCONCLUSIVE_SOURCE_GPU_OR_DRIVER_MISMATCH";
        return out;
    }
    try {
        out.r0 = runScanlineGpuDiagnostic(device, physical, queue, family, data, true);
        out.r0Run = true;
        if (errors && errors->load() != 0u)
            throw std::runtime_error("R0 Vulkan validation failed; candidate not executed");
        const auto& c = out.r0.counts;
        if (c[24] || c[25] || c[26] || c[31]) {
            out.verdict = "STOP_R0_CLASSIFICATION_OR_INPUT";
            return out;
        }
        if (!c[1] || double(c[14]) / double(c[1]) < 0.10) {
            out.verdict = "STOP_LOW_COMPACT_MASK_COVERAGE";
            return out;
        }
        RowEmptyKernelSession kernel(device, physical, queue, family, data, false, errors);
        out.referenceShared = kernel.referenceShared;
        out.candidateShared = kernel.candidateShared;
        (void)kernel.dispatch(false);
        const auto reference = kernel.gradients();
        (void)kernel.dispatch(true);
        out.gradients = compareRowGradients(reference, kernel.gradients());
        (void)kernel.dispatch(false);
        out.repeat = compareRowGradients(reference, kernel.gradients());
        out.gradientsRun = true;
        out.inputsSame = kernel.inputsUnchanged();
        if (!out.gradients.passed() || !out.repeat.passed() || !out.inputsSame) {
            out.verdict = "STOP_GRADIENT_OR_INPUT_MISMATCH";
            return out;
        }
        kernel.sample(false, 2, 8, out.shortBefore);
        kernel.sample(true, 2, 8, out.shortCandidate);
        kernel.sample(false, 2, 8, out.shortAfter);
        out.verdict = rowTimingVerdict(out.shortBefore, out.shortCandidate, out.shortAfter, false);
        if (!requestLong || out.verdict != "SHORT_GAIN_ONLY_REQUIRES_200_1000") return out;
        out.longRun = true;
        kernel.sample(false, 200, 1000, out.longBefore);
        kernel.sample(true, 200, 1000, out.longCandidate);
        kernel.sample(false, 200, 1000, out.longAfter);
        out.verdict = rowTimingVerdict(out.longBefore, out.longCandidate, out.longAfter, true);
    } catch (const std::exception& error) {
        out.verdict = "STOP_PIPELINE_OR_EXECUTION_ERROR";
        out.error = error.what();
    }
    return out;
}
