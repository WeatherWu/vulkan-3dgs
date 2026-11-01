#include "training/core/training_vksplat_adc.hpp"
#include "training/core/backward/backward_pass_context.hpp"
#include "training/core/training_vksplat_backend.hpp"
#include "utils/file_utils.hpp"
#include "utils/logger.hpp"
#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>

namespace vulkan3DGS {
namespace {
struct AdcCostTimer {
    double& total;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    ~AdcCostTimer() {
        total += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                     .count();
    }
};
constexpr std::array<uint32_t, 10> strides{12, 192, 16, 16, 24, 192, 192, 32, 32, 12};
constexpr auto usage = vk::BufferUsageFlagBits::eStorageBuffer |
                       vk::BufferUsageFlagBits::eTransferSrc |
                       vk::BufferUsageFlagBits::eTransferDst;
} // namespace
TrainingVkSplatAdc::TrainingVkSplatAdc(TrainingVkSplatBackend& b, bool reuse)
    : reuse_(reuse), core_(b), device_(b.device_), physical_(b.physical_), queue_(b.queue_),
      family_(b.family_) {
    commandPool_.create(device_, family_, vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
    command_ = commandPool_.allocateCommandBuffer();
    fence_ = device_.createFenceUnique({});
    std::array<vk::DescriptorPoolSize, 1> sizes{{{vk::DescriptorType::eStorageBuffer, 8192}}};
    descriptors_ = device_.createDescriptorPoolUnique(
        vk::DescriptorPoolCreateInfo{}.setMaxSets(1024).setPoolSizes(sizes));
    const vk::DescriptorPoolSize statisticsSize{vk::DescriptorType::eStorageBuffer, 4};
    statisticsPool_ = device_.createDescriptorPoolUnique(
        vk::DescriptorPoolCreateInfo{}.setMaxSets(1).setPoolSizes(statisticsSize));
    if (physical_.getQueueFamilyProperties()[family_].timestampValidBits) {
        timestamps_ = device_.createQueryPoolUnique({{}, vk::QueryType::eTimestamp, 2});
        timestampPeriod_ = physical_.getProperties().limits.timestampPeriod;
    }
    readback_ = allocate(16, true);
    scratch_.reserve(256);
}
Buffer TrainingVkSplatAdc::allocate(uint64_t bytes, bool host) {
    AdcCostTimer timer{costs_.allocationMs};
    if (bytes > physical_.getProperties().limits.maxStorageBufferRange)
        throw std::runtime_error("VkSplat ADC allocation exceeds maxStorageBufferRange");
    Buffer b;
    b.create(device_, physical_, queue_, family_, nullptr, std::max<uint64_t>(bytes, 4), usage,
             host ? vk::MemoryPropertyFlagBits::eHostVisible |
                        vk::MemoryPropertyFlagBits::eHostCoherent
                  : vk::MemoryPropertyFlagBits::eDeviceLocal,
             {family_});
    return b;
}
void TrainingVkSplatAdc::barrier(vk::CommandBuffer c) {
    vk::MemoryBarrier memory{vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
                             vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite};
    c.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                      vk::PipelineStageFlagBits::eAllCommands, {}, memory, {}, {});
}
void TrainingVkSplatAdc::begin() {
    if (!reuse_)
        device_.resetDescriptorPool(*descriptors_);
    else {
        scratchIndex_ = 0;
        for (auto& [name, p] : pipelines_)
            p.used = 0;
    }
    scratch_.clear();
    retired_.clear();
    command_.reset();
    command_.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    barrier(command_);
    if (timestamps_) {
        command_.resetQueryPool(*timestamps_, 0, 2);
        command_.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, *timestamps_, 0);
    }
}
void TrainingVkSplatAdc::submit() {
    barrier(command_);
    vk::MemoryBarrier hostVisible{vk::AccessFlagBits::eTransferWrite,
                                  vk::AccessFlagBits::eHostRead};
    command_.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost,
                             {}, hostVisible, {}, {});
    if (timestamps_)
        command_.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, *timestamps_, 1);
    command_.end();
    queue_.submit(vk::SubmitInfo{}.setCommandBuffers(command_), *fence_);
    {
        AdcCostTimer timer{costs_.waitMs};
        (void)device_.waitForFences(*fence_, true, UINT64_MAX);
    }
    device_.resetFences(*fence_);
    if (timestamps_) {
        std::array<uint64_t, 2> t{};
        auto status = device_.getQueryPoolResults(
            *timestamps_, 0, 2, sizeof(t), t.data(), sizeof(uint64_t),
            vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);
        if (status == vk::Result::eSuccess) {
            const uint32_t bits = physical_.getQueueFamilyProperties()[family_].timestampValidBits;
            const uint64_t mask = bits == 64 ? UINT64_MAX : (uint64_t(1) << bits) - 1;
            const double ms = double((t[1] - t[0]) & mask) * timestampPeriod_ * 1e-6;
            gpuMs_ += ms;
            gpuBatches_.push_back(ms);
        }
    }
    retired_.clear();
}
void TrainingVkSplatAdc::ensureStatistics(uint32_t capacity) {
    if (capacity <= statisticsCapacity_) return;
    auto g = allocate(uint64_t(capacity) * 8), r = allocate(uint64_t(capacity) * 4);
    begin();
    command_.fillBuffer(g.getBuffer(), 0, VK_WHOLE_SIZE, 0);
    command_.fillBuffer(r.getBuffer(), 0, VK_WHOLE_SIZE, 0);
    if (statisticsCapacity_) {
        barrier(command_);
        copy(grad_.getBuffer(), g.getBuffer(), vk::BufferCopy{0, 0, grad_.getSize()});
        copy(radius_.getBuffer(), r.getBuffer(), vk::BufferCopy{0, 0, radius_.getSize()});
    }
    submit();
    grad_ = std::move(g);
    radius_ = std::move(r);
    statisticsCapacity_ = capacity;
}
void TrainingVkSplatAdc::prepare(const TrainingDensificationConfig& config, float extent,
                                 uint32_t iteration, bool benchmark) {
    if (failed_)
        throw std::runtime_error("VkSplat ADC state is uncertain; explicitly reload the dataset");
    if (config.splitChildren != 2 ||
        config.gradientMode != TrainingDensificationGradientMode::VisibleSupport)
        throw std::runtime_error(
            "VkSplat ADC requires two split children and Visible Support statistics");
    config_ = config;
    benchmark_ = benchmark;
    u_.step = iteration;
    if (!benchmark) u_.seed = std::uniform_int_distribution<uint32_t>()(rng_);
    u_.pruneOpacity = config.minOpacity;
    u_.growGradient = config.densifyGradThreshold;
    u_.growScale = config.percentDense * extent;
    u_.pruneScale = config.worldSizePruneThreshold * extent;
    u_.start = static_cast<int32_t>(config.densifyFromIteration);
    u_.stop = static_cast<int32_t>(config.densifyUntilIteration);
    u_.resetEvery = static_cast<int32_t>(config.opacityResetInterval);
    u_.every = static_cast<int32_t>(config.densificationInterval);
}
void TrainingVkSplatAdc::recordStatistics(const BackwardPassContext& c) {
    if (benchmark_ || !config_.enabled || u_.step >= config_.densifyUntilIteration) return;
    if (!statisticsSet_) {
        ComputePipelineConfig config;
        config.pushConstantSize = sizeof(Uniforms);
        for (uint32_t i = 0; i < 4; ++i)
            config.descriptorBindings.push_back(backwardStorageBinding(i));
        statisticsPipeline_.initialize(
            device_, FileUtils::resolvePath("shaders/train_vksplat_adc_update.comp.spv"), config);
        auto layout = statisticsPipeline_.getDescriptorSetLayout();
        statisticsSet_ = device_.allocateDescriptorSets(
            vk::DescriptorSetAllocateInfo{*statisticsPool_, 1, &layout})[0];
    }
    std::array<vk::DescriptorBufferInfo, 4> info{c.buffers.projectedGradsInfo(),
                                                 core_.officialRadiiInfo(), gradients(), radii()};
    std::array<vk::WriteDescriptorSet, 4> writes;
    for (uint32_t i = 0; i < 4; ++i)
        writes[i] = {statisticsSet_, i,       0, 1, vk::DescriptorType::eStorageBuffer,
                     nullptr,        &info[i]};
    if (reuse_) {
        auto& cache = pipelines_["statistics"].sets;
        if (cache.empty()) cache.emplace_back();
        auto& previous = cache.front().infos;
        if (previous.size() != info.size() ||
            !std::equal(previous.begin(), previous.end(), info.begin())) {
            device_.updateDescriptorSets(writes, {});
            previous.assign(info.begin(), info.end());
        }
    } else
        device_.updateDescriptorSets(writes, {});
    auto u = u_;
    u.count = c.pushConstants.gaussianCount;
    u.width = c.pushConstants.width;
    u.height = c.pushConstants.height;
    auto command = c.commandBuffer;
    auto layout = statisticsPipeline_.getPipelineLayout();
    command.bindPipeline(vk::PipelineBindPoint::eCompute, statisticsPipeline_.getPipeline());
    command.bindDescriptorSets(vk::PipelineBindPoint::eCompute, layout, 0, statisticsSet_, {});
    command.pushConstants(layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(u), &u);
    if (u.count) command.dispatch((u.count + 255) / 256, 1, 1);
    barrier(command);
}
void TrainingVkSplatAdc::dispatch(const std::string& name,
                                  const std::vector<vk::DescriptorBufferInfo>& infos,
                                  uint32_t groups, const void* push, uint32_t size) {
    if (!groups) return;
    auto& p = pipelines_[name].pipeline;
    if (!p.getPipeline()) {
        ComputePipelineConfig config;
        config.pushConstantSize = size;
        for (uint32_t i = 0; i < infos.size(); ++i)
            config.descriptorBindings.push_back(backwardStorageBinding(i));
        p.initialize(device_,
                     FileUtils::resolvePath("shaders/train_vksplat_adc_" + name + ".comp.spv"),
                     config);
    }
    auto dsl = p.getDescriptorSetLayout();
    vk::DescriptorSet set;
    CachedSet* cached = nullptr;
    if (reuse_) {
        auto& item = pipelines_[name];
        if (item.used == item.sets.size()) {
            CachedSet next;
            next.set = device_.allocateDescriptorSets(
                vk::DescriptorSetAllocateInfo{*descriptors_, 1, &dsl})[0];
            item.sets.push_back(std::move(next));
        }
        cached = &item.sets[item.used++];
        set = cached->set;
    } else
        set = device_.allocateDescriptorSets(
            vk::DescriptorSetAllocateInfo{*descriptors_, 1, &dsl})[0];
    std::vector<vk::WriteDescriptorSet> writes(infos.size());
    for (uint32_t i = 0; i < infos.size(); ++i)
        writes[i] = {set, i, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &infos[i]};
    if (!cached || cached->infos != infos) {
        device_.updateDescriptorSets(writes, {});
        if (cached) cached->infos = infos;
    }
    auto layout = p.getPipelineLayout();
    command_.bindPipeline(vk::PipelineBindPoint::eCompute, p.getPipeline());
    command_.bindDescriptorSets(vk::PipelineBindPoint::eCompute, layout, 0, set, {});
    command_.pushConstants(layout, vk::ShaderStageFlagBits::eCompute, 0, size, push);
    command_.dispatch(groups, 1, 1);
    barrier(command_);
}
void TrainingVkSplatAdc::scan(Buffer& input, Buffer& output, uint32_t count) {
    const uint32_t groups = (count + 1023) / 1024;
    if (!reuse_) scratch_.push_back(allocate(uint64_t(groups) * 4));
    auto& sums = reuse_
                     ? reuseBuffer("scan" + std::to_string(scratchIndex_++), uint64_t(groups) * 4)
                     : scratch_.back();
    const auto subgroup = core_.subgroupSize_ == 64 ? "64" : "32";
    dispatch(std::string(groups == 1 ? "scan_" : "blocks_") + subgroup,
             {input.getDescriptorInfo(), output.getDescriptorInfo(), sums.getDescriptorInfo()},
             groups, &count, 4);
    if (groups > 1) {
        if (!reuse_) scratch_.push_back(allocate(uint64_t(groups) * 4));
        auto& prefix =
            reuse_ ? reuseBuffer("scan" + std::to_string(scratchIndex_++), uint64_t(groups) * 4)
                   : scratch_.back();
        scan(sums, prefix, groups);
        dispatch(
            std::string("offsets_") + subgroup,
            {input.getDescriptorInfo(), output.getDescriptorInfo(), prefix.getDescriptorInfo()},
            groups, &count, 4);
    }
}
uint32_t TrainingVkSplatAdc::readCount(Buffer& prefix, uint32_t count) {
    AdcCostTimer timer{costs_.countMs};
    if (!count) return 0;
    copy(prefix.getBuffer(), readback_.getBuffer(), vk::BufferCopy{uint64_t(count - 1) * 4, 0, 4});
    submit();
    uint32_t n = 0;
    readback_.download(&n, 4);
    return n;
}
void TrainingVkSplatAdc::copy(vk::Buffer input, vk::Buffer output, vk::BufferCopy region) {
    AdcCostTimer timer{costs_.copyRecordMs};
    command_.copyBuffer(input, output, region);
}
void TrainingVkSplatAdc::grow(uint32_t count) {
    AdcCostTimer timer{costs_.growthMs};
    if (count <= core_.capacity_) return;
    const uint32_t cap = (count + 31) / 32 * 32;
    std::array<Buffer, 10> replacement;
    for (size_t i = 0; i < 10; ++i)
        replacement[i] = allocate(uint64_t(cap) * strides[i]);
    auto rect = allocate(uint64_t(cap) * 16), radii = allocate(uint64_t(cap) * 4);
    auto g = allocate(uint64_t(cap) * 8), r = allocate(uint64_t(cap) * 4);
    for (size_t i = 0; i < 10; ++i) {
        copy(core_.model_[i].getBuffer(), replacement[i].getBuffer(),
             vk::BufferCopy{0, 0,
                            reuse_ ? uint64_t((i == 1 || i == 5 || i == 6)
                                                  ? (u_.count + 31u) / 32u * 32u
                                                  : u_.count) *
                                         strides[i]
                                   : core_.model_[i].getSize()});
        retired_.push_back(std::move(core_.model_[i]));
    }
    copy(grad_.getBuffer(), g.getBuffer(),
         vk::BufferCopy{0, 0, reuse_ ? uint64_t(u_.count) * 8u : grad_.getSize()});
    copy(radius_.getBuffer(), r.getBuffer(),
         vk::BufferCopy{0, 0, reuse_ ? uint64_t(u_.count) * 4u : radius_.getSize()});
    retired_.push_back(std::move(grad_));
    retired_.push_back(std::move(radius_));
    for (auto& binding : core_.lastBindings_)
        binding.clear();
    core_.model_ = std::move(replacement);
    core_.rectangles_ = std::move(rect);
    core_.officialRadii_ = std::move(radii);
    grad_ = std::move(g);
    radius_ = std::move(r);
    statisticsCapacity_ = cap;
    core_.capacity_ = cap;
    barrier(command_);
}
VkSplatAdcResult TrainingVkSplatAdc::execute(uint32_t count, bool refine, bool reset) {
    if (reuse_) return executeReuse(count, refine, reset);
    using Clock = std::chrono::steady_clock;
    auto started = Clock::now();
    costs_ = {};
    VkSplatAdcResult result;
    result.input = count;
    result.seed = u_.seed;
    result.reset = false;
    result.stats.outputCount = count;
    gpuMs_ = 0;
    gpuBatches_.clear();
    if (failed_ || benchmark_)
        throw std::runtime_error("VkSplat ADC requires a prepared real training step");
    if (!count) {
        result.capacity = core_.capacity_;
        return result;
    }
    bool mutation = false;
    try {
        uint32_t duplicates = 0, splits = 0, grown = count;
        if (refine) {
            auto dm = allocate(uint64_t(count) * 4), sm = allocate(uint64_t(count) * 4);
            auto dp = allocate(uint64_t(count) * 4), sp = allocate(uint64_t(count) * 4);
            Buffer di, si;
            auto u = u_;
            u.count = count;
            begin();
            dispatch("grow",
                     {gradients(), radii(), core_.model_[3].getDescriptorInfo(),
                      dm.getDescriptorInfo(), sm.getDescriptorInfo()},
                     (count + 255) / 256, &u, sizeof(u));
            scan(dm, dp, count);
            duplicates = readCount(dp, count);
            begin();
            scan(sm, sp, count);
            splits = readCount(sp, count);
            const uint64_t target = uint64_t(count) + duplicates + splits;
            const uint64_t dispatchLimit =
                uint64_t(physical_.getProperties().limits.maxComputeWorkGroupCount[0]) * 256u / 8u;
            if (target > config_.maxGaussianCount || target > uint64_t(INT32_MAX) / 48u ||
                target > dispatchLimit)
                throw std::runtime_error(
                    "VkSplat ADC growth exceeds Max Gaussians, device dispatch or native index "
                    "limits; reload with a larger budget");
            grown = static_cast<uint32_t>(target);
            result.stats.cloneSources = duplicates;
            result.stats.splitSources = splits;
            result.stats.keptSources = count - duplicates - splits;
            if (duplicates) di = allocate(uint64_t(duplicates) * 4);
            if (splits) si = allocate(uint64_t(splits) * 4);
            begin();
            if (duplicates)
                dispatch("where",
                         {dm.getDescriptorInfo(), dp.getDescriptorInfo(), di.getDescriptorInfo()},
                         (count + 255) / 256, &count, 4);
            if (splits)
                dispatch("where",
                         {sm.getDescriptorInfo(), sp.getDescriptorInfo(), si.getDescriptorInfo()},
                         (count + 255) / 256, &count, 4);
            mutation = true;
            grow(grown);
            auto growDispatch = [&](const char* name, Buffer& indices, uint32_t base, uint32_t n) {
                if (!n) return;
                std::vector<vk::DescriptorBufferInfo> info{indices.getDescriptorInfo()};
                for (uint32_t i = 0; i < 9; ++i)
                    info.push_back(core_.model_[i].getDescriptorInfo());
                info.push_back(gradients());
                info.push_back(radii());
                info.push_back(core_.model_[9].getDescriptorInfo());
                auto push = u_;
                push.count = base + n;
                push.oldCount = base;
                dispatch(name, info, (n + 255) / 256, &push, sizeof(push));
            };
            growDispatch("duplicate", di, count, duplicates);
            growDispatch("split", si, count + duplicates, splits);
            auto keep = allocate(uint64_t(grown) * 4), prefix = allocate(uint64_t(grown) * 4);
            auto reasons = allocate(12);
            command_.fillBuffer(reasons.getBuffer(), 0, VK_WHOLE_SIZE, 0);
            barrier(command_);
            u.count = grown;
            dispatch("keep",
                     {core_.model_[3].getDescriptorInfo(), radii(), keep.getDescriptorInfo(),
                      reasons.getDescriptorInfo()},
                     (grown + 255) / 256, &u, sizeof(u));
            scan(keep, prefix, grown);
            copy(reasons.getBuffer(), readback_.getBuffer(), vk::BufferCopy{0, 4, 12});
            mutation = true;
            const uint32_t remaining = readCount(prefix, grown);
            std::array<uint32_t, 4> counts{};
            readback_.download(counts.data(), 16);
            result.stats.pruneOpacityHits = counts[1];
            result.stats.pruneWorldHits = counts[2];
            result.stats.prunedSources = grown - remaining;
            result.stats.outputCount = remaining;
            if (!remaining) {
                LOG_ERROR("VkSplat ADC empty output: input={}, duplicate={}, split={}, prune={}, "
                          "output=0",
                          count, duplicates, splits, grown);
                failed_ = true;
                result.capacity = core_.capacity_;
                result.gpuMs = gpuMs_;
                result.gpuBatches = gpuBatches_;
                result.cpuMs =
                    std::chrono::duration<double, std::milli>(Clock::now() - started).count();
                throw VkSplatAdcEmptyResult(std::move(result));
            }
            AdcCostTimer compactTimer{costs_.compactionMs};
            if (remaining < grown) {
                begin();
                // Stable compaction uses the same prefix for every native field and both moments.
                std::array<Buffer, 10> compact;
                for (size_t i = 0; i < 10; ++i)
                    compact[i] = allocate(uint64_t(core_.capacity_) * strides[i]);
                for (size_t i = 0; i < 10; ++i) {
                    auto push = u_;
                    push.oldCount = grown;
                    const char* kernel = "compact";
                    uint32_t work = grown;
                    if (i == 0 || i == 4) {
                        kernel = "compact_mean";
                        push.count = i == 0 ? 1 : 2;
                    } else if (i == 1 || i == 5 || i == 6) {
                        kernel = "compact_sh";
                        push.count = 1;
                    } else {
                        push.count = strides[i] / 4;
                        work *= push.count;
                    }
                    dispatch(kernel,
                             {keep.getDescriptorInfo(), prefix.getDescriptorInfo(),
                              core_.model_[i].getDescriptorInfo(), compact[i].getDescriptorInfo()},
                             (work + 255) / 256, &push, sizeof(push));
                }
                auto compactRadius = allocate(radius_.getSize());
                auto push = u_;
                push.oldCount = grown;
                push.count = 1;
                dispatch("compact",
                         {keep.getDescriptorInfo(), prefix.getDescriptorInfo(), radii(),
                          compactRadius.getDescriptorInfo()},
                         (grown + 255) / 256, &push, sizeof(push));
                barrier(command_);
                command_.fillBuffer(grad_.getBuffer(), 0, uint64_t(remaining) * 8, 0);
                submit();
                for (auto& binding : core_.lastBindings_)
                    binding.clear();
                core_.model_ = std::move(compact);
                radius_ = std::move(compactRadius);
            } else {
                begin();
                command_.fillBuffer(grad_.getBuffer(), 0, uint64_t(remaining) * 8, 0);
                submit();
            }
            count = remaining;
        }
        if (reset) {
            AdcCostTimer timer{costs_.resetMs};
            auto u = u_;
            u.count = count;
            begin();
            dispatch("reset",
                     {core_.model_[3].getDescriptorInfo(), core_.model_[8].getDescriptorInfo()},
                     (count + 255) / 256, &u, sizeof(u));
            mutation = true;
            submit();
            result.reset = true;
        }
        result.costs = costs_;
        result.capacity = core_.capacity_;
        result.gpuMs = gpuMs_;
        result.gpuBatches = gpuBatches_;
        for (size_t i = 0; i < gpuBatches_.size(); ++i)
            LOG_INFO("VkSplat ADC GPU batch {}: {} ms", i, gpuBatches_[i]);
        result.cpuMs = std::chrono::duration<double, std::milli>(Clock::now() - started).count();
        if (refine && result.stats.outputCount !=
                          result.input + duplicates + splits - result.stats.prunedSources)
            throw std::runtime_error("VkSplat ADC count conservation failed");
        LOG_INFO("VkSplat ADC: input={}, duplicate={}, split={}, prune={}, output={}, capacity={}, "
                 "reset={}, seed={}, CPU={} ms, GPU batches={} ms",
                 result.input, duplicates, splits, result.stats.prunedSources,
                 result.stats.outputCount, result.capacity, reset, result.seed, result.cpuMs,
                 result.gpuMs);
        return result;
    } catch (...) {
        // Submitted native writes cannot be described as a successful rollback.
        if (mutation) failed_ = true;
        throw;
    }
}
Buffer& TrainingVkSplatAdc::reuseBuffer(const std::string& name, uint64_t bytes) {
    auto& b = reusable_[name];
    if (b.getSize() < std::max<uint64_t>(bytes, 4)) b = allocate(bytes);
    return b;
}
uint32_t TrainingVkSplatAdc::sum(Buffer& input, uint32_t count) {
    auto& total = reuseBuffer("sum", 4);
    command_.fillBuffer(total.getBuffer(), 0, 4, 0);
    barrier(command_);
    dispatch("sum", {input.getDescriptorInfo(), total.getDescriptorInfo()}, (count + 1023) / 1024,
             &count, 4);
    return readCount(total, 1);
}
VkSplatAdcResult TrainingVkSplatAdc::executeReuse(uint32_t count, bool refine, bool reset) {
    using Clock = std::chrono::steady_clock;
    auto started = Clock::now();
    u_.count = count;
    costs_ = {};
    VkSplatAdcResult result;
    result.input = count;
    result.seed = u_.seed;
    result.reset = false;
    result.stats.outputCount = count;
    gpuMs_ = 0;
    gpuBatches_.clear();
    if (failed_ || benchmark_)
        throw std::runtime_error("VkSplat ADC requires a prepared real training step");
    if (!count) {
        result.capacity = core_.capacity_;
        return result;
    }
    bool mutation = false;
    try {
        uint32_t duplicates = 0, splits = 0, grown = count;
        if (refine) {
            auto& dm = reuseBuffer("duplicate-mask", uint64_t(count) * 4);
            auto& sm = reuseBuffer("split-mask", uint64_t(count) * 4);
            auto& dp = reuseBuffer("duplicate-prefix", uint64_t(count) * 4);
            auto& sp = reuseBuffer("split-prefix", uint64_t(count) * 4);
            auto& di = reusable_["duplicate-index"];
            auto& si = reusable_["split-index"];
            auto u = u_;
            u.count = count;
            begin();
            dispatch("grow",
                     {gradients(), radii(), core_.model_[3].getDescriptorInfo(),
                      dm.getDescriptorInfo(), sm.getDescriptorInfo()},
                     (count + 255) / 256, &u, sizeof(u));
            duplicates = sum(dm, count);
            begin();
            splits = sum(sm, count);
            const uint64_t target = uint64_t(count) + duplicates + splits;
            const uint64_t dispatchLimit =
                uint64_t(physical_.getProperties().limits.maxComputeWorkGroupCount[0]) * 256u / 8u;
            if (target > config_.maxGaussianCount || target > uint64_t(INT32_MAX) / 48u ||
                target > dispatchLimit)
                throw std::runtime_error(
                    "VkSplat ADC growth exceeds Max Gaussians, device dispatch or native index "
                    "limits; reload with a larger budget");
            grown = static_cast<uint32_t>(target);
            result.stats.cloneSources = duplicates;
            result.stats.splitSources = splits;
            result.stats.keptSources = count - duplicates - splits;
            if (duplicates) (void)reuseBuffer("duplicate-index", uint64_t(duplicates) * 4);
            if (splits) (void)reuseBuffer("split-index", uint64_t(splits) * 4);
            begin();
            if (duplicates) scan(dm, dp, count);
            if (splits) scan(sm, sp, count);
            if (duplicates)
                dispatch("where",
                         {dm.getDescriptorInfo(), dp.getDescriptorInfo(), di.getDescriptorInfo()},
                         (count + 255) / 256, &count, 4);
            if (splits)
                dispatch("where",
                         {sm.getDescriptorInfo(), sp.getDescriptorInfo(), si.getDescriptorInfo()},
                         (count + 255) / 256, &count, 4);
            mutation = true;
            grow(grown);
            auto growDispatch = [&](const char* name, Buffer& indices, uint32_t base, uint32_t n) {
                if (!n) return;
                std::vector<vk::DescriptorBufferInfo> info{indices.getDescriptorInfo()};
                for (uint32_t i = 0; i < 9; ++i)
                    info.push_back(core_.model_[i].getDescriptorInfo());
                info.push_back(gradients());
                info.push_back(radii());
                info.push_back(core_.model_[9].getDescriptorInfo());
                auto push = u_;
                push.count = base + n;
                push.oldCount = base;
                dispatch(name, info, (n + 255) / 256, &push, sizeof(push));
            };
            growDispatch("duplicate", di, count, duplicates);
            growDispatch("split", si, count + duplicates, splits);
            auto& keep = reuseBuffer("keep", uint64_t(grown) * 4);
            auto& prefix = reuseBuffer("keep-prefix", uint64_t(grown) * 4);
            auto& reasons = reuseBuffer("reasons", 12);
            command_.fillBuffer(reasons.getBuffer(), 0, VK_WHOLE_SIZE, 0);
            barrier(command_);
            u.count = grown;
            dispatch("keep",
                     {core_.model_[3].getDescriptorInfo(), radii(), keep.getDescriptorInfo(),
                      reasons.getDescriptorInfo()},
                     (grown + 255) / 256, &u, sizeof(u));
            copy(reasons.getBuffer(), readback_.getBuffer(), vk::BufferCopy{0, 4, 12});
            mutation = true;
            const uint32_t remaining = sum(keep, grown);
            std::array<uint32_t, 4> counts{};
            readback_.download(counts.data(), 16);
            result.stats.pruneOpacityHits = counts[1];
            result.stats.pruneWorldHits = counts[2];
            result.stats.prunedSources = grown - remaining;
            result.stats.outputCount = remaining;
            if (!remaining) {
                LOG_ERROR("VkSplat ADC empty output: input={}, duplicate={}, split={}, prune={}, "
                          "output=0",
                          count, duplicates, splits, grown);
                failed_ = true;
                result.capacity = core_.capacity_;
                result.gpuMs = gpuMs_;
                result.gpuBatches = gpuBatches_;
                result.cpuMs =
                    std::chrono::duration<double, std::milli>(Clock::now() - started).count();
                throw VkSplatAdcEmptyResult(std::move(result));
            }
            AdcCostTimer compactTimer{costs_.compactionMs};
            if (remaining < grown) {
                begin();
                scan(keep, prefix, grown);
                auto& temp = reuseBuffer("attribute", uint64_t(core_.capacity_) * 192);
                for (size_t i : {size_t(1), size_t(5), size_t(6), size_t(0), size_t(2), size_t(3),
                                 size_t(4), size_t(7), size_t(8), size_t(9)}) {
                    auto push = u_;
                    push.oldCount = grown;
                    const char* kernel = "compact";
                    uint32_t work = grown;
                    if (i == 0 || i == 4) {
                        kernel = "compact_mean";
                        push.count = i == 0 ? 1 : 2;
                    } else if (i == 1 || i == 5 || i == 6) {
                        kernel = "compact_sh";
                        push.count = 1;
                    } else {
                        push.count = strides[i] / 4;
                        work *= push.count;
                    }
                    dispatch(kernel,
                             {keep.getDescriptorInfo(), prefix.getDescriptorInfo(),
                              core_.model_[i].getDescriptorInfo(), temp.getDescriptorInfo()},
                             (work + 255) / 256, &push, sizeof(push));
                    const uint64_t rows = (i == 1 || i == 5 || i == 6)
                                              ? (uint64_t(remaining) + 31) / 32 * 32
                                              : remaining;
                    copy(temp.getBuffer(), core_.model_[i].getBuffer(),
                         vk::BufferCopy{0, 0, rows * strides[i]});
                    barrier(command_);
                }
                auto push = u_;
                push.oldCount = grown;
                push.count = 1;
                dispatch("compact",
                         {keep.getDescriptorInfo(), prefix.getDescriptorInfo(), radii(),
                          temp.getDescriptorInfo()},
                         (grown + 255) / 256, &push, sizeof(push));
                copy(temp.getBuffer(), radius_.getBuffer(),
                     vk::BufferCopy{0, 0, uint64_t(remaining) * 4});
                barrier(command_);
                command_.fillBuffer(grad_.getBuffer(), 0, uint64_t(remaining) * 8, 0);
                submit();
            } else {
                begin();
                command_.fillBuffer(grad_.getBuffer(), 0, uint64_t(remaining) * 8, 0);
                submit();
            }
            count = remaining;
        }
        if (reset) {
            AdcCostTimer timer{costs_.resetMs};
            auto u = u_;
            u.count = count;
            begin();
            dispatch("reset",
                     {core_.model_[3].getDescriptorInfo(), core_.model_[8].getDescriptorInfo()},
                     (count + 255) / 256, &u, sizeof(u));
            mutation = true;
            submit();
            result.reset = true;
        }
        result.costs = costs_;
        result.capacity = core_.capacity_;
        result.gpuMs = gpuMs_;
        result.gpuBatches = gpuBatches_;
        for (size_t i = 0; i < gpuBatches_.size(); ++i)
            LOG_INFO("VkSplat ADC GPU batch {}: {} ms", i, gpuBatches_[i]);
        result.cpuMs = std::chrono::duration<double, std::milli>(Clock::now() - started).count();
        if (refine && result.stats.outputCount !=
                          result.input + duplicates + splits - result.stats.prunedSources)
            throw std::runtime_error("VkSplat ADC count conservation failed");
        LOG_INFO("VkSplat ADC: input={}, duplicate={}, split={}, prune={}, output={}, capacity={}, "
                 "reset={}, seed={}, CPU={} ms, GPU batches={} ms",
                 result.input, duplicates, splits, result.stats.prunedSources,
                 result.stats.outputCount, result.capacity, reset, result.seed, result.cpuMs,
                 result.gpuMs);
        return result;
    } catch (...) {
        // Submitted native writes cannot be described as a successful rollback.
        if (mutation) failed_ = true;
        throw;
    }
}
bool TrainingVkSplatAdc::inspectOpacityReset(uint32_t count, float minimum) {
    if (failed_ || !count || count > core_.capacity_) return false;
    std::vector<glm::vec4> scales(count), moments(size_t(count) * 2);
    core_.model_[3].download(scales.data(), uint64_t(count) * 16);
    core_.model_[8].download(moments.data(), uint64_t(count) * 32);
    for (uint32_t i = 0; i < count; ++i)
        if (!(scales[i].w <= 2 * minimum) || moments[size_t(i) * 2].w != 0 ||
            moments[size_t(i) * 2 + 1].w != 0)
            return false;
    return true;
}
uint64_t TrainingVkSplatAdc::allocatedBytes() const {
    uint64_t n = grad_.getSize() + radius_.getSize() + readback_.getSize();
    for (const auto& [name, b] : reusable_)
        n += b.getSize();
    for (const auto& b : scratch_)
        n += b.getSize();
    return n;
}
} // namespace vulkan3DGS
