#pragma once
#include "training_full_sort_reference.hpp"
#include "training_range_key_session.hpp"
#include "vulkan/diagnostic_radix_sort.hpp"

namespace full_sort {
struct Sample {
    double activeMs = 0, spanMs = 0, rangeMs = 0, prepareMs = 0, mainMs = 0;
    double restoreCpuMs = 0, cpuMs = 0;
};
// Borrowed for the producer session lifetime; the next run replaces their contents.
struct OutputView {
    vk::DescriptorBufferInfo projected, items, ranges;
    TrainingPushConstants push;
};
// Frozen projected inputs; all list producers run on GPU in the timed prepare/main sequence.
class Session {
    enum Pass : size_t {
        Count,
        Emit,
        Prepare,
        OrderedEmit,
        GatherHigh,
        GatherIDs,
        IndirectRanges,
        DirectRanges,
        Finalize,
        Clear
    };

public:
    Session(vk::Device device, vk::PhysicalDevice physical, vk::Queue queue, uint32_t family,
            const Inputs& input)
        : device_(device), physical_(physical), queue_(queue), family_(family), input_(input),
          expected_(oracle(input)), visible_(visibleOrder(input)),
          transfer_(device, physical, queue, family) {}
    ~Session() {
        try {
            device_.waitIdle();
            if (mapped_) device_.unmapMemory(readback_.getMemory());
        } catch (...) {}
    }
    void initialize() {
        const auto& h = input_.header.source;
        g_ = h.gaussianCount;
        n_ = h.itemCount;
        t_ = h.tileCount;
        check(g_ <= 2000000 && n_ <= (1u << 22) && input_.header.capacity >= std::max(n_, 1u) &&
                  input_.header.capacity <= (1u << 24),
              "Full sort workload bounds");
        const auto limits = physical_.getProperties().limits;
        auto align = [&](uint64_t bytes) {
            const auto a = uint64_t(limits.minStorageBufferOffsetAlignment);
            return (bytes + a - 1) / a * a;
        };
        itemBytes_ = std::max(uint64_t(input_.header.capacity + guardWords) * 4,
                              align(uint64_t(n_) * 4) + 64);
        gBytes_ = std::max(uint64_t(g_) * 4, align(uint64_t(g_) * 4)) + 64;
        uint32_t blocks = (g_ + 255) / 256;
        do {
            prefixWords_ += blocks;
            blocks = (blocks + 255) / 256;
        } while (blocks > 1);
        prefixWords_ = std::max(prefixWords_ + 1, 1u);
        rangeBytes_ = uint64_t(t_) * 16;
        check(t_ && itemBytes_ <= limits.maxStorageBufferRange &&
                  uint64_t(std::max(g_, 1u)) * 48 + 64 <= limits.maxStorageBufferRange &&
                  rangeBytes_ + 64 <= limits.maxStorageBufferRange &&
                  limits.maxComputeWorkGroupInvocations >= 512 &&
                  limits.maxComputeWorkGroupSize[0] >= 512 &&
                  (uint64_t(std::max({g_, n_ + 1, t_})) + 255) / 256 <=
                      limits.maxComputeWorkGroupCount[0],
              "Full sort device bounds");
        VrdxSorterCreateInfo create{physical_, device_, VK_NULL_HANDLE};
        VrdxSorter sorter = nullptr;
        vrdxCreateSorter(&create, &sorter);
        sorter_.reset(sorter);
        VrdxSorterStorageRequirements req{};
        vrdxGetSorterKeyValueStorageRequirements(sorter, std::max({g_, n_, 2u}), &req);
        scratchBytes_ = req.size;
        const uint64_t budget = itemBytes_ * 6 + uint64_t(g_) * 96 + gBytes_ * 4 + rangeBytes_ +
                                scratchBytes_ + (1ull << 20);
        check(availableSystemMemoryBytes() >= budget + (256ull << 20), "Full sort RAM budget");
        check(scratchBytes_ + 64 <= limits.maxStorageBufferRange,
              "Full sort scratch descriptor limit");
        checkOptimizerCaptureBudget(physical_, 0, budget);
        auto allocate = [&](Buffer& b, uint64_t bytes) {
            b.create(device_, physical_, queue_, family_, nullptr, bytes,
                     vk::BufferUsageFlagBits::eStorageBuffer |
                         vk::BufferUsageFlagBits::eTransferSrc |
                         vk::BufferUsageFlagBits::eTransferDst,
                     vk::MemoryPropertyFlagBits::eDeviceLocal, {family_});
        };
        allocate(projected_, uint64_t(std::max(g_, 1u)) * 48 + 64);
        allocate(densification_, uint64_t(std::max(g_, 1u)) * 16 + 64);
        for (auto& b : banks_)
            allocate(b, uint64_t(std::max(g_, 1u)) * 8 + 64);
        for (auto& b : controls_)
            allocate(b, 80);
        allocate(prefix_, uint64_t(prefixWords_) * 4 + 64);
        allocate(depth_, gBytes_);
        allocate(visibleIDs_, gBytes_);
        for (auto& b : items_)
            allocate(b, itemBytes_);
        allocate(ranges_, rangeBytes_ + 64);
        allocate(storage_, scratchBytes_ + 64);
        readback_.create(
            device_, physical_, queue_, family_, nullptr, 32, vk::BufferUsageFlagBits::eTransferDst,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            {family_});
        mapped_ = device_.mapMemory(readback_.getMemory(), 0, 32);
        std::vector<std::byte> seed(projected_.getSize(), std::byte{0xcd});
        if (g_) std::memcpy(seed.data(), input_.projected.data(), uint64_t(g_) * 48);
        uploadSortReplay(transfer_, projected_.getDescriptorInfo(), seed);
        initializePipelines();
        push_.gaussianCount = g_;
        push_.tileItemCount = n_;
        push_.width = h.width;
        push_.height = h.height;
        push_.pixelCount = h.width * h.height;
        push_.tileSize = 16;
        push_.optimizerEnabled = 0;
        pool_.create(device_, family_, vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
        command_ = pool_.allocateCommandBuffer();
        fence_ = device_.createFenceUnique({});
        bits_ = physical_.getQueueFamilyProperties()[family_].timestampValidBits;
        period_ = limits.timestampPeriod;
        check(bits_ && period_ > 0, "Full sort timestamps unavailable");
        queries_ = device_.createQueryPoolUnique({{}, vk::QueryType::eTimestamp, 5});
    }
    uint32_t participatingCount() const {
        return uint32_t(visible_.size());
    }
    OutputView outputView() const {
        return {projected_.getDescriptorInfo(), items_[5].getDescriptorInfo(),
                ranges_.getDescriptorInfo(), push_};
    }
    uint64_t logicalBufferBytes() const {
        uint64_t bytes = 0;
        for (const auto* buffer : {&projected_, &densification_, &prefix_, &depth_, &visibleIDs_,
                                   &ranges_, &storage_, &readback_})
            bytes += buffer->getSize();
        for (const auto& buffer : banks_)
            bytes += buffer.getSize();
        for (const auto& buffer : controls_)
            bytes += buffer.getSize();
        for (const auto& buffer : items_)
            bytes += buffer.getSize();
        return bytes;
    }
    Sample run(bool candidate) {
        return runWithConsumers(candidate, [](vk::CommandBuffer, const OutputView&) {});
    }
    // Consumers record after list timestamps. This overload does not time the combined sequence.
    template <class Consumer> Sample runWithConsumers(bool candidate, Consumer&& consume) {
        Sample out;
        candidate_ = candidate;
        begin();
        allBarrier();
        for (auto& b : banks_)
            fill(b, canary);
        for (auto& b : items_)
            fill(b, canary);
        for (auto* b : {&depth_, &visibleIDs_, &prefix_, &storage_, &ranges_})
            fill(*b, canary);
        const uint64_t densificationBytes = uint64_t(std::max(g_, 1u)) * 16;
        command_.fillBuffer(densification_.getBuffer(), 0, densificationBytes, 0);
        command_.fillBuffer(densification_.getBuffer(), densificationBytes, 64, canary);
        for (auto& b : controls_) {
            command_.fillBuffer(b.getBuffer(), 16, 64, canary);
            command_.fillBuffer(b.getBuffer(), 0, 16, 0);
        }
        allBarrier();
        out.restoreCpuMs = submit();
        const auto start = std::chrono::steady_clock::now();
        begin();
        allBarrier();
        command_.resetQueryPool(*queries_, 0, 5);
        stamp(0);
        dispatch(Clear, t_, push_);
        computeBarrier();
        dispatch(Count, g_, push_);
        computeBarrier();
        prefix(0, g_);
        if (candidate) {
            auto push = push_;
            push.validationIteration = 0;
            dispatch(Prepare, g_, push);
            computeBarrier();
            prefix(1, g_);
            push.validationIteration = 1;
            dispatch(Prepare, g_, push);
            computeBarrier();
        }
        allBarrier();
        command_.copyBuffer(controls_[0].getBuffer(), readback_.getBuffer(),
                            vk::BufferCopy{0, 0, 16});
        if (candidate)
            command_.copyBuffer(controls_[1].getBuffer(), readback_.getBuffer(),
                                vk::BufferCopy{0, 16, 16});
        vk::MemoryBarrier host{vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eHostRead};
        command_.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                 vk::PipelineStageFlagBits::eHost, {}, host, {}, {});
        stamp(1);
        submit();
        const auto* counts = static_cast<const uint32_t*>(mapped_);
        if (counts[0] != n_)
            throw std::runtime_error("GPU coverage count " + std::to_string(counts[0]) +
                                     " differs from captured raw count " + std::to_string(n_) +
                                     "; no input repair allowed");
        nv_ = candidate ? counts[4] : 0;
        check(!candidate || (nv_ <= g_ && nv_ == visible_.size()), "GPU compact count mismatch");
        begin();
        allBarrier();
        stamp(2);
        if (!candidate) {
            dispatch(Emit, g_, push_);
            computeBarrier();
            if (n_ > 1) {
                sort(items_[1], items_[3], n_, 32, false);
                computeBarrier();
                dispatch(GatherHigh, n_, push_);
                computeBarrier();
                sort(items_[4], items_[3], n_, 32, false);
                computeBarrier();
            }
            dispatch(GatherIDs, n_, push_);
            computeBarrier();
        } else {
            sort(depth_, visibleIDs_, nv_, 32, true);
            computeBarrier();
            auto push = push_;
            push.gaussianCount = nv_;
            push.validationIteration = 2;
            dispatch(Prepare, nv_, push);
            computeBarrier();
            prefix(2, nv_);
            dispatch(OrderedEmit, nv_, push);
            computeBarrier();
            sort(items_[4], items_[5], n_, tileBits(t_), true);
            computeBarrier();
        }
        stamp(4);
        dispatch(candidate ? DirectRanges : IndirectRanges, n_ + 1, push_);
        computeBarrier();
        dispatch(Finalize, t_, push_);
        computeBarrier();
        stamp(3);
        consume(command_, outputView());
        submit();
        out.cpuMs =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                .count();
        std::array<uint64_t, 5> ticks{};
        check(device_.getQueryPoolResults(*queries_, 0, 5, sizeof(ticks), ticks.data(),
                                          sizeof(uint64_t),
                                          vk::QueryResultFlagBits::e64) == vk::Result::eSuccess,
              "Full sort timestamps unavailable");
        auto elapsed = [&](uint32_t a, uint32_t b) {
            const auto value = schedulingTimestampInterval(ticks[a], ticks[b], bits_, period_);
            check(value.has_value(), "Invalid full sort timestamp interval");
            return *value;
        };
        out.prepareMs = elapsed(0, 1);
        out.mainMs = elapsed(2, 3);
        out.activeMs = out.prepareMs + out.mainMs;
        out.spanMs = elapsed(0, 3);
        out.rangeMs = elapsed(4, 3);
        return out;
    }
    void verify() {
        auto unchanged = [&](Buffer& b, uint64_t firstByte) {
            const auto bytes = download(b);
            check(std::all_of(bytes.begin() + firstByte, bytes.end(),
                              [](auto v) { return v == std::byte{0xcd}; }),
                  "Full sort unused buffer/tail changed");
        };
        auto expect = [&](Buffer& b, auto& values) {
            const auto bytes = std::as_bytes(std::span(values));
            auto result = download(b);
            check(bytes.empty() || std::memcmp(result.data(), bytes.data(), bytes.size()) == 0,
                  "Full sort exact input/intermediate/output contract failed");
            check(std::all_of(result.begin() + bytes.size(), result.end(),
                              [](auto v) { return v == std::byte{0xcd}; }),
                  "Full sort tail/canary changed");
        };
        expect(projected_, input_.projected);
        expect(banks_[0], input_.gaussianRanges);
        expect(ranges_, expected_.ranges);
        expect(items_[5], expected_.items);
        if (candidate_) {
            expect(visibleIDs_, visible_);
            std::vector<uint32_t> keys;
            std::vector<glm::uvec2> ranks, ordered;
            uint32_t rank = 0, cursor = 0;
            for (const auto& r : input_.gaussianRanges) {
                ranks.emplace_back(rank, r.y ? 1 : 0);
                rank += r.y ? 1 : 0;
            }
            for (auto g : visible_) {
                keys.push_back(depthKey(input_.projected[g].centerRadius.z));
                ordered.emplace_back(cursor, input_.gaussianRanges[g].y);
                cursor += input_.gaussianRanges[g].y;
            }
            expect(depth_, keys);
            expect(banks_[1], ranks);
            expect(banks_[2], ordered);
            expect(items_[4], expected_.high);
            const auto control = download(controls_[2]);
            uint32_t total = 0;
            std::memcpy(&total, control.data(), 4);
            check(total == n_, "Reordered prefix changed item total");
        } else {
            expect(items_[0], input_.items);
            expect(items_[2], input_.tileKeys);
            expect(items_[1], expected_.low);
            expect(items_[3], expected_.order);
            if (n_ > 1)
                expect(items_[4], expected_.high);
            else
                unchanged(items_[4], 0);
            for (auto* b : {&visibleIDs_, &depth_, &banks_[1], &banks_[2]})
                unchanged(*b, 0);
        }
        if (candidate_)
            for (size_t i = 0; i < 4; ++i)
                unchanged(items_[i], 0);
        auto guard = [&](Buffer& b, uint64_t offset) {
            std::array<uint32_t, guardWords> words{};
            transfer_.download(b.getDescriptorInfo(), offset,
                               std::as_writable_bytes(std::span(words)));
            check(std::all_of(words.begin(), words.end(), [](auto v) { return v == canary; }),
                  "Full sort workspace guard changed");
        };
        guard(storage_, scratchBytes_);
        guard(prefix_, uint64_t(prefixWords_) * 4);
        guard(densification_, uint64_t(std::max(g_, 1u)) * 16);
        for (size_t i = 0; i < controls_.size(); ++i) {
            const auto bytes = download(controls_[i]);
            std::array<uint32_t, 4> values{};
            std::memcpy(values.data(), bytes.data(), 16);
            const uint32_t total = i == 0 ? n_ : (candidate_ ? (i == 1 ? nv_ : n_) : 0u);
            check(values == std::array<uint32_t, 4>{total, 0, 0, 0},
                  "Private count bank fields changed");
            guard(controls_[i], 16);
        }
    }

private:
    void initializePipelines() {
        const std::array<const char*, 10> names{"train_forward_tile_count",
                                                "train_forward_tile_emit",
                                                "train_full_sort_prepare_test",
                                                "train_full_sort_emit_test",
                                                "train_forward_tile_gather_high",
                                                "train_forward_tile_gather_items",
                                                "train_forward_tile_range_boundaries",
                                                "train_range_boundaries_sorted_keys_test",
                                                "train_forward_tile_sort",
                                                "train_forward_tile_clear"};
        const std::array<std::vector<uint32_t>, 10> bindings{{{3, 16, 32},
                                                              {3, 4, 20, 21, 22, 32},
                                                              {3, 32, 60, 61, 62, 63},
                                                              {3, 21, 24, 32, 62},
                                                              {21, 22, 23},
                                                              {4, 22, 24},
                                                              {5, 21, 22},
                                                              {5, 23},
                                                              {5},
                                                              {5}}};
        auto info = [](Buffer& b) { return b.getDescriptorInfo(); };
        auto range = info(ranges_);
        range.range = rangeBytes_;
        const std::array<std::vector<vk::DescriptorBufferInfo>, 10> infos{
            {{info(projected_), info(densification_), info(banks_[0])},
             {info(projected_), info(items_[0]), info(items_[1]), info(items_[2]), info(items_[3]),
              info(banks_[0])},
             {info(projected_), info(banks_[0]), info(banks_[1]), info(depth_), info(visibleIDs_),
              info(banks_[2])},
             {info(projected_), info(items_[4]), info(items_[5]), info(banks_[2]),
              info(visibleIDs_)},
             {info(items_[2]), info(items_[3]), info(items_[4])},
             {info(items_[0]), info(items_[3]), info(items_[5])},
             {range, info(items_[2]), info(items_[3])},
             {range, info(items_[4])},
             {range},
             {range}}};
        const std::array<const char*, 5> prefixNames{"ranges", "scratch", "top", "add_scratch",
                                                     "add_ranges"};
        const std::array<std::vector<uint32_t>, 5> prefixBindings{
            {{32, 33}, {33}, {9, 33}, {33}, {32, 33}}};
        std::vector<vk::DescriptorSetLayout> layouts;
        auto initialize = [&](ComputePipeline& p, const std::string& name, const auto& bs,
                              size_t pushBytes) {
            ComputePipelineConfig config;
            config.pushConstantSize = uint32_t(pushBytes);
            for (auto b : bs)
                config.descriptorBindings.push_back(backwardStorageBinding(b));
            p.initialize(device_, "shaders/" + name + ".comp.spv", config);
        };
        for (size_t i = 0; i < 10; ++i) {
            initialize(pipelines_[i], names[i], bindings[i], sizeof(TrainingPushConstants));
            layouts.push_back(pipelines_[i].getDescriptorSetLayout());
        }
        for (size_t i = 0; i < 5; ++i)
            initialize(prefixPipelines_[i],
                       std::string("train_forward_gaussian_prefix_") + prefixNames[i],
                       prefixBindings[i], sizeof(TrainingPrefixPushConstants));
        for (size_t b = 0; b < 3; ++b)
            for (auto& p : prefixPipelines_)
                layouts.push_back(p.getDescriptorSetLayout());
        const auto sets = descriptors_.create(device_, 72, 0, layouts);
        std::copy_n(sets.begin(), 10, sets_.begin());
        auto write = [&](vk::DescriptorSet set, const auto& bs, const auto& ds) {
            std::vector<vk::WriteDescriptorSet> writes(bs.size());
            for (size_t j = 0; j < bs.size(); ++j)
                writes[j]
                    .setDstSet(set)
                    .setDstBinding(bs[j])
                    .setDescriptorCount(1)
                    .setDescriptorType(vk::DescriptorType::eStorageBuffer)
                    .setPBufferInfo(&ds[j]);
            device_.updateDescriptorSets(writes, {});
        };
        for (size_t i = 0; i < 10; ++i)
            write(sets_[i], bindings[i], infos[i]);
        for (size_t b = 0; b < 3; ++b) {
            const std::array<std::vector<vk::DescriptorBufferInfo>, 5> ds{
                {{info(banks_[b]), info(prefix_)},
                 {info(prefix_)},
                 {info(controls_[b]), info(prefix_)},
                 {info(prefix_)},
                 {info(banks_[b]), info(prefix_)}}};
            for (size_t i = 0; i < 5; ++i) {
                prefixSets_[b][i] = sets[10 + b * 5 + i];
                write(prefixSets_[b][i], prefixBindings[i], ds[i]);
            }
        }
    }
    void prefix(size_t bank, uint32_t count) {
        if (!count) return;
        struct Level {
            uint32_t offset, count;
        };
        std::vector<Level> levels{{0, (count + 255) / 256}};
        uint32_t next = levels[0].count;
        auto dispatchPrefix = [&](size_t i, TrainingPrefixPushConstants push, uint32_t groups) {
            auto& p = prefixPipelines_[i];
            command_.bindPipeline(vk::PipelineBindPoint::eCompute, p.getPipeline());
            command_.bindDescriptorSets(vk::PipelineBindPoint::eCompute, p.getPipelineLayout(), 0,
                                        prefixSets_[bank][i], {});
            command_.pushConstants(p.getPipelineLayout(), vk::ShaderStageFlagBits::eCompute, 0,
                                   sizeof(push), &push);
            command_.dispatch(groups, 1, 1);
            computeBarrier();
        };
        TrainingPrefixPushConstants p{};
        p.elementCount = count;
        dispatchPrefix(0, p, levels[0].count);
        while (levels.back().count > 256) {
            const auto child = levels.back();
            Level parent{next, (child.count + 255) / 256};
            levels.push_back(parent);
            next += parent.count;
            p = {};
            p.elementCount = child.count;
            p.inputOffset = child.offset;
            p.outputOffset = parent.offset;
            dispatchPrefix(1, p, parent.count);
        }
        p = {};
        p.elementCount = levels.back().count;
        p.inputOffset = levels.back().offset;
        dispatchPrefix(2, p, 1);
        for (size_t i = levels.size(); i > 1; --i) {
            p = {};
            p.elementCount = levels[i - 2].count;
            p.inputOffset = levels[i - 2].offset;
            p.parentOffset = levels[i - 1].offset;
            dispatchPrefix(3, p, (p.elementCount + 255) / 256);
        }
        p = {};
        p.elementCount = count;
        p.inputOffset = levels[0].offset;
        dispatchPrefix(4, p, (count + 255) / 256);
    }
    void sort(Buffer& keys, Buffer& values, uint32_t count, uint32_t bits, bool diagnostic) {
        if (count <= 1) return;
        const auto reuse = trainingSortScratchReuseBarrier(storage_.getBuffer());
        command_.pipelineBarrier(kTrainingSortScratchReuseSourceStages,
                                 kTrainingSortScratchReuseDestinationStages, {}, 0, nullptr, 1,
                                 &reuse, 0, nullptr);
        if (diagnostic)
            recordDiagnosticRadixKeyValue(command_, sorter_.get(), count, keys.getBuffer(),
                                          values.getBuffer(), storage_.getBuffer(), bits);
        else
            vrdxCmdSortKeyValue(command_, sorter_.get(), count, keys.getBuffer(), 0,
                                values.getBuffer(), 0, storage_.getBuffer(), 0, {}, 0);
    }
    void dispatch(Pass i, uint32_t elements, const TrainingPushConstants& push) {
        if (!elements) return;
        auto& p = pipelines_[i];
        command_.bindPipeline(vk::PipelineBindPoint::eCompute, p.getPipeline());
        command_.bindDescriptorSets(vk::PipelineBindPoint::eCompute, p.getPipelineLayout(), 0,
                                    sets_[i], {});
        command_.pushConstants(p.getPipelineLayout(), vk::ShaderStageFlagBits::eCompute, 0,
                               sizeof(push), &push);
        command_.dispatch((elements + 255) / 256, 1, 1);
    }
    void begin() {
        command_.reset();
        command_.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    }
    double submit() {
        command_.end();
        device_.resetFences(*fence_);
        vk::SubmitInfo submit;
        submit.setCommandBuffers(command_);
        const auto start = std::chrono::steady_clock::now();
        queue_.submit(submit, *fence_);
        check(device_.waitForFences(*fence_, true, UINT64_MAX) == vk::Result::eSuccess,
              "Full sort fence wait failed");
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
            .count();
    }
    void allBarrier() {
        vk::MemoryBarrier b{vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
                            vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite};
        command_.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                                 vk::PipelineStageFlagBits::eAllCommands, {}, b, {}, {});
    }
    void computeBarrier() {
        vk::MemoryBarrier b{vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
                            vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite};
        command_.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                 vk::PipelineStageFlagBits::eComputeShader, {}, b, {}, {});
    }
    void stamp(uint32_t i) {
        command_.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands, *queries_, i);
    }
    void fill(Buffer& b, uint32_t value) {
        command_.fillBuffer(b.getBuffer(), 0, b.getSize(), value);
    }
    std::vector<std::byte> download(Buffer& b) {
        std::vector<std::byte> bytes(b.getSize());
        downloadSortReplay(transfer_, b.getDescriptorInfo(), bytes);
        return bytes;
    }
    vk::Device device_;
    vk::PhysicalDevice physical_;
    vk::Queue queue_;
    uint32_t family_;
    const Inputs& input_;
    Oracle expected_;
    std::vector<uint32_t> visible_;
    uint32_t g_ = 0, n_ = 0, t_ = 0, nv_ = 0, prefixWords_ = 0, bits_ = 0;
    uint64_t itemBytes_ = 0, gBytes_ = 0, rangeBytes_ = 0, scratchBytes_ = 0;
    double period_ = 0;
    bool candidate_ = false;
    void* mapped_ = nullptr;
    OptimizerCaptureTransfer transfer_;
    std::unique_ptr<std::remove_pointer_t<VrdxSorter>, decltype(&vrdxDestroySorter)> sorter_{
        nullptr, &vrdxDestroySorter};
    Buffer projected_, densification_, prefix_, depth_, visibleIDs_, ranges_, storage_, readback_;
    std::array<Buffer, 3> banks_, controls_;
    std::array<Buffer, 6> items_;
    std::array<ComputePipeline, 10> pipelines_;
    std::array<ComputePipeline, 5> prefixPipelines_;
    BackwardDescriptorPool descriptors_;
    std::array<vk::DescriptorSet, 10> sets_{};
    std::array<std::array<vk::DescriptorSet, 5>, 3> prefixSets_{};
    TrainingPushConstants push_{};
    CommandPool pool_;
    vk::CommandBuffer command_;
    vk::UniqueFence fence_;
    vk::UniqueQueryPool queries_;
};
} // namespace full_sort
