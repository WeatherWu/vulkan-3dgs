#pragma once
#include "training/core/training_optimizer_capture_gpu.hpp"
#include "utils/file_utils.hpp"
#include "vulkan/radix_key16.hpp"
#include <algorithm>
#include <cstring>
#include <numeric>
#include <sstream>

namespace key16_checks {
using namespace vulkan3DGS;
inline void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
inline uint32_t run(vk::Device device, vk::PhysicalDevice physical, vk::Queue queue,
                    uint32_t family) {
    for (const char* module :
         {"train_tile_key16_upsweep", "train_tile_key16_downsweep", "train_full_sort_emit_test_u16",
          "train_range_boundaries_sorted_keys_test_u16"}) {
        const auto bytes =
            FileUtils::readBinaryFile(std::string("shaders/") + module + ".comp.spv");
        require(bytes.size() % 4 == 0, "Invalid uint16 SPIR-V length");
        std::vector<uint32_t> words(bytes.size() / 4);
        std::memcpy(words.data(), bytes.data(), bytes.size());
        require(words.size() >= 5 && words[0] == 0x07230203, "Invalid uint16 SPIR-V header");
        bool type16 = false, stride2 = false;
        for (size_t at = 5; at < words.size();) {
            const uint32_t length = words[at] >> 16, opcode = words[at] & 65535;
            require(length && at + length <= words.size(), "Invalid SPIR-V instruction");
            if (opcode == 21 && length >= 4 && words[at + 2] == 16) type16 = true;
            if (opcode == 71 && length >= 4 && words[at + 2] == 6 && words[at + 3] == 2)
                stride2 = true;
            at += length;
        }
        require(type16 && stride2, "Compiler did not produce physical two-byte key storage");
    }
    struct Sorter {
        VrdxSorter handle = nullptr;
        ~Sorter() {
            if (handle) vrdxDestroySorter(handle);
        }
    } sorter;
    VrdxSorterCreateInfo create{};
    create.device = device;
    create.physicalDevice = physical;
    vrdxCreateSorter(&create, &sorter.handle);
    RadixKey16 sort16(device, physical, queue, family, sorter.handle);
    OptimizerCaptureTransfer transfer(device, physical, queue, family);
    CommandPool pool;
    pool.create(device, family, vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
    auto command = pool.allocateCommandBuffer();
    auto fence = device.createFenceUnique({});
    uint32_t cases = 0;
    const auto usage = vk::BufferUsageFlagBits::eStorageBuffer |
                       vk::BufferUsageFlagBits::eTransferSrc |
                       vk::BufferUsageFlagBits::eTransferDst;
    auto buffer = [&](uint64_t bytes) {
        Buffer out;
        out.create(device, physical, queue, family, nullptr, bytes, usage,
                   vk::MemoryPropertyFlagBits::eDeviceLocal, {family});
        return out;
    };
    auto upload = [&](vk::DescriptorBufferInfo target, std::span<const std::byte> data) {
        for (size_t at = 0; at < data.size();) {
            auto part =
                data.subspan(at, std::min(data.size() - at, size_t(optimizerCaptureChunkBytes)));
            transfer.upload(target, at, part);
            at += part.size();
        }
    };
    auto download = [&](vk::DescriptorBufferInfo source) {
        std::vector<std::byte> data(size_t(source.range));
        for (size_t at = 0; at < data.size();) {
            auto part = std::span(data).subspan(
                at, std::min(data.size() - at, size_t(optimizerCaptureChunkBytes)));
            transfer.download(source, at, part);
            at += part.size();
        }
        return data;
    };
    for (uint32_t count :
         {0u, 1u, 2u, 255u, 256u, 257u, 821u, 822u, 823u, 4095u, 4096u, 4097u, 65537u, 3u}) {
        sort16.ensure(count);
        const auto layout = radixKey16Layout(
            std::max(count, 1u), physical.getProperties().limits.minStorageBufferOffsetAlignment);
        auto values = buffer(layout.valueBytes + 64);
        auto keys32 = buffer(layout.valueBytes + 64);
        auto values32 = buffer(layout.valueBytes + 64);
        VrdxSorterStorageRequirements required{};
        vrdxGetSorterKeyValueStorageRequirements(sorter.handle, std::max(count, 2u), &required);
        auto scratch32 = buffer(required.size);
        for (uint32_t pattern = 0; pattern < 3; ++pattern) {
            std::vector<uint16_t> input(size_t(sort16.keysInfo().range / 2), 0xbeef);
            std::vector<uint32_t> refKeys(size_t(keys32.getSize() / 4), 0xcdcdcdcd),
                payload(size_t(values.getSize() / 4), 0xcdcdcdcd);
            for (uint32_t i = 0; i < count; ++i) {
                input[i] = pattern == 0   ? uint16_t((i * 73u) ^ ((i % 17u) * 321u))
                           : pattern == 1 ? uint16_t(65535)
                                          : uint16_t((count - i) % 257);
                refKeys[i] = input[i];
                payload[i] = 0xfed00000u + i;
            }
            upload(sort16.keysInfo(), std::as_bytes(std::span(input)));
            upload(values.getDescriptorInfo(), std::as_bytes(std::span(payload)));
            upload(keys32.getDescriptorInfo(), std::as_bytes(std::span(refKeys)));
            upload(values32.getDescriptorInfo(), std::as_bytes(std::span(payload)));
            std::vector<std::byte> scratch(size_t(sort16.scratchInfo().range), std::byte{0xcd});
            upload(sort16.scratchInfo(), scratch);
            command.reset();
            command.begin(
                vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
            sort16.record(command, count, values.getBuffer());
            recordRadixKeyValue(command, sorter.handle, count, keys32.getBuffer(),
                                values32.getBuffer(), scratch32.getBuffer(), 16);
            command.end();
            device.resetFences(*fence);
            vk::SubmitInfo submit;
            submit.setCommandBuffers(command);
            queue.submit(submit, *fence);
            (void)device.waitForFences(*fence, true, UINT64_MAX);
            const auto actualKeys = download(sort16.keysInfo()),
                       actualValues = download(values.getDescriptorInfo());
            const auto referenceValues = download(values32.getDescriptorInfo());
            require(std::memcmp(actualValues.data(), referenceValues.data(), payload.size() * 4) ==
                        0,
                    "uint16/uint32 stable payload mismatch");
            std::vector<uint32_t> order(count);
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(),
                             [&](auto a, auto b) { return input[a] < input[b]; });
            for (uint32_t i = 0; i < count; ++i) {
                uint16_t key;
                uint32_t value;
                std::memcpy(&key, actualKeys.data() + 2ull * i, 2);
                std::memcpy(&value, actualValues.data() + 4ull * i, 4);
                require(key == input[order[i]] && value == payload[order[i]],
                        "uint16 sort CPU oracle mismatch");
            }
            require(std::memcmp(actualKeys.data() + 2ull * count,
                                reinterpret_cast<const std::byte*>(input.data()) + 2ull * count,
                                actualKeys.size() - 2ull * count) == 0,
                    "uint16 odd/tail key overwrite");
            require(std::memcmp(actualValues.data() + 4ull * count,
                                reinterpret_cast<const std::byte*>(payload.data()) + 4ull * count,
                                actualValues.size() - 4ull * count) == 0,
                    "uint16 payload tail overwrite");
            const auto after = download(sort16.scratchInfo());
            require(std::all_of(after.end() - 64, after.end(),
                                [](auto b) { return b == std::byte{0xcd}; }),
                    "uint16 scratch canary overwritten");
            ++cases;
        }
    }
    return cases;
}
} // namespace key16_checks
