#include "training_sort_replay_reference.hpp"
#include "training_sort_replay_transfer.hpp"
#include "training/core/training_sort_synchronization.hpp"

#include <iostream>

using namespace sort_replay;

namespace {
void synchronizationTests() {
    const auto scratch = vk::Buffer(reinterpret_cast<VkBuffer>(uintptr_t{0x1234}));
    const auto barrier = trainingSortScratchReuseBarrier(scratch);
    check(barrier.buffer == scratch && barrier.offset == 0 && barrier.size == VK_WHOLE_SIZE &&
              barrier.srcQueueFamilyIndex == VK_QUEUE_FAMILY_IGNORED &&
              barrier.dstQueueFamilyIndex == VK_QUEUE_FAMILY_IGNORED,
          "Scratch reuse dependency must cover only its buffer, without queue ownership transfer");
    check(barrier.srcAccessMask == (vk::AccessFlagBits::eTransferWrite |
              vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite) &&
              barrier.dstAccessMask == vk::AccessFlagBits::eTransferWrite &&
              kTrainingSortScratchReuseSourceStages ==
                  (vk::PipelineStageFlagBits::eTransfer | vk::PipelineStageFlagBits::eComputeShader) &&
              kTrainingSortScratchReuseDestinationStages == vk::PipelineStageFlagBits::eTransfer,
          "Scratch reuse must order transfer resets and compute accesses before transfer writes");
}
TrainingScanlineCapture synchronizationFixture() {
    TrainingScanlineCapture source;
    source.header.gaussianCount = 5;
    source.header.width = 32;
    source.header.height = 16;
    source.header.itemCount = 5;
    source.header.tileCount = 2;
    source.projected.resize(5);
    constexpr std::array<float, 5> depths{2, 1, 1, 3, 2};
    for (size_t i = 0; i < source.projected.size(); ++i) {
        source.projected[i].centerRadius = {i % 2 ? 24.0f : 8.0f, 8, depths[i], 4};
        source.projected[i].conicOpacity = {1, 0, 1, 0.5f};
    }
    source.items = {2, 0, 4, 1, 3};
    source.ranges = {{0, 3, 0, 0}, {3, 2, 0, 0}};
    source.blends.resize(32 * 16);
    source.pixelGrads.resize(32 * 16);
    return source;
}
struct TransferCall {
    uint64_t descriptorOffset, offset;
    size_t bytes;
};
struct RecordingTransfer {
    std::vector<std::byte> deviceBytes;
    std::vector<TransferCall> uploads, downloads;
    void upload(vk::DescriptorBufferInfo buffer, uint64_t offset, std::span<const std::byte> bytes) {
        check(bytes.size() <= optimizerCaptureChunkBytes && offset <= buffer.range &&
                  bytes.size() <= buffer.range - offset,
              "Upload exceeds single-chunk staging bounds");
        check(buffer.offset <= deviceBytes.size() && offset <= deviceBytes.size() - buffer.offset &&
                  bytes.size() <= deviceBytes.size() - buffer.offset - offset,
              "Upload exceeds backing allocation");
        uploads.push_back({buffer.offset, offset, bytes.size()});
        std::copy(bytes.begin(), bytes.end(), deviceBytes.begin() + buffer.offset + offset);
    }
    void download(vk::DescriptorBufferInfo buffer, uint64_t offset, std::span<std::byte> bytes) {
        check(bytes.size() <= optimizerCaptureChunkBytes && offset <= buffer.range &&
                  bytes.size() <= buffer.range - offset,
              "Download exceeds single-chunk staging bounds");
        check(buffer.offset <= deviceBytes.size() && offset <= deviceBytes.size() - buffer.offset &&
                  bytes.size() <= deviceBytes.size() - buffer.offset - offset,
              "Download exceeds backing allocation");
        downloads.push_back({buffer.offset, offset, bytes.size()});
        std::copy_n(deviceBytes.begin() + buffer.offset + offset, bytes.size(), bytes.begin());
    }
};
void verifyCalls(const std::vector<TransferCall>& calls, size_t bytes) {
    size_t cursor = 0;
    for (const auto& call : calls) {
        check(call.descriptorOffset == 16 && call.offset == cursor &&
                  call.bytes == std::min(optimizerCaptureChunkBytes, bytes - cursor),
              "Transfer chunks overlap, skip bytes or ignore descriptor offset");
        cursor += call.bytes;
    }
    check(cursor == bytes, "Transfer does not cover full payload");
}
void transferTests() {
    constexpr auto chunk = optimizerCaptureChunkBytes;
    for (size_t bytes : {size_t(0), size_t(4), chunk - 4, chunk, chunk + 4,
                         2 * chunk, 2 * chunk + 12, size_t(59910000)}) {
        std::vector<std::byte> input(bytes), output(bytes);
        for (size_t i = 0; i < bytes; ++i)
            input[i] = std::byte((i * 37 + (i / chunk) * 13) & 255u);
        RecordingTransfer transfer;
        transfer.deviceBytes.assign(bytes + 32, std::byte{0xA5});
        vk::DescriptorBufferInfo buffer;
        buffer.offset = 16;
        buffer.range = bytes;
        uploadSortReplay(transfer, buffer, std::span<const std::byte>(input));
        downloadSortReplay(transfer, buffer, std::span<std::byte>(output));
        verifyCalls(transfer.uploads, bytes);
        verifyCalls(transfer.downloads, bytes);
        check(input == output, "Chunked roundtrip altered payload");
        for (size_t i = 0; i < 16; ++i)
            check(transfer.deviceBytes[i] == std::byte{0xA5} &&
                      transfer.deviceBytes[16 + bytes + i] == std::byte{0xA5},
                  "Chunked transfer altered buffer canary");
    }
    RecordingTransfer transfer;
    std::vector<std::byte> bytes(chunk + 4);
    vk::DescriptorBufferInfo tooSmall;
    tooSmall.range = bytes.size() - 1;
    bool uploadRejected = false, downloadRejected = false;
    try { uploadSortReplay(transfer, tooSmall, std::span<const std::byte>(bytes)); }
    catch (const std::runtime_error&) { uploadRejected = true; }
    try { downloadSortReplay(transfer, tooSmall, std::span<std::byte>(bytes)); }
    catch (const std::runtime_error&) { downloadRejected = true; }
    check(uploadRejected && downloadRejected && transfer.uploads.empty() && transfer.downloads.empty(),
          "Invalid full-buffer range must be rejected before any partial transfer");
    vk::DescriptorBufferInfo valid;
    valid.range = bytes.size();
    size_t calls = 0;
    bool propagated = false;
    try {
        forEachReplayTransferChunk(valid, std::span<std::byte>(bytes),
            [&](auto, size_t, auto) {
                ++calls;
                throw std::runtime_error("Transfer failure");
            });
    } catch (const std::runtime_error&) { propagated = true; }
    check(propagated && calls == 1, "Failed transfer must propagate without retrying");
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 3 && std::string_view(argv[1]) == "--write-sync-fixture") {
            writeTrainingScanlineCapture(argv[2], synchronizationFixture());
            return 0;
        }
        check(argc == 1, "Usage: training_sort_replay_tests [--write-sync-fixture NEW.bin]");
        synchronizationTests();
        transferTests();
        TrainingScanlineCapture source;
        source.header.gaussianCount = 3;
        source.header.width = 32;
        source.header.height = 16;
        source.header.itemCount = 3;
        source.header.tileCount = 2;
        source.projected.resize(3);
        source.projected[0].centerRadius.z = 2;
        source.projected[1].centerRadius.z = 1;
        source.projected[2].centerRadius.z = 1;
        source.items = {1, 0, 0};
        source.ranges = {{0, 2, 0, 0}, {2, 1, 0, 0}};
        Inputs input;
        input.header.source = source.header;
        input.header.capacity = 8;
        input.projected = source.projected;
        input.gaussianRanges = {{0, 2}, {2, 1}, {3, 0}};
        input.items = {0, 0, 1};
        input.depthKeys = {depthKey(2), depthKey(2), depthKey(1)};
        input.tileKeys = {0, 1, 0};
        input.indices = {0, 1, 2};
        verify(input, source);
        check(serialize(input).size() == 104 + 56 * 3 + 16 * 3, "Serialized size mismatch");
        auto reject = [&](auto mutate) {
            auto invalid = input;
            mutate(invalid);
            bool rejected = false;
            try { verify(invalid, source); } catch (const std::runtime_error&) { rejected = true; }
            check(rejected, "Invalid raw sort input accepted");
        };
        reject([](Inputs& x) { x.header.version = 2; });
        reject([](Inputs& x) { x.header.source.frame = 1; });
        reject([](Inputs& x) { x.header.capacity = 2; });
        reject([](Inputs& x) { x.projected[0].centerRadius.z = 5; });
        reject([](Inputs& x) { x.gaussianRanges[2].x = 2; });
        reject([](Inputs& x) { x.items[2] = 2; });
        reject([](Inputs& x) { x.depthKeys[0] = 0; });
        reject([](Inputs& x) { x.indices[0] = 2; });
        reject([](Inputs& x) { x.tileKeys[0] = 2; });
        reject([](Inputs& x) { x.tileKeys[0] = 1; });
        std::cout << "Sort replay CPU contracts passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
