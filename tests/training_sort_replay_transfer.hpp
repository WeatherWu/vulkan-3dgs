#pragma once

#include "training/core/training_optimizer_capture.hpp"

#include <algorithm>
#include <span>
#include <stdexcept>

namespace sort_replay {

template<class Byte, class Operation>
void forEachReplayTransferChunk(vk::DescriptorBufferInfo buffer, std::span<Byte> bytes,
                                Operation operation) {
    if (bytes.size() > buffer.range)
        throw std::runtime_error("Sort replay transfer exceeds buffer range");
    for (size_t offset = 0; offset < bytes.size();) {
        const auto count = std::min(vulkan3DGS::optimizerCaptureChunkBytes, bytes.size() - offset);
        operation(buffer, offset, bytes.subspan(offset, count));
        offset += count;
    }
}

template<class Transfer>
void uploadSortReplay(Transfer& transfer, vk::DescriptorBufferInfo destination,
                       std::span<const std::byte> bytes) {
    forEachReplayTransferChunk(destination, bytes, [&](auto buffer, size_t offset, auto chunk) {
        transfer.upload(buffer, offset, chunk);
    });
}

template<class Transfer>
void downloadSortReplay(Transfer& transfer, vk::DescriptorBufferInfo source,
                         std::span<std::byte> bytes) {
    forEachReplayTransferChunk(source, bytes, [&](auto buffer, size_t offset, auto chunk) {
        transfer.download(buffer, offset, chunk);
    });
}

} // namespace sort_replay
