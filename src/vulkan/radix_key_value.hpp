#pragma once
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vk_radix_sort.h>

namespace vulkan3DGS {
// External key/value descriptors use the sorter's aligned byte span; active counts stay unchanged.
inline uint32_t radixKeyValueCapacity(uint32_t count, VkDeviceSize alignment) {
    if (!alignment || (alignment & (alignment - 1)))
        throw std::invalid_argument("Radix buffer alignment must be a nonzero power of two");
    alignment = std::max<VkDeviceSize>(alignment, sizeof(uint32_t));
    if (alignment > uint64_t(std::numeric_limits<uint32_t>::max()) * sizeof(uint32_t))
        throw std::overflow_error("Radix buffer alignment exceeds uint32 capacity");
    const uint64_t bytes = uint64_t(count) * sizeof(uint32_t);
    const uint64_t capacity = ((bytes + alignment - 1) / alignment * alignment) / sizeof(uint32_t);
    if (capacity > std::numeric_limits<uint32_t>::max())
        throw std::overflow_error("Aligned radix buffer capacity exceeds uint32");
    return uint32_t(capacity);
}
// Uint32 storage, stable 16/32 effective-bit keys, in-place output after an even pass count.
void recordRadixKeyValue(VkCommandBuffer command, VrdxSorter sorter, uint32_t count, VkBuffer keys,
                         VkBuffer values, VkBuffer storage, uint32_t effectiveBits);
VkPipelineLayout radixKeyValuePipelineLayout(VrdxSorter sorter);
struct RadixKey16Layout {
    VkDeviceSize countBytes, histogramBytes, keyBytes, valueBytes, valueOffset, storageBytes;
};
RadixKey16Layout radixKey16Layout(uint32_t count, VkDeviceSize alignment);
void recordRadixKeyValue16(VkCommandBuffer command, VrdxSorter sorter, uint32_t count,
                           VkBuffer keys, VkBuffer values, VkBuffer storage, VkPipeline upsweep,
                           VkPipeline downsweep);
} // namespace vulkan3DGS
