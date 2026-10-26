#pragma once
#include <vk_radix_sort.h>

namespace vulkan3DGS {
// Uint32 storage, stable 16/32 effective-bit keys, in-place output after an even pass count.
void recordRadixKeyValue(VkCommandBuffer command, VrdxSorter sorter, uint32_t count, VkBuffer keys,
                         VkBuffer values, VkBuffer storage, uint32_t effectiveBits);
} // namespace vulkan3DGS
