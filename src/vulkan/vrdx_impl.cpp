#define VRDX_IMPLEMENTATION
#include <vk_radix_sort.h>

#include "vulkan/radix_key_value.hpp"
#include <stdexcept>
#include <utility>

void vulkan3DGS::recordRadixKeyValue(VkCommandBuffer command, VrdxSorter sorter, uint32_t count,
                                     VkBuffer keys, VkBuffer values, VkBuffer storage,
                                     uint32_t bits) {
    if (bits != 16 && bits != 32)
        throw std::invalid_argument("Diagnostic radix expects 16/32 effective bits");
    if (count <= 1) return;
    if (!command || !sorter || !keys || !values || !storage)
        throw std::invalid_argument("Diagnostic radix missing resource");
    const uint32_t passes = bits / 8;
    const auto align = sorter->minStorageBufferOffsetAlignment;
    const VkDeviceSize countBytes = Align(4, align);
    const VkDeviceSize histogramBytes = HistogramSize(count, align);
    const VkDeviceSize inoutBytes = InoutSize(count, align);
    const VkDeviceSize temporaryOffset = countBytes + histogramBytes;
    vkCmdUpdateBuffer(command, storage, 0, 4, &count);
    vkCmdFillBuffer(command, storage, countBytes, 4 * RADIX * 4, 0);
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0,
                         nullptr);
    for (uint32_t pass = 0; pass < passes; ++pass) {
        VkDescriptorBufferInfo infos[7] = {{storage, 0, 4},
                                           {storage, countBytes, 4 * RADIX * 4},
                                           {storage, countBytes + 4 * RADIX * 4, VK_WHOLE_SIZE},
                                           {keys, 0, inoutBytes},
                                           {storage, temporaryOffset, inoutBytes},
                                           {values, 0, inoutBytes},
                                           {storage, temporaryOffset + inoutBytes, inoutBytes}};
        if (pass & 1u) {
            std::swap(infos[3], infos[4]);
            std::swap(infos[5], infos[6]);
        }
        VkWriteDescriptorSet writes[7]{};
        for (uint32_t i = 0; i < 7; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i];
        }
        sorter->cmdPushDescriptorSet(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                                     sorter->pipelineLayout, 0, 7, writes);
        PushConstants push{};
        push.pass = pass;
        vkCmdPushConstants(command, sorter->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(push), &push);
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, sorter->upsweepPipeline);
        vkCmdDispatch(command, RoundUp(count, PARTITION_SIZE), 1, 1);
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0,
                             nullptr);
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, sorter->spinePipeline);
        vkCmdDispatch(command, RADIX, 1, 1);
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0,
                             nullptr);
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                          sorter->downsweepKeyValuePipeline);
        vkCmdDispatch(command, RoundUp(count, PARTITION_SIZE), 1, 1);
        if (pass + 1 < passes)
            vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr,
                                 0, nullptr);
    }
}
