#include "graphics_command_recorder.hpp"

#include "graphics_splat_sorter.hpp"
#include "pipeline.hpp"

#include <imgui.h>
#include <imgui_impl_vulkan.h>

#include <array>

namespace vulkan3DGS {

void GraphicsCommandRecorder::record(const GraphicsRecordContext& context) const {
    vk::CommandBufferBeginInfo beginInfo{};
    beginInfo.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    context.commandBuffer.begin(beginInfo);

    context.sorter.record(context.commandBuffer, context.computeDescriptorSet);
    context.sorter.recordReadBarrier(context.commandBuffer);

    std::array<vk::ClearValue, 2> clearValues{};
    clearValues[0].setColor(vk::ClearColorValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}));
    clearValues[1].setDepthStencil(vk::ClearDepthStencilValue(1.0f, 0));
    vk::RenderPassBeginInfo renderPassInfo{};
    renderPassInfo.setRenderPass(context.renderPass)
        .setFramebuffer(context.framebuffer)
        .setRenderArea(vk::Rect2D({0, 0}, context.extent))
        .setClearValueCount(static_cast<uint32_t>(clearValues.size()))
        .setPClearValues(clearValues.data());
    context.commandBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);

    const vk::Viewport viewport(0.0f, 0.0f, static_cast<float>(context.extent.width),
                                static_cast<float>(context.extent.height), 0.0f, 1.0f);
    const vk::Rect2D scissor({0, 0}, context.extent);
    context.commandBuffer.setViewport(0, 1, &viewport);
    context.commandBuffer.setScissor(0, 1, &scissor);
    context.commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics,
                                       context.pipeline.getPipeline());

    const vk::Buffer vertexBuffer = context.pipeline.getQuadVertexBuffer();
    constexpr vk::DeviceSize offset = 0;
    context.commandBuffer.bindVertexBuffers(0, 1, &vertexBuffer, &offset);
    context.commandBuffer.bindIndexBuffer(context.pipeline.getQuadIndexBuffer(), 0,
                                          vk::IndexType::eUint16);
    if (context.graphicsDescriptorSet) {
        context.commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                                                 context.pipeline.getPipelineLayout(), 0, 1,
                                                 &context.graphicsDescriptorSet, 0, nullptr);
    }

    if (context.hasModel) {
        if (context.sorter.indirectBuffer()) {
            context.commandBuffer.drawIndexedIndirect(context.sorter.indirectBuffer(), 0, 1,
                                                      sizeof(VkDrawIndexedIndirectCommand));
        } else {
            context.commandBuffer.drawIndexed(4, context.pointCount, 0, 0, 0);
        }
    }
    if (context.drawImGui) {
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), context.commandBuffer);
    }
    context.commandBuffer.endRenderPass();
    context.commandBuffer.end();
}

} // namespace vulkan3DGS
