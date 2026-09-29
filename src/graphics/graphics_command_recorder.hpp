#pragma once

#include <vulkan/vulkan.hpp>

#include <cstdint>

namespace vulkan3DGS {

class GraphicsSplatSorter;
class Pipeline;

struct GraphicsRecordContext {
    vk::CommandBuffer commandBuffer;
    vk::RenderPass renderPass;
    vk::Framebuffer framebuffer;
    vk::Extent2D extent;
    Pipeline& pipeline;
    vk::DescriptorSet graphicsDescriptorSet;
    vk::DescriptorSet computeDescriptorSet;
    const GraphicsSplatSorter& sorter;
    bool hasModel = false;
    uint32_t pointCount = 0;
    bool drawImGui = false;
};

class GraphicsCommandRecorder {
public:
    void record(const GraphicsRecordContext& context) const;
};

} // namespace vulkan3DGS
