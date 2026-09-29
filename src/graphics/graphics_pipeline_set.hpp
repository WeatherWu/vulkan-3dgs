#pragma once

#include "pipeline.hpp"
#include "render_profile.hpp"
#include "renderpass.hpp"

#include <vulkan/vulkan.hpp>

namespace vulkan3DGS {

class GraphicsPipelineSet {
public:
    GraphicsPipelineSet() = default;
    ~GraphicsPipelineSet();

    GraphicsPipelineSet(const GraphicsPipelineSet&) = delete;
    GraphicsPipelineSet& operator=(const GraphicsPipelineSet&) = delete;

    void initialize(vk::Format imageFormat, vk::Extent2D extent);
    void recreate(vk::Format imageFormat, vk::Extent2D extent);
    void cleanup();

    Pipeline& active(GaussianRenderProfile profile);
    const Pipeline& legacy() const {
        return legacy_;
    }
    vk::RenderPass renderPass() const {
        return renderPass_.getRenderPass();
    }
    vk::DescriptorSetLayout descriptorSetLayout() const {
        return legacy_.getDescriptorSetLayout();
    }

private:
    RenderPass renderPass_;
    Pipeline legacy_;
    Pipeline compatible_;
};

} // namespace vulkan3DGS
