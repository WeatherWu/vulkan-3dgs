#include "graphics_pipeline_set.hpp"

#include "context/context.hpp"

namespace vulkan3DGS {

GraphicsPipelineSet::~GraphicsPipelineSet() {
    cleanup();
}

void GraphicsPipelineSet::initialize(vk::Format imageFormat, vk::Extent2D extent) {
    cleanup();
    renderPass_.initialize(imageFormat);
    const vk::Device device = Context::Instance().getDevice().getDevice();
    legacy_.initialize(device, renderPass_.getRenderPass(), extent);
    compatible_.initialize(device, renderPass_.getRenderPass(), extent, true);
}

void GraphicsPipelineSet::recreate(vk::Format imageFormat, vk::Extent2D extent) {
    cleanup();
    initialize(imageFormat, extent);
}

void GraphicsPipelineSet::cleanup() {
    compatible_.cleanup();
    legacy_.cleanup();
    renderPass_.cleanup();
}

Pipeline& GraphicsPipelineSet::active(GaussianRenderProfile profile) {
    return profile == GaussianRenderProfile::SuperSplatCompatible ? compatible_ : legacy_;
}

} // namespace vulkan3DGS
