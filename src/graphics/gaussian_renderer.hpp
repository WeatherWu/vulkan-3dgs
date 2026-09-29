#pragma once

#include "graphics_command_recorder.hpp"
#include "graphics_frame_runtime.hpp"
#include "graphics_pipeline_set.hpp"
#include "graphics_splat_resources.hpp"
#include "graphics_splat_sorter.hpp"
#include "render/renderer.hpp"
#include "render_profile.hpp"
#include "viewer/camera.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace vulkan3DGS {

class GaussianModel;

// Facade for the normal PLY graphics path. Focused collaborators own Vulkan
// frame, pipeline, resource, sorting, and command-recording details.
class GaussianRenderer : public Renderer {
public:
    GaussianRenderer();
    ~GaussianRenderer() override;

    void initialize(GLFWwindow* window) override;
    void cleanup() override;
    void render() override;
    void renderToImage() override;
    void presentImage() override;
    void renderToBuffer() override;
    void onResize(uint32_t width, uint32_t height) override;

    void setPresentModePreference(PresentModePreference preference);
    PresentModePreference getPresentModePreference() const {
        return presentModePreference_;
    }
    void setRenderProfile(GaussianRenderProfile profile) {
        renderProfile_ = profile;
    }
    GaussianRenderProfile getRenderProfile() const { return renderProfile_; }
    void setSHBands(uint32_t bands) { shBands_ = std::min(bands, 3u); }
    uint32_t getSHBands() const { return shBands_; }

    void setRenderData(const GaussianModel* model,
                       const glm::mat4& view,
                       const glm::mat4& projection,
                       const Camera& camera,
                       const glm::mat4& modelMatrix);

private:
    void initializeImGui(GLFWwindow* window);
    void shutdownImGui();
    void beginImGuiFrame();
    void prepareFrameData();
    vk::RenderPass rebuildFormatDependentResources(vk::Format imageFormat,
                                                    vk::Extent2D extent);

    GraphicsFrameRuntime frameRuntime_;
    GraphicsPipelineSet pipelineSet_;
    GraphicsSplatSorter sorter_;
    GraphicsSplatResources resources_;
    GraphicsCommandRecorder commandRecorder_;
    std::vector<GraphicsUniformData> uniforms_;

    const GaussianModel* currentModel_ = nullptr;
    Camera camera_;
    GLFWwindow* window_ = nullptr;
    bool imguiInitialized_ = false;
    PresentModePreference presentModePreference_ = PresentModePreference::MaxFps;
    GaussianRenderProfile renderProfile_ = GaussianRenderProfile::Legacy;
    uint32_t shBands_ = 3;
};

} // namespace vulkan3DGS
