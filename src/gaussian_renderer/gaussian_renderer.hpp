#pragma once

#include <memory>
#include <vector>
#include <glm/glm.hpp>
#include "renderer.hpp"
#include "vulkan/swapchain.hpp"
#include "renderpass.hpp"
#include "pipeline.hpp"
#include "gaussian_model.hpp"
#include "vulkan/buffer.hpp"
#include "vulkan/command_pool.hpp"
#include "vulkan/compute_pipeline.hpp"
#include "utils/camera.hpp"
#include <vk_radix_sort.h>

namespace vulkan3DGS {

// 高斯渲染器，继承自通用渲染器基类
class GaussianRenderer : public Renderer {
public:
    GaussianRenderer();
    ~GaussianRenderer() override;

    // ---- Renderer 接口 ----
    void initialize(GLFWwindow* window) override;
    void cleanup() override;

    void render() override;
    void renderToImage() override;
    void presentImage() override;
    void renderToBuffer() override;

    void onResize(uint32_t width, uint32_t height) override;

    // ---- 原有公开接口 ----
    void setPresentModePreference(PresentModePreference preference);
    PresentModePreference getPresentModePreference() const { return presentModePreference_; }

    void setRenderData(const GaussianModel* model, const glm::mat4& view, const glm::mat4& projection, const vulkan3DGS::Camera& camera, const glm::mat4& modelMatrix);

private:
    // ---- ImGui ----
    void initializeImGui(GLFWwindow* window);
    void shutdownImGui();
    void beginImGuiFrame();

    // ---- 资源创建 ----
    void createBuffers();
    void createSyncObjects();
    void recreateRenderFinishedSemaphores();
    void createComputePipeline();
    void createDescriptorSets();
    void destroyDescriptorPool();
    void updateDescriptorSets();
    void updateUniformBuffer(const glm::mat4& view, const glm::mat4& projection, vk::Extent2D extent);
    void prepareFrameData(vk::Extent2D extent);

    // ---- Swapchain ----
    void recreateSwapchain(uint32_t width, uint32_t height);

    // ---- GPU 深度排序 ----
    void recordSortCommands(vk::CommandBuffer commandBuffer, uint32_t pointCount);
    void ensureSortBuffers(uint32_t pointCount);

    // ---- 数据上传 ----
    void updateVertexBuffer();
    void recordCommandBuffer(uint32_t imageIndex);
    void recordRenderCommands(vk::CommandBuffer commandBuffer,
                              vk::RenderPass renderPass,
                              vk::Framebuffer framebuffer,
                              vk::Extent2D extent,
                              Pipeline& pipeline,
                              bool drawImGui);

    // ---- Swapchain ----
    std::unique_ptr<Swapchain> swapchain_;
    std::unique_ptr<RenderPass> renderPass_;
    std::unique_ptr<Pipeline> pipeline_;
    std::unique_ptr<ComputePipeline> radixKeygenPipeline_;
    VrdxSorter radixSorter_ = VK_NULL_HANDLE;

    // ---- 高斯数据缓冲区 ----
    Buffer instanceBuffer_;
    Buffer uniformBuffer_;
    Buffer screenInfoBuffer_;
    Buffer gpuIndexBuffer_;
    Buffer gpuKeyBuffer_;
    Buffer radixSortStorageBuffer_;
    Buffer drawIndirectBuffer_;
    uint32_t sortBufferCapacity_ = 0;

    // ---- Descriptor ----
    vk::DescriptorPool descriptorPool_;
    std::vector<vk::DescriptorSet> descriptorSets_;
    std::vector<vk::DescriptorSet> computeDescriptorSets_;

    // ---- 同步对象 ----
    std::vector<vk::Semaphore> imageAvailableSemaphores_;
    std::vector<vk::Semaphore> renderFinishedSemaphores_;
    std::vector<vk::Fence> inFlightFences_;

    // ---- 命令 ----
    CommandPool commandPool_;
    std::vector<vk::CommandBuffer> commandBuffers_;

    size_t currentFrame_ = 0;
    uint32_t swapchainImageCount_ = 0;
    uint32_t frameResourceCount_ = 0;

    // ---- UBO ----
    struct UniformBufferObject {
        alignas(16) glm::mat4 view;
        alignas(16) glm::mat4 projection;
        alignas(16) glm::mat4 model;
        alignas(16) glm::vec4 cameraPositionTime;
        alignas(16) glm::vec4 focal;
    };
    std::vector<UniformBufferObject> ubo_;

    // ---- 模型 & 相机 ----
    const GaussianModel* current_model_ = nullptr;
    vulkan3DGS::Camera camera_;

    bool imguiInitialized_ = false;
    PresentModePreference presentModePreference_ = PresentModePreference::MaxFps;

    // ---- GPU 排序缓存 ----
    bool gpu_sort_completed_ = false;
    uint32_t last_sorted_point_count_ = 0;
    const GaussianModel* last_sorted_model_ = nullptr;
    glm::vec3 last_camera_position_;
    glm::mat4 last_view_matrix_{1.0f};
    glm::mat4 last_projection_matrix_{1.0f};
    glm::mat4 last_model_matrix_{1.0f};
    bool record_sort_this_frame_ = false;
    uint32_t sort_point_count_this_frame_ = 0;
    uint32_t acquiredImageIndex_ = 0;
    bool imageReadyForPresent_ = false;
    bool presentWaitSemaphoreConsumed_ = false;

};

} // namespace vulkan3DGS
