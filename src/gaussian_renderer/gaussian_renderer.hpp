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

namespace vk_gs {

// 高斯渲染器，继承自通用渲染器基类
class GaussianRenderer : public Renderer {
public:
    GaussianRenderer();
    ~GaussianRenderer() override;
    
    void initialize(GLFWwindow* window) override;
    void cleanup() override;
    void render() override;
    void onResize(uint32_t width, uint32_t height) override;
    void setPresentModePreference(PresentModePreference preference);
    PresentModePreference getPresentModePreference() const { return presentModePreference_; }
    
    // 设置当前要渲染的模型和相机参数
    void setRenderData(const GaussianModel* model, const glm::mat4& view, const glm::mat4& projection, const vk_gs::Camera& camera, const glm::mat4& modelMatrix);
    
private:
    void initializeImGui(GLFWwindow* window);
    void shutdownImGui();
    void beginImGuiFrame();

    void createBuffers();
    void createSyncObjects();
    void recreateRenderFinishedSemaphores();
    void createComputePipeline();
    void createDescriptorSets();
    void destroyDescriptorPool();
    void updateDescriptorSets();
    void updateUniformBuffer(const glm::mat4& view, const glm::mat4& projection);

    void recreateSwapchain(uint32_t width, uint32_t height);
    
    // GPU 深度排序
    void recordSortCommands(vk::CommandBuffer commandBuffer, uint32_t pointCount);
    void ensureSortBuffers(uint32_t pointCount);
    
    // 更新顶点缓冲区数据
    void updateVertexBuffer();
    
    // 记录命令缓冲区
    void recordCommandBuffer(uint32_t imageIndex);

    std::unique_ptr<Swapchain> swapchain_;
    std::unique_ptr<RenderPass> renderPass_;
    std::unique_ptr<Pipeline> pipeline_;
    std::unique_ptr<ComputePipeline> radixKeygenPipeline_;
    VrdxSorter radixSorter_ = VK_NULL_HANDLE;
    
    // 高斯数据缓冲区
    Buffer instanceBuffer_;    // 实例数据缓冲区（SSBO，存储所有高斯属性）
    Buffer uniformBuffer_;     // 主Uniform Buffer (View/Projection/Camera)
    Buffer screenInfoBuffer_;  // 屏幕信息Uniform Buffer (分辨率)
    Buffer gpuIndexBuffer_;    // GPU排序索引缓冲
    Buffer gpuKeyBuffer_;      // GPU排序key缓冲
    Buffer radixSortStorageBuffer_;
    Buffer drawIndirectBuffer_;
    uint32_t sortBufferCapacity_ = 0;
    
    // Descriptor Set 相关
    vk::DescriptorPool descriptorPool_;
    std::vector<vk::DescriptorSet> descriptorSets_;          // Graphics Pipeline使用的Descriptor Sets
    std::vector<vk::DescriptorSet> computeDescriptorSets_;   // Compute Pipeline使用的Descriptor Sets
    
    // 同步对象（基于Swapchain图像索引）
    std::vector<vk::Semaphore> imageAvailableSemaphores_;  // 按frameIndex索引
    std::vector<vk::Semaphore> renderFinishedSemaphores_;  // 按imageIndex索引
    std::vector<vk::Fence> inFlightFences_;                // 按frameIndex索引
    
    // 命令池和命令缓冲区
    CommandPool commandPool_;
    std::vector<vk::CommandBuffer> commandBuffers_;        // 按frameIndex索引
    
    // 当前帧索引
    size_t currentFrame_ = 0;
    uint32_t swapchainImageCount_ = 0;
    uint32_t frameResourceCount_ = 0;
    
    // Uniform Buffer Object (必须与GLSL std140布局严格对齐)
    struct UniformBufferObject {
        alignas(16) glm::mat4 view;              // offset 0, size 64
        alignas(16) glm::mat4 projection;        // offset 64, size 64
        alignas(16) glm::mat4 model;             // offset 128, size 64
        alignas(16) glm::vec4 cameraPositionTime;// xyz: camera position, w: time
        alignas(16) glm::vec4 focal;             // xy: pixel focal lengths, zw: screen size
    };
    std::vector<UniformBufferObject> ubo_;
    
    // 当前渲染的高斯模型
    const GaussianModel* current_model_ = nullptr;

    vk_gs::Camera camera_;

    bool imguiInitialized_ = false;
    PresentModePreference presentModePreference_ = PresentModePreference::MaxFps;
    
    // GPU排序缓存状态
    bool gpu_sort_completed_ = false;
    uint32_t last_sorted_point_count_ = 0;
    const GaussianModel* last_sorted_model_ = nullptr;
    glm::vec3 last_camera_position_;
    glm::mat4 last_view_matrix_{1.0f};
    glm::mat4 last_projection_matrix_{1.0f};
    glm::mat4 last_model_matrix_{1.0f};
    bool record_sort_this_frame_ = false;
    uint32_t sort_point_count_this_frame_ = 0;
    
};

} // namespace vk_gs
