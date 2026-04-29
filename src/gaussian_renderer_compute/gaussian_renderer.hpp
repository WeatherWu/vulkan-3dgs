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
    
    // 设置当前要渲染的模型和相机参数
    void setRenderData(const GaussianModel* model, const glm::mat4& view, const glm::mat4& projection);
    
private:
    void createBuffers();
    void createSyncObjects();
    void createComputePipeline();
    void createDescriptorSets();
    void updateDescriptorSets();
    void updateUniformBuffer(const glm::mat4& view, const glm::mat4& projection);

    void recreateSwapchain(uint32_t width, uint32_t height);
    
    // GPU 深度排序
    void sortGaussiansByDepthGPU();
    void computeDistances(const glm::vec3& cameraPosition);
    
    // 更新顶点缓冲区数据
    void updateVertexBuffer();
    
    // 记录命令缓冲区
    void recordCommandBuffer(uint32_t imageIndex);
    void recordComputeCommands(uint32_t stage, uint32_t substage);

    std::unique_ptr<Swapchain> swapchain_;
    std::unique_ptr<RenderPass> renderPass_;
    std::unique_ptr<Pipeline> pipeline_;
    std::unique_ptr<ComputePipeline> computePipeline_;
    
    // 高斯数据缓冲区
    Buffer instanceBuffer_;    // 实例数据缓冲区（每个高斯点一个实例）
    Buffer uniformBuffer_;     // 主Uniform Buffer (View/Projection/Camera)
    Buffer screenInfoBuffer_;  // 屏幕信息Uniform Buffer (分辨率)
    Buffer gpuIndexBuffer_;    // GPU排序索引缓冲
    Buffer gpuDistanceBuffer_; // GPU距离计算缓冲
    
    // Descriptor Set 相关
    vk::DescriptorPool descriptorPool_;
    std::vector<vk::DescriptorSet> descriptorSets_;          // Graphics Pipeline使用的Descriptor Sets
    std::vector<vk::DescriptorSet> computeDescriptorSets_;   // Compute Pipeline使用的Descriptor Sets
    
    // 同步对象（基于Swapchain图像索引）
    std::vector<vk::Semaphore> imageAvailableSemaphores_;  // 按imageIndex索引
    std::vector<vk::Semaphore> renderFinishedSemaphores_;  // 按imageIndex索引
    std::vector<vk::Fence> inFlightFences_;                // 按frameIndex索引
    
    // 命令池和命令缓冲区
    CommandPool commandPool_;
    CommandPool computeCommandPool_;
    std::vector<vk::CommandBuffer> commandBuffers_;        // 按frameIndex索引
    vk::CommandBuffer computeCommandBuffer_;
    
    // 当前帧索引
    size_t currentFrame_ = 0;
    uint32_t swapchainImageCount_ = 0;  // Swapchain图像数量（仅用于参考）
    
    // 同步对象配置（恢复为2以匹配Swapchain图像数）
    static constexpr int MAX_FRAMES_IN_FLIGHT = 2;
    
    // Uniform Buffer Object
    struct UniformBufferObject {
        alignas(16) glm::mat4 view;
        alignas(16) glm::mat4 projection;
        alignas(16) glm::vec3 cameraPosition;
        alignas(4) float time;
        alignas(8) glm::vec2 screenSize; // 屏幕分辨率
    } ubo_;
    
    // 当前渲染的高斯模型
    const GaussianModel* current_model_ = nullptr;
    
    // GPU排序缓存状态
    bool gpu_sort_completed_ = false;
    uint32_t last_sorted_point_count_ = 0;
    glm::vec3 last_camera_position_;
    
};

} // namespace vk_gs