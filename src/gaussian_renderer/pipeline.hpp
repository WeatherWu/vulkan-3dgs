#pragma once

#include <vulkan/vulkan.hpp>
#include <vector>
#include "vulkan/shader.hpp"

namespace vk_gs {

class Pipeline {
public:
    Pipeline();
    ~Pipeline();
    
    void initialize(vk::Device device, vk::RenderPass render_pass, vk::Extent2D extent);
    void cleanup();
    
    vk::Pipeline getPipeline() const { return pipeline_; }
    vk::PipelineLayout getPipelineLayout() const { return pipelineLayout_; }
    vk::DescriptorSetLayout getDescriptorSetLayout() const { return descriptorSetLayout_; }
    
    vk::Buffer getQuadVertexBuffer() const { return quadVertexBuffer_; }
    vk::Buffer getQuadIndexBuffer() const { return quadIndexBuffer_; }

private:
    vk::Device device_;
    vk::Pipeline pipeline_ = nullptr;
    vk::PipelineLayout pipelineLayout_ = nullptr;
    vk::DescriptorSetLayout descriptorSetLayout_ = nullptr;
    
    // 预定义的四边形顶点缓冲区和索引缓冲区
    vk::Buffer quadVertexBuffer_ = nullptr;
    vk::DeviceMemory quadVertexBufferMemory_ = nullptr;
    vk::Buffer quadIndexBuffer_ = nullptr;
    vk::DeviceMemory quadIndexBufferMemory_ = nullptr;
    
    std::vector<Shader> shaders_;
};

} // namespace vk_gs
