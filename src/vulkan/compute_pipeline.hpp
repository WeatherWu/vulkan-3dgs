#pragma once

#include <vulkan/vulkan.hpp>
#include <vector>
#include <memory>
#include "vulkan/shader.hpp"

namespace vk_gs {

class ComputePipeline {
public:
    ComputePipeline();
    ~ComputePipeline();
    
    void initialize(vk::Device device, const std::string& shaderPath);
    void cleanup();
    
    vk::Pipeline getPipeline() const { return pipeline_; }
    vk::PipelineLayout getPipelineLayout() const { return pipelineLayout_; }
    vk::DescriptorSetLayout getDescriptorSetLayout() const { return descriptorSetLayout_; }
    
private:
    vk::Device device_;
    vk::Pipeline pipeline_ = nullptr;
    vk::PipelineLayout pipelineLayout_ = nullptr;
    vk::DescriptorSetLayout descriptorSetLayout_ = nullptr;
    
    std::unique_ptr<Shader> computeShader_;
};

} // namespace vk_gs
