#pragma once

#include <vulkan/vulkan.hpp>
#include <vector>
#include <memory>
#include <string>
#include "vulkan/shader.hpp"

namespace vulkan3DGS {

struct ComputePipelineConfig {
    std::vector<vk::DescriptorSetLayoutBinding> descriptorBindings;
    uint32_t pushConstantSize = 0;
    vk::ShaderStageFlags pushConstantStages = vk::ShaderStageFlagBits::eCompute;
};

class ComputePipeline {
public:
    ComputePipeline();
    ~ComputePipeline();
    
    void initialize(vk::Device device, const std::string& shaderPath, const ComputePipelineConfig& config);
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

} // namespace vulkan3DGS
