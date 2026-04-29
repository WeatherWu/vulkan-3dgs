#pragma once

#include <vulkan/vulkan.hpp>
#include <string>
#include <vector>

namespace vk_gs {

class Shader {
public:
    Shader();
    Shader(Shader&& other) noexcept;
    Shader& operator=(Shader&& other) noexcept;
    ~Shader();
    
    // 从SPIR-V文件创建着色器模块
    void createFromSpv(vk::Device device, const std::string& filepath, vk::ShaderStageFlagBits stage);
    
    // 清理资源
    void cleanup();
    
    vk::ShaderModule getShaderModule() const { return shaderModule_; }
    vk::ShaderStageFlagBits getStage() const { return stage_; }
    
    // 获取着色器阶段配置信息（用于管线创建）
    vk::PipelineShaderStageCreateInfo getStageCreateInfo() const;
    
private:
    vk::Device device_;
    vk::ShaderModule shaderModule_ = nullptr;
    vk::ShaderStageFlagBits stage_;
    std::string entryPoint_ = "main";
};

} // namespace vk_gs
