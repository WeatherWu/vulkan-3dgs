#include "shader.hpp"
#include "utils/file_utils.hpp"
#include "utils/logger.hpp"

#include <stdexcept>

namespace vulkan3DGS {

Shader::Shader() = default;

Shader::Shader(Shader&& other) noexcept 
    : device_(other.device_)
    , shaderModule_(other.shaderModule_)
    , stage_(other.stage_)
    , entryPoint_(std::move(other.entryPoint_)) {
    // 将源对象的shaderModule_置为nullptr，避免重复销毁
    other.shaderModule_ = nullptr;
}

Shader& Shader::operator=(Shader&& other) noexcept {
    if (this != &other) {
        // 清理当前资源
        cleanup();
        
        // 移动资源
        device_ = other.device_;
        shaderModule_ = other.shaderModule_;
        stage_ = other.stage_;
        entryPoint_ = std::move(other.entryPoint_);
        
        // 将源对象的shaderModule_置为nullptr
        other.shaderModule_ = nullptr;
    }
    return *this;
}

Shader::~Shader() {
    cleanup();
}

void Shader::createFromSpv(vk::Device device, const std::string& filepath, vk::ShaderStageFlagBits stage) {
    device_ = device;
    stage_ = stage;
    
    LOG_DEBUG("Loading shader module: {}", filepath);
    
    // 加载SPIR-V文件
    auto shaderCode = FileUtils::readBinaryFile(filepath);
    
    // 创建着色器模块
    vk::ShaderModuleCreateInfo createInfo{};
    createInfo.setCodeSize(shaderCode.size());
    createInfo.setPCode(reinterpret_cast<const uint32_t*>(shaderCode.data()));
    
    shaderModule_ = device_.createShaderModule(createInfo);
    
    if (!shaderModule_) {
        LOG_ERROR("Failed to create shader module: {}", filepath);
        throw std::runtime_error("Failed to create shader module: " + filepath);
    }
    
    LOG_DEBUG("Shader module created successfully: {}", filepath);
}

void Shader::cleanup() {
    if (shaderModule_) {
        device_.destroyShaderModule(shaderModule_);
        shaderModule_ = nullptr;
    }
}

vk::PipelineShaderStageCreateInfo Shader::getStageCreateInfo() const {
    vk::PipelineShaderStageCreateInfo stageInfo{};
    stageInfo.setStage(stage_)
             .setModule(shaderModule_)
             .setPName(entryPoint_.c_str());
    
    return stageInfo;
}

} // namespace vulkan3DGS
