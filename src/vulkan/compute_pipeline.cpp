#include "compute_pipeline.hpp"
#include "utils/logger.hpp"

#include <stdexcept>

namespace vulkan3DGS {

ComputePipeline::ComputePipeline() = default;

ComputePipeline::~ComputePipeline() {
    cleanup();
}

void ComputePipeline::initialize(vk::Device device, const std::string& shaderPath, const ComputePipelineConfig& config) {
    cleanup();

    device_ = device;

    LOG_DEBUG("Creating compute pipeline: {}", shaderPath);

    // 加载 Compute Shader
    computeShader_ = std::make_unique<Shader>();
    computeShader_->createFromSpv(device_, shaderPath, vk::ShaderStageFlagBits::eCompute);

    vk::DescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.setBindingCount(static_cast<uint32_t>(config.descriptorBindings.size()))
              .setPBindings(config.descriptorBindings.empty() ? nullptr : config.descriptorBindings.data());
    
    descriptorSetLayout_ = device_.createDescriptorSetLayout(layoutInfo);
    
    vk::PushConstantRange pushConstantRange{};
    if (config.pushConstantSize > 0) {
        pushConstantRange.setStageFlags(config.pushConstantStages)
                         .setOffset(0)
                         .setSize(config.pushConstantSize);
    }
    
    // 创建 Pipeline Layout
    vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.setSetLayoutCount(1)
                      .setPSetLayouts(&descriptorSetLayout_)
                      .setPushConstantRangeCount(config.pushConstantSize > 0 ? 1u : 0u)
                      .setPPushConstantRanges(config.pushConstantSize > 0 ? &pushConstantRange : nullptr);
    
    pipelineLayout_ = device_.createPipelineLayout(pipelineLayoutInfo);
    
    // 创建 Compute Pipeline
    vk::ComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.setLayout(pipelineLayout_)
                  .setStage(computeShader_->getStageCreateInfo());
    
    auto result = device_.createComputePipeline(nullptr, pipelineInfo);
    if (result.result != vk::Result::eSuccess) {
        LOG_ERROR("Failed to create compute pipeline");
        throw std::runtime_error("Failed to create compute pipeline");
    }
    
    pipeline_ = result.value;
    
    LOG_DEBUG("Compute pipeline created successfully");
}

void ComputePipeline::cleanup() {
    if (pipeline_) {
        device_.destroyPipeline(pipeline_);
        pipeline_ = nullptr;
    }
    
    if (pipelineLayout_) {
        device_.destroyPipelineLayout(pipelineLayout_);
        pipelineLayout_ = nullptr;
    }
    
    if (descriptorSetLayout_) {
        device_.destroyDescriptorSetLayout(descriptorSetLayout_);
        descriptorSetLayout_ = nullptr;
    }
    
    computeShader_.reset();
}

} // namespace vulkan3DGS
