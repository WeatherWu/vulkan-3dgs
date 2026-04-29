#include "compute_pipeline.hpp"
#include "utils/logger.hpp"

#include <stdexcept>
#include <fstream>

namespace vk_gs {

ComputePipeline::ComputePipeline() = default;

ComputePipeline::~ComputePipeline() {
    cleanup();
}

void ComputePipeline::initialize(vk::Device device, const std::string& shaderPath) {
    device_ = device;
    
    LOG_INFO("Creating compute pipeline: {}", shaderPath);
    
    // 加载 Compute Shader
    computeShader_ = std::make_unique<Shader>();
    computeShader_->createFromSpv(device_, shaderPath, vk::ShaderStageFlagBits::eCompute);
    
    // 创建 Descriptor Set Layout（支持两个 Storage Buffer）
    std::array<vk::DescriptorSetLayoutBinding, 2> storageBufferBindings{};
    
    // Binding 0: IndexBuffer
    storageBufferBindings[0].setBinding(0)
                            .setDescriptorType(vk::DescriptorType::eStorageBuffer)
                            .setDescriptorCount(1)
                            .setStageFlags(vk::ShaderStageFlagBits::eCompute);
    
    // Binding 1: DistanceBuffer
    storageBufferBindings[1].setBinding(1)
                            .setDescriptorType(vk::DescriptorType::eStorageBuffer)
                            .setDescriptorCount(1)
                            .setStageFlags(vk::ShaderStageFlagBits::eCompute);
    
    vk::DescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.setBindingCount(static_cast<uint32_t>(storageBufferBindings.size()))
              .setPBindings(storageBufferBindings.data());
    
    descriptorSetLayout_ = device_.createDescriptorSetLayout(layoutInfo);
    
    // 创建 Push Constant Range
    vk::PushConstantRange pushConstantRange{};
    pushConstantRange.setStageFlags(vk::ShaderStageFlagBits::eCompute)
                     .setOffset(0)
                     .setSize(sizeof(uint32_t) * 3); // count, stage, substage
    
    // 创建 Pipeline Layout
    vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.setSetLayoutCount(1)
                      .setPSetLayouts(&descriptorSetLayout_)
                      .setPushConstantRangeCount(1)
                      .setPPushConstantRanges(&pushConstantRange);
    
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
    
    LOG_INFO("Compute pipeline created successfully");
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
    
    LOG_INFO("Compute pipeline cleaned up");
}

} // namespace vk_gs
