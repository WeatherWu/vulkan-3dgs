#include "pipeline.hpp"
#include "context/context.hpp"
#include "utils/logger.hpp"

#include <stdexcept>
#include <cstring>

namespace vk_gs {

Pipeline::Pipeline() = default;

Pipeline::~Pipeline() {
    cleanup();
}

void Pipeline::initialize(vk::Device device, vk::RenderPass render_pass, vk::Extent2D extent) {
    device_ = device;
    
    // 创建四边形顶点缓冲区（4个顶点：-1,-1 / 1,-1 / -1,1 / 1,1）
    std::array<float, 8> quadVertices = {
        -1.0f, -1.0f,  // 左下
         1.0f, -1.0f,  // 右下
        -1.0f,  1.0f,  // 左上
         1.0f,  1.0f   // 右上
    };
    
    vk::BufferCreateInfo vertexBufferInfo{};
    vertexBufferInfo.setSize(sizeof(quadVertices))
                      .setUsage(vk::BufferUsageFlagBits::eVertexBuffer);
    quadVertexBuffer_ = device_.createBuffer(vertexBufferInfo);
    
    vk::MemoryRequirements vertexMemReqs = device_.getBufferMemoryRequirements(quadVertexBuffer_);
    
    // 查找合适的内存类型（需要主机可见且可写）
    uint32_t vertexMemoryTypeIndex = 0;
    vk::PhysicalDevice physicalDevice = Context::Instance().getDevice().getPhysicalDevice();
    vk::PhysicalDeviceMemoryProperties memProps = physicalDevice.getMemoryProperties();
    
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if ((vertexMemReqs.memoryTypeBits & (1 << i)) &&
            (memProps.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eHostVisible) &&
            (memProps.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eHostCoherent)) {
            vertexMemoryTypeIndex = i;
            break;
        }
    }
    
    vk::MemoryAllocateInfo vertexAllocInfo{};
    vertexAllocInfo.setAllocationSize(vertexMemReqs.size)
                   .setMemoryTypeIndex(vertexMemoryTypeIndex);
    quadVertexBufferMemory_ = device_.allocateMemory(vertexAllocInfo);
    device_.bindBufferMemory(quadVertexBuffer_, quadVertexBufferMemory_, 0);
    
    void* vertexData = device_.mapMemory(quadVertexBufferMemory_, 0, sizeof(quadVertices));
    std::memcpy(vertexData, quadVertices.data(), sizeof(quadVertices));
    device_.unmapMemory(quadVertexBufferMemory_);
    // 创建索引缓冲区（三角形带顺序：0, 1, 2, 3）
    std::array<uint16_t, 4> quadIndices = {0, 1, 2, 3};
    
    vk::BufferCreateInfo indexBufferInfo{};
    indexBufferInfo.setSize(sizeof(quadIndices))
                   .setUsage(vk::BufferUsageFlagBits::eIndexBuffer);
    quadIndexBuffer_ = device_.createBuffer(indexBufferInfo);
    
    vk::MemoryRequirements indexMemReqs = device_.getBufferMemoryRequirements(quadIndexBuffer_);
    
    // 查找合适的内存类型
    uint32_t indexMemoryTypeIndex = 0;
    
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if ((indexMemReqs.memoryTypeBits & (1 << i)) &&
            (memProps.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eHostVisible) &&
            (memProps.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eHostCoherent)) {
            indexMemoryTypeIndex = i;
            break;
        }
    }
    
    vk::MemoryAllocateInfo indexAllocInfo{};
    indexAllocInfo.setAllocationSize(indexMemReqs.size)
                  .setMemoryTypeIndex(indexMemoryTypeIndex);
    quadIndexBufferMemory_ = device_.allocateMemory(indexAllocInfo);
    device_.bindBufferMemory(quadIndexBuffer_, quadIndexBufferMemory_, 0);
    
    void* indexData = device_.mapMemory(quadIndexBufferMemory_, 0, sizeof(quadIndices));
    std::memcpy(indexData, quadIndices.data(), sizeof(quadIndices));
    device_.unmapMemory(quadIndexBufferMemory_);
    // 创建顶点着色器
    Shader vertShader;
    vertShader.createFromSpv(device_, "shaders/gaussian_compute_shader.vert.spv", vk::ShaderStageFlagBits::eVertex);
    shaders_.push_back(std::move(vertShader));
    
    // 创建片段着色器
    Shader fragShader;
    fragShader.createFromSpv(device_, "shaders/gaussian_compute_shader.frag.spv", vk::ShaderStageFlagBits::eFragment);
    shaders_.push_back(std::move(fragShader));
    
    // 配置着色器阶段
    std::vector<vk::PipelineShaderStageCreateInfo> shader_stages;
    for (const auto& shader : shaders_) {
        shader_stages.push_back(shader.getStageCreateInfo());
    }
    
    // === 顶点绑定描述 ===
    // Binding 0: 四边形顶点数据（4个顶点，每个实例共享）
    vk::VertexInputBindingDescription vertexBinding{};
    vertexBinding.setBinding(0)
                 .setStride(sizeof(float) * 2)  // vec2 quadVertex
                 .setInputRate(vk::VertexInputRate::eVertex);
    
    std::array<vk::VertexInputBindingDescription, 1> bindings = {vertexBinding};
    
    // === 顶点属性描述 ===
    // 只保留四边形顶点的属性，移除所有实例数据属性
    std::array<vk::VertexInputAttributeDescription, 1> attributes{};
    
    // Binding 0 的属性（四边形顶点）
    attributes[0].setBinding(0)
                 .setLocation(0)
                 .setFormat(vk::Format::eR32G32Sfloat)  // vec2 quadVertex
                 .setOffset(0);
    
    // 顶点输入状态
    vk::PipelineVertexInputStateCreateInfo vertex_input{};
    vertex_input.setVertexBindingDescriptionCount(static_cast<uint32_t>(bindings.size()))
                .setPVertexBindingDescriptions(bindings.data())
                .setVertexAttributeDescriptionCount(static_cast<uint32_t>(attributes.size()))
                .setPVertexAttributeDescriptions(attributes.data());
    
    // 输入装配状态（使用三角形带）
    vk::PipelineInputAssemblyStateCreateInfo input_assembly{};
    input_assembly.setTopology(vk::PrimitiveTopology::eTriangleStrip)
                  .setPrimitiveRestartEnable(false);
    
    // 视口和剪刀状态（设为动态，但Pipeline创建时仍需提供）
    vk::Viewport viewport{};
    viewport.setX(0.0f)
            .setY(0.0f)
            .setWidth(static_cast<float>(extent.width))
            .setHeight(static_cast<float>(extent.height))
            .setMinDepth(0.0f)
            .setMaxDepth(1.0f);
    
    vk::Rect2D scissor{};
    scissor.setOffset({0, 0})
           .setExtent(extent);
    
    vk::PipelineViewportStateCreateInfo viewport_state{};
    viewport_state.setViewportCount(1)
                  .setPViewports(&viewport)
                  .setScissorCount(1)
                  .setPScissors(&scissor);
    
    // 光栅化状态
    vk::PipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.setDepthClampEnable(false)
              .setRasterizerDiscardEnable(false)
              .setPolygonMode(vk::PolygonMode::eFill)
              .setLineWidth(1.0f)
              .setCullMode(vk::CullModeFlagBits::eNone) // 禁用背面剔除
              .setFrontFace(vk::FrontFace::eCounterClockwise)
              .setDepthBiasEnable(false);
    
    // 多重采样状态
    vk::PipelineMultisampleStateCreateInfo multisampling{};
    multisampling.setSampleShadingEnable(false)
                 .setRasterizationSamples(vk::SampleCountFlagBits::e1);
    
    // Match vkgs: splats depth-test against the pass depth attachment but do
    // not write depth, so blending still follows the sorted back-to-front order.
    vk::PipelineDepthStencilStateCreateInfo depth_stencil{};
    depth_stencil.setDepthTestEnable(vk::True)
                 .setDepthWriteEnable(vk::False)
                 .setDepthCompareOp(vk::CompareOp::eLess);
    
    // 颜色混合状态（vkgs-style non-premultiplied alpha blending）
    vk::PipelineColorBlendAttachmentState color_blend_attachment{};
    color_blend_attachment.setColorWriteMask(
        vk::ColorComponentFlagBits::eR |
        vk::ColorComponentFlagBits::eG |
        vk::ColorComponentFlagBits::eB |
        vk::ColorComponentFlagBits::eA
    );
    color_blend_attachment.setBlendEnable(vk::True);
    color_blend_attachment.setSrcColorBlendFactor(vk::BlendFactor::eSrcAlpha);
    color_blend_attachment.setDstColorBlendFactor(vk::BlendFactor::eOneMinusSrcAlpha);
    color_blend_attachment.setColorBlendOp(vk::BlendOp::eAdd);
    color_blend_attachment.setSrcAlphaBlendFactor(vk::BlendFactor::eSrcAlpha);
    color_blend_attachment.setDstAlphaBlendFactor(vk::BlendFactor::eOneMinusSrcAlpha);
    color_blend_attachment.setAlphaBlendOp(vk::BlendOp::eAdd);
    
    vk::PipelineColorBlendStateCreateInfo color_blending{};
    color_blending.setLogicOpEnable(false)
                  .setAttachmentCount(1)
                  .setPAttachments(&color_blend_attachment);
    
    // 动态状态
    std::vector<vk::DynamicState> dynamic_states = {
        vk::DynamicState::eViewport,
        vk::DynamicState::eScissor
    };
    
    vk::PipelineDynamicStateCreateInfo dynamic_state{};
    dynamic_state.setDynamicStates(dynamic_states);
    
    // 管线布局（添加 Uniform Buffer Descriptor Set）
    std::array<vk::DescriptorSetLayoutBinding, 4> descriptorBindings{};
    
    // Binding 0: 主Uniform Buffer (View/Projection等)
    descriptorBindings[0].setBinding(0)
               .setDescriptorType(vk::DescriptorType::eUniformBuffer)
               .setDescriptorCount(1)
               .setStageFlags(vk::ShaderStageFlagBits::eVertex);
    
    // Binding 1: 屏幕信息Uniform Buffer
    descriptorBindings[1].setBinding(1)
               .setDescriptorType(vk::DescriptorType::eUniformBuffer)
               .setDescriptorCount(1)
               .setStageFlags(vk::ShaderStageFlagBits::eFragment);
    
    // Binding 2: 高斯实例数据 SSBO
    descriptorBindings[2].setBinding(2)
               .setDescriptorType(vk::DescriptorType::eStorageBuffer)
               .setDescriptorCount(1)
               .setStageFlags(vk::ShaderStageFlagBits::eVertex);

    // Binding 3: GPU排序后的实例索引 SSBO
    descriptorBindings[3].setBinding(3)
               .setDescriptorType(vk::DescriptorType::eStorageBuffer)
               .setDescriptorCount(1)
               .setStageFlags(vk::ShaderStageFlagBits::eVertex);
    
    vk::DescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.setBindingCount(static_cast<uint32_t>(descriptorBindings.size()))
              .setPBindings(descriptorBindings.data());
    
    vk::DescriptorSetLayout descriptorSetLayout = device_.createDescriptorSetLayout(layoutInfo);
    
    vk::PipelineLayoutCreateInfo pipeline_layout_info{};
    pipeline_layout_info.setSetLayoutCount(1)
                        .setPSetLayouts(&descriptorSetLayout)
                        .setPushConstantRangeCount(0);
    
    pipelineLayout_ = device_.createPipelineLayout(pipeline_layout_info);
    
    // 创建图形管线
    vk::GraphicsPipelineCreateInfo pipeline_info{};
    pipeline_info.setStageCount(static_cast<uint32_t>(shader_stages.size()))
                 .setPStages(shader_stages.data())
                 .setPVertexInputState(&vertex_input)
                 .setPInputAssemblyState(&input_assembly)
                 .setPViewportState(&viewport_state)
                 .setPRasterizationState(&rasterizer)
                 .setPMultisampleState(&multisampling)
                 .setPDepthStencilState(&depth_stencil)
                 .setPColorBlendState(&color_blending)
                 .setPDynamicState(&dynamic_state)
                 .setLayout(pipelineLayout_)
                 .setRenderPass(render_pass)
                 .setSubpass(0);
    
    try {
        pipeline_ = device_.createGraphicsPipeline(nullptr, pipeline_info).value;
        LOG_DEBUG("Graphics pipeline created successfully");
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to create graphics pipeline: {}", e.what());
        throw;
    }
    
    // 保存 Descriptor Set Layout 供后续使用
    descriptorSetLayout_ = descriptorSetLayout;
    
    LOG_INFO("Graphics pipeline initialized");
}

void Pipeline::cleanup() {
    // 清理着色器模块（Shader析构函数会自动调用cleanup）
    shaders_.clear();
    
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

    // 清理四边形缓冲区
    if (quadVertexBuffer_) {
        device_.destroyBuffer(quadVertexBuffer_);
        quadVertexBuffer_ = nullptr;
    }
    if (quadVertexBufferMemory_) {
        device_.freeMemory(quadVertexBufferMemory_);
        quadVertexBufferMemory_ = nullptr;
    }
    if (quadIndexBuffer_) {
        device_.destroyBuffer(quadIndexBuffer_);
        quadIndexBuffer_ = nullptr;
    }
    if (quadIndexBufferMemory_) {
        device_.freeMemory(quadIndexBufferMemory_);
        quadIndexBufferMemory_ = nullptr;
    }
}

} // namespace vk_gs
