#include "graphics_splat_sorter.hpp"

#include "context/context.hpp"
#include "utils/logger.hpp"

#include <cmath>
#include <stdexcept>

namespace vulkan3DGS {

namespace {

bool matrixChanged(const glm::mat4& lhs, const glm::mat4& rhs, float epsilon) {
    for (int column = 0; column < 4; ++column) {
        for (int row = 0; row < 4; ++row) {
            if (std::abs(lhs[column][row] - rhs[column][row]) > epsilon) {
                return true;
            }
        }
    }
    return false;
}

vk::DescriptorSetLayoutBinding makeComputeBinding(uint32_t binding,
                                                   vk::DescriptorType type) {
    vk::DescriptorSetLayoutBinding layoutBinding{};
    layoutBinding.setBinding(binding)
        .setDescriptorType(type)
        .setDescriptorCount(1)
        .setStageFlags(vk::ShaderStageFlagBits::eCompute);
    return layoutBinding;
}

} // namespace

GraphicsSplatSorter::~GraphicsSplatSorter() {
    cleanup();
}

void GraphicsSplatSorter::initialize() {
    cleanup();

    auto& context = Context::Instance();
    const vk::Device device = context.getDevice().getDevice();

    ComputePipelineConfig keygenConfig{};
    keygenConfig.descriptorBindings = {
        makeComputeBinding(0, vk::DescriptorType::eStorageBuffer),
        makeComputeBinding(1, vk::DescriptorType::eStorageBuffer),
        makeComputeBinding(6, vk::DescriptorType::eStorageBuffer),
        makeComputeBinding(7, vk::DescriptorType::eUniformBuffer),
        makeComputeBinding(8, vk::DescriptorType::eStorageBuffer),
    };
    keygenConfig.pushConstantSize = sizeof(uint32_t) * 4;

    keygenPipeline_ = std::make_unique<ComputePipeline>();
    keygenPipeline_->initialize(device, "shaders/radix_keygen.comp.spv", keygenConfig);

    VrdxSorterCreateInfo sorterInfo{};
    sorterInfo.physicalDevice = context.PhysicalDevice();
    sorterInfo.device = device;
    sorterInfo.pipelineCache = VK_NULL_HANDLE;
    vrdxCreateSorter(&sorterInfo, &radixSorter_);
    invalidateSortCache();
}

void GraphicsSplatSorter::cleanup() {
    resetModelResources();
    keygenPipeline_.reset();
    if (radixSorter_) {
        vrdxDestroySorter(radixSorter_);
        radixSorter_ = VK_NULL_HANDLE;
    }
}

void GraphicsSplatSorter::resetModelResources() {
    indexBuffer_.cleanup();
    keyBuffer_.cleanup();
    storageBuffer_.cleanup();
    indirectBuffer_.cleanup();
    capacity_ = 0;
    invalidateSortCache();
}

void GraphicsSplatSorter::beginFrame() {
    recordSortThisFrame_ = false;
    pointCountThisFrame_ = 0;
}

bool GraphicsSplatSorter::prepare(uint32_t pointCount,
                                  const void* modelIdentity,
                                  const glm::mat4& view,
                                  const glm::mat4& projection,
                                  const glm::mat4& model,
                                  const std::vector<vk::Fence>& inFlightFences) {
    beginFrame();
    if (pointCount == 0u || modelIdentity == nullptr) {
        return false;
    }

    constexpr float matrixEpsilon = 1e-5f;
    const bool needsSort = !sortCompleted_ ||
                           modelIdentity != lastModelIdentity_ ||
                           pointCount != lastPointCount_ ||
                           matrixChanged(view, lastView_, matrixEpsilon) ||
                           matrixChanged(projection, lastProjection_, matrixEpsilon) ||
                           matrixChanged(model, lastModel_, matrixEpsilon);
    if (!needsSort) {
        return false;
    }

    const bool buffersChanged = ensureCapacity(pointCount, inFlightFences);
    recordSortThisFrame_ = true;
    pointCountThisFrame_ = pointCount;
    sortCompleted_ = true;
    lastPointCount_ = pointCount;
    lastModelIdentity_ = modelIdentity;
    lastView_ = view;
    lastProjection_ = projection;
    lastModel_ = model;
    return buffersChanged;
}

void GraphicsSplatSorter::record(vk::CommandBuffer commandBuffer,
                                 vk::DescriptorSet keygenDescriptorSet) const {
    if (!recordSortThisFrame_ || pointCountThisFrame_ == 0u) {
        return;
    }

    LOG_DEBUG("Sorting {} points", pointCountThisFrame_);
    const uint32_t groupCount = (pointCountThisFrame_ + 255u) / 256u;
    commandBuffer.fillBuffer(indirectBuffer_.getBuffer(), 0,
                             sizeof(VkDrawIndexedIndirectCommand), 0);

    vk::MemoryBarrier drawClearBarrier{};
    drawClearBarrier.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
        .setDstAccessMask(vk::AccessFlagBits::eShaderRead |
                          vk::AccessFlagBits::eShaderWrite);
    commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                  vk::PipelineStageFlagBits::eComputeShader,
                                  {}, 1, &drawClearBarrier, 0, nullptr, 0, nullptr);

    struct PushConstants {
        uint32_t count;
        uint32_t shift;
        uint32_t groupCount;
        uint32_t padding;
    } pushConstants{};
    pushConstants.count = pointCountThisFrame_;
    pushConstants.groupCount = groupCount;

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute,
                               keygenPipeline_->getPipeline());
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                     keygenPipeline_->getPipelineLayout(), 0, 1,
                                     &keygenDescriptorSet, 0, nullptr);
    commandBuffer.pushConstants(keygenPipeline_->getPipelineLayout(),
                                vk::ShaderStageFlagBits::eCompute, 0,
                                sizeof(PushConstants), &pushConstants);
    commandBuffer.dispatch(groupCount, 1, 1);

    vk::MemoryBarrier sortInputBarrier{};
    sortInputBarrier.setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
        .setDstAccessMask(vk::AccessFlagBits::eShaderRead |
                          vk::AccessFlagBits::eShaderWrite |
                          vk::AccessFlagBits::eTransferRead);
    commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                  vk::PipelineStageFlagBits::eComputeShader |
                                      vk::PipelineStageFlagBits::eTransfer,
                                  {}, 1, &sortInputBarrier, 0, nullptr, 0, nullptr);

    vrdxCmdSortKeyValueIndirect(commandBuffer, radixSorter_, pointCountThisFrame_,
                                indirectBuffer_.getBuffer(), sizeof(uint32_t),
                                keyBuffer_.getBuffer(), 0,
                                indexBuffer_.getBuffer(), 0,
                                storageBuffer_.getBuffer(), 0,
                                VK_NULL_HANDLE, 0);
}

void GraphicsSplatSorter::recordReadBarrier(vk::CommandBuffer commandBuffer) const {
    if (!indexBuffer_.getBuffer()) {
        return;
    }
    vk::MemoryBarrier barrier{};
    barrier.setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
        .setDstAccessMask(vk::AccessFlagBits::eShaderRead |
                          vk::AccessFlagBits::eIndirectCommandRead);
    commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                  vk::PipelineStageFlagBits::eVertexShader |
                                      vk::PipelineStageFlagBits::eDrawIndirect,
                                  {}, 1, &barrier, 0, nullptr, 0, nullptr);
}

vk::DescriptorSetLayout GraphicsSplatSorter::descriptorSetLayout() const {
    return keygenPipeline_ ? keygenPipeline_->getDescriptorSetLayout()
                           : vk::DescriptorSetLayout{};
}

bool GraphicsSplatSorter::ensureCapacity(
    uint32_t pointCount, const std::vector<vk::Fence>& inFlightFences) {
    if (capacity_ == pointCount) {
        return false;
    }

    auto& context = Context::Instance();
    const vk::Device device = context.getDevice().getDevice();
    if (indexBuffer_.getBuffer() && !inFlightFences.empty()) {
        const vk::Result result = device.waitForFences(inFlightFences, VK_TRUE, UINT64_MAX);
        if (result != vk::Result::eSuccess) {
            throw std::runtime_error("Failed to wait before resizing graphics sort buffers");
        }
    }

    indexBuffer_.cleanup();
    keyBuffer_.cleanup();
    storageBuffer_.cleanup();
    indirectBuffer_.cleanup();

    const vk::PhysicalDevice physicalDevice = context.PhysicalDevice();
    const vk::Queue queue = context.getDevice().getGraphicsQueue();
    const uint32_t queueFamily =
        context.getDevice().getQueueFamilyIndices().graphicsIndex.value();
    const vk::BufferUsageFlags keyValueUsage =
        vk::BufferUsageFlagBits::eStorageBuffer |
        vk::BufferUsageFlagBits::eTransferSrc |
        vk::BufferUsageFlagBits::eTransferDst;
    const vk::DeviceSize byteSize =
        static_cast<vk::DeviceSize>(pointCount) * sizeof(uint32_t);

    indexBuffer_.create(device, physicalDevice, queue, queueFamily, nullptr,
                        byteSize, keyValueUsage,
                        vk::MemoryPropertyFlagBits::eDeviceLocal);
    keyBuffer_.create(device, physicalDevice, queue, queueFamily, nullptr,
                      byteSize, keyValueUsage,
                      vk::MemoryPropertyFlagBits::eDeviceLocal);

    VrdxSorterStorageRequirements requirements{};
    vrdxGetSorterKeyValueStorageRequirements(radixSorter_, pointCount, &requirements);
    storageBuffer_.create(device, physicalDevice, queue, queueFamily, nullptr,
                          requirements.size, vk::BufferUsageFlags(requirements.usage),
                          vk::MemoryPropertyFlagBits::eDeviceLocal);
    indirectBuffer_.create(device, physicalDevice, queue, queueFamily, nullptr,
                           sizeof(VkDrawIndexedIndirectCommand),
                           vk::BufferUsageFlagBits::eStorageBuffer |
                               vk::BufferUsageFlagBits::eIndirectBuffer |
                               vk::BufferUsageFlagBits::eTransferSrc |
                               vk::BufferUsageFlagBits::eTransferDst,
                           vk::MemoryPropertyFlagBits::eDeviceLocal);
    capacity_ = pointCount;
    return true;
}

void GraphicsSplatSorter::invalidateSortCache() {
    sortCompleted_ = false;
    lastPointCount_ = 0;
    lastModelIdentity_ = nullptr;
    lastView_ = glm::mat4(1.0f);
    lastProjection_ = glm::mat4(1.0f);
    lastModel_ = glm::mat4(1.0f);
    beginFrame();
}

} // namespace vulkan3DGS
