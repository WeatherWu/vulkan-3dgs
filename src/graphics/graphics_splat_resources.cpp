#include "graphics_splat_resources.hpp"

#include "context/context.hpp"
#include "gaussian_model.hpp"
#include "graphics_splat_sorter.hpp"
#include "utils/logger.hpp"

#include <glm/gtc/packing.hpp>

#include <array>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace vulkan3DGS {

namespace {

uint32_t graphicsQueueFamilyIndex() {
    const auto index = Context::Instance().getDevice().getQueueFamilyIndices().graphicsIndex;
    if (!index) throw std::runtime_error("Graphics queue family is unavailable");
    return *index;
}

} // namespace

GraphicsSplatResources::~GraphicsSplatResources() {
    cleanup();
}

void GraphicsSplatResources::initialize(uint32_t frameCount, vk::DescriptorSetLayout graphicsLayout,
                                        vk::DescriptorSetLayout computeLayout) {
    cleanup();
    createBuffers();
    createDescriptorSets(frameCount, graphicsLayout, computeLayout);
}

void GraphicsSplatResources::cleanup() {
    destroyDescriptorPool();
    resetModel();
    uniformBuffer_.cleanup();
    screenInfoBuffer_.cleanup();
}

void GraphicsSplatResources::resetModel() {
    instanceBuffer_.cleanup();
}

void GraphicsSplatResources::releaseDescriptors() {
    destroyDescriptorPool();
}

void GraphicsSplatResources::rebuildDescriptors(uint32_t frameCount,
                                                vk::DescriptorSetLayout graphicsLayout,
                                                vk::DescriptorSetLayout computeLayout,
                                                const GraphicsSplatSorter& sorter) {
    destroyDescriptorPool();
    createDescriptorSets(frameCount, graphicsLayout, computeLayout);
    updateDescriptors(sorter);
}

void GraphicsSplatResources::ensureModelUploaded(const GaussianModel& model,
                                                 const GraphicsSplatSorter& sorter) {
    if (instanceBuffer_.getBuffer() || model.isEmpty()) {
        return;
    }
    uploadModel(model);
    updateDescriptors(sorter);
}

void GraphicsSplatResources::updateDescriptors(const GraphicsSplatSorter& sorter) {
    const vk::Device device = Context::Instance().getDevice().getDevice();

    for (vk::DescriptorSet set : graphicsDescriptorSets_) {
        std::array<vk::DescriptorBufferInfo, 4> infos{};
        std::array<vk::WriteDescriptorSet, 4> writes{};
        uint32_t writeCount = 0;

        auto addWrite = [&](uint32_t binding, vk::DescriptorType type, vk::Buffer buffer,
                            vk::DeviceSize range) {
            if (!buffer) {
                return;
            }
            infos[writeCount].setBuffer(buffer).setOffset(0).setRange(range);
            writes[writeCount]
                .setDstSet(set)
                .setDstBinding(binding)
                .setDstArrayElement(0)
                .setDescriptorCount(1)
                .setDescriptorType(type)
                .setPBufferInfo(&infos[writeCount]);
            ++writeCount;
        };

        addWrite(0, vk::DescriptorType::eUniformBuffer, uniformBuffer_.getBuffer(),
                 sizeof(GraphicsUniformData));
        addWrite(1, vk::DescriptorType::eUniformBuffer, screenInfoBuffer_.getBuffer(),
                 sizeof(glm::vec4));
        addWrite(2, vk::DescriptorType::eStorageBuffer, instanceBuffer_.getBuffer(), VK_WHOLE_SIZE);
        addWrite(3, vk::DescriptorType::eStorageBuffer, sorter.indexBuffer(), VK_WHOLE_SIZE);
        if (writeCount > 0u) {
            device.updateDescriptorSets(writeCount, writes.data(), 0, nullptr);
        }
    }

    if (!sorter.indexBuffer() || !sorter.keyBuffer() || !sorter.indirectBuffer() ||
        !instanceBuffer_.getBuffer() || !uniformBuffer_.getBuffer()) {
        LOG_DEBUG("Skipping radix descriptor update until sort buffers exist");
        return;
    }

    for (vk::DescriptorSet set : computeDescriptorSets_) {
        std::array<vk::DescriptorBufferInfo, 5> infos{};
        infos[0].setBuffer(sorter.indexBuffer()).setOffset(0).setRange(VK_WHOLE_SIZE);
        infos[1].setBuffer(sorter.keyBuffer()).setOffset(0).setRange(VK_WHOLE_SIZE);
        infos[2].setBuffer(instanceBuffer_.getBuffer()).setOffset(0).setRange(VK_WHOLE_SIZE);
        infos[3]
            .setBuffer(uniformBuffer_.getBuffer())
            .setOffset(0)
            .setRange(sizeof(GraphicsUniformData));
        infos[4]
            .setBuffer(sorter.indirectBuffer())
            .setOffset(0)
            .setRange(sizeof(VkDrawIndexedIndirectCommand));

        constexpr std::array<uint32_t, 5> bindings = {0, 1, 6, 7, 8};
        std::array<vk::WriteDescriptorSet, bindings.size()> writes{};
        for (size_t index = 0; index < bindings.size(); ++index) {
            writes[index]
                .setDstSet(set)
                .setDstBinding(bindings[index])
                .setDstArrayElement(0)
                .setDescriptorCount(1)
                .setDescriptorType(bindings[index] == 7 ? vk::DescriptorType::eUniformBuffer
                                                        : vk::DescriptorType::eStorageBuffer)
                .setPBufferInfo(&infos[index]);
        }
        device.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0,
                                    nullptr);
    }
}

void GraphicsSplatResources::updateUniform(const GraphicsUniformData& uniform,
                                           vk::Extent2D extent) {
    const vk::Device device = Context::Instance().getDevice().getDevice();
    void* data = device.mapMemory(uniformBuffer_.getMemory(), 0, sizeof(GraphicsUniformData));
    std::memcpy(data, &uniform, sizeof(GraphicsUniformData));
    device.unmapMemory(uniformBuffer_.getMemory());

    const glm::vec4 screenInfo(static_cast<float>(extent.width), static_cast<float>(extent.height),
                               0.0f, 0.0f);
    data = device.mapMemory(screenInfoBuffer_.getMemory(), 0, sizeof(screenInfo));
    std::memcpy(data, &screenInfo, sizeof(screenInfo));
    device.unmapMemory(screenInfoBuffer_.getMemory());
}

vk::DescriptorSet GraphicsSplatResources::graphicsDescriptorSet(uint32_t frameIndex) const {
    return frameIndex < graphicsDescriptorSets_.size() ? graphicsDescriptorSets_[frameIndex]
                                                       : vk::DescriptorSet{};
}

vk::DescriptorSet GraphicsSplatResources::computeDescriptorSet(uint32_t frameIndex) const {
    const size_t index = static_cast<size_t>(frameIndex) * 2u;
    return index < computeDescriptorSets_.size() ? computeDescriptorSets_[index]
                                                 : vk::DescriptorSet{};
}

void GraphicsSplatResources::createBuffers() {
    auto& context = Context::Instance();
    const vk::Device device = context.getDevice().getDevice();
    const vk::PhysicalDevice physicalDevice = context.PhysicalDevice();
    const vk::Queue queue = context.getDevice().getGraphicsQueue();
    const uint32_t queueFamily = graphicsQueueFamilyIndex();

    GraphicsUniformData initialUniform{};
    uniformBuffer_.create(device, physicalDevice, queue, queueFamily, &initialUniform,
                          sizeof(initialUniform), vk::BufferUsageFlagBits::eUniformBuffer,
                          vk::MemoryPropertyFlagBits::eHostVisible |
                              vk::MemoryPropertyFlagBits::eHostCoherent);
    const glm::vec4 initialScreenInfo{};
    screenInfoBuffer_.create(device, physicalDevice, queue, queueFamily, &initialScreenInfo,
                             sizeof(initialScreenInfo), vk::BufferUsageFlagBits::eUniformBuffer,
                             vk::MemoryPropertyFlagBits::eHostVisible |
                                 vk::MemoryPropertyFlagBits::eHostCoherent);
}

void GraphicsSplatResources::createDescriptorSets(uint32_t frameCount,
                                                  vk::DescriptorSetLayout graphicsLayout,
                                                  vk::DescriptorSetLayout computeLayout) {
    if (frameCount == 0u || !graphicsLayout || !computeLayout) {
        throw std::runtime_error("Invalid graphics descriptor allocation request");
    }

    const vk::Device device = Context::Instance().getDevice().getDevice();
    std::array<vk::DescriptorPoolSize, 2> poolSizes{};
    poolSizes[0].setType(vk::DescriptorType::eUniformBuffer).setDescriptorCount(frameCount * 4u);
    poolSizes[1].setType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(frameCount * 20u);
    vk::DescriptorPoolCreateInfo poolInfo{};
    poolInfo.setPoolSizeCount(static_cast<uint32_t>(poolSizes.size()))
        .setPPoolSizes(poolSizes.data())
        .setMaxSets(frameCount * 3u);
    descriptorPool_ = device.createDescriptorPool(poolInfo);

    std::vector<vk::DescriptorSetLayout> graphicsLayouts(frameCount, graphicsLayout);
    vk::DescriptorSetAllocateInfo graphicsAlloc{};
    graphicsAlloc.setDescriptorPool(descriptorPool_)
        .setDescriptorSetCount(static_cast<uint32_t>(graphicsLayouts.size()))
        .setPSetLayouts(graphicsLayouts.data());
    graphicsDescriptorSets_ = device.allocateDescriptorSets(graphicsAlloc);

    std::vector<vk::DescriptorSetLayout> computeLayouts(static_cast<size_t>(frameCount) * 2u,
                                                        computeLayout);
    vk::DescriptorSetAllocateInfo computeAlloc{};
    computeAlloc.setDescriptorPool(descriptorPool_)
        .setDescriptorSetCount(static_cast<uint32_t>(computeLayouts.size()))
        .setPSetLayouts(computeLayouts.data());
    computeDescriptorSets_ = device.allocateDescriptorSets(computeAlloc);
}

void GraphicsSplatResources::destroyDescriptorPool() {
    if (!descriptorPool_) {
        return;
    }
    Context::Instance().getDevice().getDevice().destroyDescriptorPool(descriptorPool_);
    descriptorPool_ = nullptr;
    graphicsDescriptorSets_.clear();
    computeDescriptorSets_.clear();
}

void GraphicsSplatResources::uploadModel(const GaussianModel& model) {
    struct alignas(8) PackedSH {
        uint32_t xy;
        uint32_t z0;
    };
    static_assert(sizeof(PackedSH) == sizeof(uint32_t) * 2);

    struct alignas(16) GaussianInstanceData {
        glm::vec4 position;
        glm::vec4 scale;
        glm::vec4 rotation;
        PackedSH sh0;
        PackedSH sh1[3];
        PackedSH sh2[5];
        PackedSH sh3[7];
    };
    static_assert(sizeof(GaussianInstanceData) == sizeof(glm::vec4) * 3 + sizeof(PackedSH) * 16);

    const auto packSH = [](const glm::vec3& value) {
        return PackedSH{glm::packHalf2x16(glm::vec2(value.x, value.y)),
                        glm::packHalf2x16(glm::vec2(value.z, 0.0f))};
    };

    const uint32_t pointCount = static_cast<uint32_t>(std::distance(model.begin(), model.end()));
    std::vector<GaussianInstanceData> instances(pointCount);
    size_t index = 0;
    for (const auto& point : model) {
        auto& instance = instances[index++];
        instance.position = glm::vec4(point.position, point.alpha);
        instance.scale = glm::vec4(point.scale, 0.0f);
        instance.rotation =
            glm::vec4(point.rotation.x, point.rotation.y, point.rotation.z, point.rotation.w);
        instance.sh0 = packSH(point.color.sh0);
        for (int coefficient = 0; coefficient < 3; ++coefficient) {
            instance.sh1[coefficient] = packSH(point.color.sh1[coefficient]);
        }
        for (int coefficient = 0; coefficient < 5; ++coefficient) {
            instance.sh2[coefficient] = packSH(point.color.sh2[coefficient]);
        }
        for (int coefficient = 0; coefficient < 7; ++coefficient) {
            instance.sh3[coefficient] = packSH(point.color.sh3[coefficient]);
        }
    }

    auto& context = Context::Instance();
    const vk::DeviceSize bufferSize = instances.size() * sizeof(GaussianInstanceData);
    instanceBuffer_.create(context.getDevice().getDevice(), context.PhysicalDevice(),
                           context.getDevice().getGraphicsQueue(), graphicsQueueFamilyIndex(),
                           instances.data(), bufferSize,
                           vk::BufferUsageFlagBits::eStorageBuffer |
                               vk::BufferUsageFlagBits::eTransferSrc |
                               vk::BufferUsageFlagBits::eTransferDst,
                           vk::MemoryPropertyFlagBits::eDeviceLocal);

    std::ostringstream sizeMb;
    sizeMb << std::fixed << std::setprecision(2)
           << static_cast<double>(bufferSize) / (1024.0 * 1024.0);
    LOG_INFO("Created instance SSBO with {} instances ({} bytes, {} MB)", pointCount, bufferSize,
             sizeMb.str());
}

} // namespace vulkan3DGS
