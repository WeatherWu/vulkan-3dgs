#pragma once

#include "vulkan/buffer.hpp"
#include "vulkan/compute_pipeline.hpp"

#include <glm/mat4x4.hpp>
#include <vk_radix_sort.h>
#include <vulkan/vulkan.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace vulkan3DGS {

class GraphicsSplatSorter {
public:
    GraphicsSplatSorter() = default;
    ~GraphicsSplatSorter();

    GraphicsSplatSorter(const GraphicsSplatSorter&) = delete;
    GraphicsSplatSorter& operator=(const GraphicsSplatSorter&) = delete;

    void initialize();
    void cleanup();
    void resetModelResources();
    void beginFrame();

    // Returns true when buffer handles changed and descriptors must be rewritten.
    bool prepare(uint32_t pointCount, const void* modelIdentity, const glm::mat4& view,
                 const glm::mat4& projection, const glm::mat4& model,
                 const std::vector<vk::Fence>& inFlightFences);

    void record(vk::CommandBuffer commandBuffer, vk::DescriptorSet keygenDescriptorSet) const;
    void recordReadBarrier(vk::CommandBuffer commandBuffer) const;

    vk::DescriptorSetLayout descriptorSetLayout() const;
    vk::Buffer indexBuffer() const {
        return indexBuffer_.getBuffer();
    }
    vk::Buffer keyBuffer() const {
        return keyBuffer_.getBuffer();
    }
    vk::Buffer indirectBuffer() const {
        return indirectBuffer_.getBuffer();
    }
    bool recordsSortThisFrame() const {
        return recordSortThisFrame_;
    }
    uint32_t pointCountThisFrame() const {
        return pointCountThisFrame_;
    }

private:
    bool ensureCapacity(uint32_t pointCount, const std::vector<vk::Fence>& inFlightFences);
    void invalidateSortCache();

    std::unique_ptr<ComputePipeline> keygenPipeline_;
    VrdxSorter radixSorter_ = VK_NULL_HANDLE;
    Buffer indexBuffer_;
    Buffer keyBuffer_;
    Buffer storageBuffer_;
    Buffer indirectBuffer_;
    uint32_t capacity_ = 0;

    bool sortCompleted_ = false;
    uint32_t lastPointCount_ = 0;
    const void* lastModelIdentity_ = nullptr;
    glm::mat4 lastView_{1.0f};
    glm::mat4 lastProjection_{1.0f};
    glm::mat4 lastModel_{1.0f};

    bool recordSortThisFrame_ = false;
    uint32_t pointCountThisFrame_ = 0;
};

} // namespace vulkan3DGS
