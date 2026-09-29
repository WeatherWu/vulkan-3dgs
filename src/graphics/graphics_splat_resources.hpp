#pragma once

#include "vulkan/buffer.hpp"

#include <glm/glm.hpp>
#include <vulkan/vulkan.hpp>

#include <cstdint>
#include <vector>

namespace vulkan3DGS {

class GaussianModel;
class GraphicsSplatSorter;

struct GraphicsUniformData {
    alignas(16) glm::mat4 view{1.0f};
    alignas(16) glm::mat4 projection{1.0f};
    alignas(16) glm::mat4 model{1.0f};
    alignas(16) glm::vec4 cameraPositionTime{};
    alignas(16) glm::vec4 focal{};
    alignas(16) glm::uvec4 renderSettings{};
};

class GraphicsSplatResources {
public:
    GraphicsSplatResources() = default;
    ~GraphicsSplatResources();

    GraphicsSplatResources(const GraphicsSplatResources&) = delete;
    GraphicsSplatResources& operator=(const GraphicsSplatResources&) = delete;

    void initialize(uint32_t frameCount, vk::DescriptorSetLayout graphicsLayout,
                    vk::DescriptorSetLayout computeLayout);
    void cleanup();
    void resetModel();
    void releaseDescriptors();
    void rebuildDescriptors(uint32_t frameCount, vk::DescriptorSetLayout graphicsLayout,
                            vk::DescriptorSetLayout computeLayout,
                            const GraphicsSplatSorter& sorter);

    void ensureModelUploaded(const GaussianModel& model, const GraphicsSplatSorter& sorter);
    void updateDescriptors(const GraphicsSplatSorter& sorter);
    void updateUniform(const GraphicsUniformData& uniform, vk::Extent2D extent);

    vk::Buffer instanceBuffer() const {
        return instanceBuffer_.getBuffer();
    }
    vk::Buffer uniformBuffer() const {
        return uniformBuffer_.getBuffer();
    }
    vk::DescriptorSet graphicsDescriptorSet(uint32_t frameIndex) const;
    vk::DescriptorSet computeDescriptorSet(uint32_t frameIndex) const;

private:
    void createBuffers();
    void createDescriptorSets(uint32_t frameCount, vk::DescriptorSetLayout graphicsLayout,
                              vk::DescriptorSetLayout computeLayout);
    void destroyDescriptorPool();
    void uploadModel(const GaussianModel& model);

    Buffer instanceBuffer_;
    Buffer uniformBuffer_;
    Buffer screenInfoBuffer_;
    vk::DescriptorPool descriptorPool_;
    std::vector<vk::DescriptorSet> graphicsDescriptorSets_;
    std::vector<vk::DescriptorSet> computeDescriptorSets_;
};

} // namespace vulkan3DGS
