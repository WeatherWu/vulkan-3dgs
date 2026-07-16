#pragma once

#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

#include <vulkan/vulkan.hpp>

#include "image/image_types.hpp"
#include "vulkan/buffer.hpp"

namespace vulkan3DGS {

enum class DeviceImageCacheMode : uint32_t {
    Streaming = 0,
    Partial,
    Full,
};

struct DeviceImageCacheStats {
    DeviceImageCacheMode mode = DeviceImageCacheMode::Streaming;
    uint64_t budgetBytes = 0;
    uint64_t allocatedBytes = 0;
    uint32_t slotCount = 0;
    uint32_t residentImages = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t uploads = 0;
    uint64_t evictions = 0;
    uint64_t stagingBytes = 0;
    uint64_t heapBudgetBytes = 0;
    uint64_t heapUsageBytes = 0;
    uint64_t budgetResizes = 0;
    uint64_t uploadWaits = 0;
    uint32_t pendingUploads = 0;
    bool asynchronousUploads = false;
    bool memoryBudgetAvailable = false;
};

struct DeviceImageBinding {
    vk::DescriptorBufferInfo descriptor{};
    uint64_t readyValue = 0;
};

class DeviceImageCache {
public:
    ~DeviceImageCache();

    void initialize(vk::Device device,
                    vk::PhysicalDevice physicalDevice,
                    vk::Queue transferQueue,
                    uint32_t transferQueueFamilyIndex,
                    uint32_t width,
                    uint32_t height,
                    uint32_t imageCount,
                    uint64_t budgetBytes);
    void cleanup();

    DeviceImageBinding getOrUpload(ImageId id, const ImageRgba8& image);
    bool refreshMemoryBudget(bool allowGrowth, bool reserveForDensification);
    DeviceImageCacheStats stats() const;
    bool isInitialized() const { return initialized_; }
    vk::Semaphore uploadSemaphore() const { return uploadSemaphore_; }
    void waitForUploads();

private:
    struct Slot {
        std::optional<ImageId> imageId;
        uint64_t lastUse = 0;
        uint64_t readyValue = 0;
    };

    struct UploadSlot {
        vk::CommandBuffer commandBuffer = nullptr;
        uint64_t completionValue = 0;
    };

    uint32_t selectSlot();
    void resizeSlots(uint32_t slotCount, uint64_t targetBudgetBytes);
    void initializeUploadResources(vk::PhysicalDevice physicalDevice,
                                   vk::Queue transferQueue,
                                   uint32_t transferQueueFamilyIndex);
    void destroyUploadResources();
    uint64_t uploadAsync(uint32_t slotIndex, const ImageRgba8& image);
    uint32_t findHostMemoryType(vk::PhysicalDevice physicalDevice,
                                uint32_t memoryTypeBits,
                                bool& coherent) const;

    Buffer buffer_;
    std::vector<Slot> slots_;
    std::unordered_map<ImageId, uint32_t> imageSlots_;
    vk::Device device_ = nullptr;
    vk::PhysicalDevice physicalDevice_ = nullptr;
    vk::Queue transferQueue_ = nullptr;
    uint32_t transferQueueFamilyIndex_ = 0;
    uint32_t imageCount_ = 0;
    uint32_t deviceLocalHeapIndex_ = 0;
    vk::Buffer stagingBuffer_ = nullptr;
    vk::DeviceMemory stagingMemory_ = nullptr;
    void* stagingMapped_ = nullptr;
    vk::CommandPool uploadCommandPool_ = nullptr;
    vk::Semaphore uploadSemaphore_ = nullptr;
    std::vector<UploadSlot> uploadSlots_;
    vk::DeviceSize imageBytes_ = 0;
    vk::DeviceSize slotStride_ = 0;
    vk::DeviceSize stagingStride_ = 0;
    uint64_t nextUploadValue_ = 0;
    uint64_t uploadCounter_ = 0;
    uint64_t useCounter_ = 0;
    bool stagingCoherent_ = true;
    bool asynchronousUploads_ = false;
    DeviceImageCacheStats stats_{};
    bool initialized_ = false;
};

} // namespace vulkan3DGS
