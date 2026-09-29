#pragma once

#include "image/image_streamer.hpp"
#include "training/cache/device_image_cache.hpp"
#include "training/core/training_buffers.hpp"
#include "training/core/training_dataset.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace vulkan3DGS {

struct TrainingImageUploadResult {
    float imageRequestMilliseconds = 0.0f;
    float targetUploadMilliseconds = 0.0f;
};

class TrainingImageRuntime {
public:
    TrainingImageRuntime() = default;
    ~TrainingImageRuntime() = default;

    TrainingImageRuntime(const TrainingImageRuntime&) = delete;
    TrainingImageRuntime& operator=(const TrainingImageRuntime&) = delete;

    void initialize(vk::Device device, vk::PhysicalDevice physicalDevice, vk::Queue transferQueue,
                    uint32_t transferQueueFamilyIndex, uint32_t computeQueueFamilyIndex,
                    const TrainingDataset& dataset, TrainingBuffers& buffers);
    void cleanup(TrainingBuffers& buffers);

    [[nodiscard]] TrainingImageUploadResult uploadFrame(size_t frameIndex,
                                                        const TrainingCameraFrame& frame,
                                                        const TrainingForwardCamera& camera,
                                                        TrainingBuffers& buffers);
    void prefetch(std::span<const size_t> frameIndices);
    void reserveForDensification(uint32_t trainingIteration, TrainingBuffers& buffers);
    void refreshBudget(uint32_t trainingIteration, TrainingBuffers& buffers);

    [[nodiscard]] ImageStreamerStats imageCacheStats() const;
    [[nodiscard]] DeviceImageCacheStats deviceImageCacheStats() const;
    [[nodiscard]] uint64_t pendingUploadValue() const noexcept {
        return pendingUploadValue_;
    }
    void clearPendingUpload() noexcept {
        pendingUploadValue_ = 0;
    }
    [[nodiscard]] vk::Semaphore uploadSemaphore() const noexcept;

    [[nodiscard]] static std::vector<ImageSourceDesc> makeSources(const TrainingDataset& dataset);

private:
    [[nodiscard]] uint64_t initialDeviceCacheBudget() const;
    bool refreshDeviceCache(bool reserveForDensification, uint32_t trainingIteration,
                            TrainingBuffers& buffers);

    vk::Device device_ = nullptr;
    vk::PhysicalDevice physicalDevice_ = nullptr;
    vk::Queue transferQueue_ = nullptr;
    uint32_t transferQueueFamilyIndex_ = 0;
    uint32_t computeQueueFamilyIndex_ = 0;
    uint32_t frameWidth_ = 0;
    uint32_t frameHeight_ = 0;
    uint32_t imageCount_ = 0;
    std::unique_ptr<ImageStreamer> imageStreamer_;
    std::unique_ptr<DeviceImageCache> deviceImageCache_;
    uint64_t pendingUploadValue_ = 0;
    uint32_t deviceCacheGrowthResumeIteration_ = 0;
};

} // namespace vulkan3DGS
