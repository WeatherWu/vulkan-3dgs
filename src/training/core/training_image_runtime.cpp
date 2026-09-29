#include "training/core/training_image_runtime.hpp"

#include "image/image_decoder.hpp"
#include "utils/logger.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>

namespace vulkan3DGS {
namespace {

uint64_t alignUp(uint64_t value, uint64_t alignment) {
    if (alignment <= 1u) return value;
    return ((value + alignment - 1u) / alignment) * alignment;
}

} // namespace

void TrainingImageRuntime::initialize(
    vk::Device device,
    vk::PhysicalDevice physicalDevice,
    vk::Queue transferQueue,
    uint32_t transferQueueFamilyIndex,
    uint32_t computeQueueFamilyIndex,
    const TrainingDataset& dataset,
    TrainingBuffers& buffers) {
    cleanup(buffers);
    if (dataset.empty()) return;

    device_ = device;
    physicalDevice_ = physicalDevice;
    transferQueue_ = transferQueue;
    transferQueueFamilyIndex_ = transferQueueFamilyIndex;
    computeQueueFamilyIndex_ = computeQueueFamilyIndex;
    frameWidth_ = dataset.frames.front().width;
    frameHeight_ = dataset.frames.front().height;
    imageCount_ = static_cast<uint32_t>(dataset.size());

    imageStreamer_ = std::make_unique<ImageStreamer>();
    imageStreamer_->refreshMemoryBudget();
    imageStreamer_->setSources(makeSources(dataset));

    try {
        auto cache = std::make_unique<DeviceImageCache>();
        cache->initialize(device_, physicalDevice_, transferQueue_,
                          transferQueueFamilyIndex_,
                          computeQueueFamilyIndex_, frameWidth_, frameHeight_,
                          imageCount_, initialDeviceCacheBudget());
        const DeviceImageCacheStats stats = cache->stats();
        LOG_INFO("Initialized device image cache with {} slots ({:.1f} MiB, mode {})",
                 stats.slotCount,
                 static_cast<double>(stats.allocatedBytes) / (1024.0 * 1024.0),
                 static_cast<uint32_t>(stats.mode));
        deviceImageCache_ = std::move(cache);
    } catch (const std::exception& error) {
        LOG_WARN("Device image cache unavailable; using the per-frame target buffer: {}",
                 error.what());
    }
}

void TrainingImageRuntime::cleanup(TrainingBuffers& buffers) {
    buffers.clearTargetColorDescriptor();
    if (deviceImageCache_) {
        deviceImageCache_->cleanup();
        deviceImageCache_.reset();
    }
    imageStreamer_.reset();
    pendingUploadValue_ = 0;
    deviceCacheGrowthResumeIteration_ = 0;
    device_ = nullptr;
    physicalDevice_ = nullptr;
    transferQueue_ = nullptr;
    transferQueueFamilyIndex_ = 0;
    computeQueueFamilyIndex_ = 0;
    frameWidth_ = 0;
    frameHeight_ = 0;
    imageCount_ = 0;
}

TrainingImageUploadResult TrainingImageRuntime::uploadFrame(
    size_t frameIndex,
    const TrainingCameraFrame& frame,
    const TrainingForwardCamera& camera,
    TrainingBuffers& buffers) {
    const TrainingExtent extent = buffers.extent();
    if (frame.width != extent.width || frame.height != extent.height) {
        throw std::runtime_error(
            "Dataset frame size does not match GaussianTraining extent. Resize training buffers to " +
            std::to_string(frame.width) + "x" + std::to_string(frame.height) +
            " before trainStep.");
    }

    TrainingImageUploadResult result{};
    const auto requestStart = std::chrono::steady_clock::now();
    if (imageStreamer_) {
        const ImageHandle handle =
            imageStreamer_->request(static_cast<ImageId>(frameIndex));
        const ImageRgba8& image = handle.image();
        result.imageRequestMilliseconds =
            std::chrono::duration<float, std::milli>(
                std::chrono::steady_clock::now() - requestStart).count();
        if (image.width != frame.width || image.height != frame.height) {
            throw std::runtime_error(
                "Cached training image dimensions do not match frame metadata");
        }

        const auto uploadStart = std::chrono::steady_clock::now();
        if (deviceImageCache_ && deviceImageCache_->isInitialized()) {
            const DeviceImageBinding binding =
                deviceImageCache_->getOrUpload(
                    static_cast<ImageId>(frameIndex), image);
            buffers.setTargetColorDescriptor(binding.descriptor);
            pendingUploadValue_ = binding.readyValue;
        } else {
            buffers.clearTargetColorDescriptor();
            pendingUploadValue_ = 0;
            buffers.uploadTargetColor(image.pixels.data(), image.width,
                                      image.height);
        }
        buffers.uploadCamera(camera);
        result.targetUploadMilliseconds =
            std::chrono::duration<float, std::milli>(
                std::chrono::steady_clock::now() - uploadStart).count();
        return result;
    }

    ImageSourceDesc source{};
    source.path = frame.imagePath;
    source.expectedWidth = frame.width;
    source.expectedHeight = frame.height;
    const ImageRgba8 image = ImageDecoder::decodeRgba8(source);
    result.imageRequestMilliseconds =
        std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - requestStart).count();
    const auto uploadStart = std::chrono::steady_clock::now();
    buffers.clearTargetColorDescriptor();
    pendingUploadValue_ = 0;
    buffers.uploadTargetColor(image.pixels.data(), image.width, image.height);
    buffers.uploadCamera(camera);
    result.targetUploadMilliseconds =
        std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - uploadStart).count();
    return result;
}

void TrainingImageRuntime::prefetch(
    std::span<const size_t> frameIndices) {
    if (!imageStreamer_ || frameIndices.empty()) return;
    std::vector<ImageId> imageIds;
    imageIds.reserve(frameIndices.size());
    for (const size_t frameIndex : frameIndices) {
        imageIds.push_back(static_cast<ImageId>(frameIndex));
    }
    imageStreamer_->prefetch(imageIds);
}

void TrainingImageRuntime::reserveForDensification(
    uint32_t trainingIteration,
    TrainingBuffers& buffers) {
    deviceCacheGrowthResumeIteration_ = trainingIteration + 33u;
    (void)refreshDeviceCache(true, trainingIteration, buffers);
}

void TrainingImageRuntime::refreshBudget(
    uint32_t trainingIteration,
    TrainingBuffers& buffers) {
    (void)refreshDeviceCache(false, trainingIteration, buffers);
}

ImageStreamerStats TrainingImageRuntime::imageCacheStats() const {
    return imageStreamer_ ? imageStreamer_->stats() : ImageStreamerStats{};
}

DeviceImageCacheStats TrainingImageRuntime::deviceImageCacheStats() const {
    return deviceImageCache_ ? deviceImageCache_->stats()
                             : DeviceImageCacheStats{};
}

vk::Semaphore TrainingImageRuntime::uploadSemaphore() const noexcept {
    return deviceImageCache_ ? deviceImageCache_->uploadSemaphore() : nullptr;
}

std::vector<ImageSourceDesc> TrainingImageRuntime::makeSources(
    const TrainingDataset& dataset) {
    std::vector<ImageSourceDesc> sources;
    sources.reserve(dataset.frames.size());
    for (size_t frameIndex = 0; frameIndex < dataset.frames.size();
         ++frameIndex) {
        const TrainingCameraFrame& frame = dataset.frames[frameIndex];
        ImageSourceDesc source{};
        source.id = static_cast<ImageId>(frameIndex);
        source.path = frame.imagePath;
        source.expectedWidth = frame.width;
        source.expectedHeight = frame.height;
        source.format = ImagePixelFormat::Rgba8Unorm;
        source.colorSpace = ImageColorSpace::LinearUnorm;
        sources.push_back(std::move(source));
    }
    return sources;
}

uint64_t TrainingImageRuntime::initialDeviceCacheBudget() const {
    if (!physicalDevice_ || imageCount_ == 0u) return 0;

    const uint64_t imageBytes =
        static_cast<uint64_t>(frameWidth_) * frameHeight_ * sizeof(uint32_t);
    const vk::PhysicalDeviceLimits limits =
        physicalDevice_.getProperties().limits;
    const uint64_t slotStride = alignUp(
        imageBytes,
        std::max<uint64_t>(limits.minStorageBufferOffsetAlignment, 1u));
    const uint64_t fullDatasetBytes = slotStride * imageCount_;
    const vk::PhysicalDeviceMemoryProperties memory =
        physicalDevice_.getMemoryProperties();
    uint64_t largestDeviceLocalHeap = 0;
    uint32_t deviceLocalHeapIndex = 0;
    for (uint32_t typeIndex = 0; typeIndex < memory.memoryTypeCount;
         ++typeIndex) {
        const vk::MemoryType& type = memory.memoryTypes[typeIndex];
        if ((type.propertyFlags & vk::MemoryPropertyFlagBits::eDeviceLocal) ==
                vk::MemoryPropertyFlagBits::eDeviceLocal &&
            memory.memoryHeaps[type.heapIndex].size > largestDeviceLocalHeap) {
            largestDeviceLocalHeap = memory.memoryHeaps[type.heapIndex].size;
            deviceLocalHeapIndex = type.heapIndex;
        }
    }

    const auto extensions = physicalDevice_.enumerateDeviceExtensionProperties();
    const bool hasMemoryBudget = std::any_of(
        extensions.begin(), extensions.end(),
        [](const vk::ExtensionProperties& extension) {
            return std::strcmp(extension.extensionName.data(),
                               vk::EXTMemoryBudgetExtensionName) == 0;
        });
    if (hasMemoryBudget) {
        vk::PhysicalDeviceMemoryBudgetPropertiesEXT budgetProperties{};
        vk::PhysicalDeviceMemoryProperties2 memoryProperties{};
        memoryProperties.pNext = &budgetProperties;
        physicalDevice_.getMemoryProperties2(&memoryProperties);
        const uint64_t heapBudget =
            budgetProperties.heapBudget[deviceLocalHeapIndex];
        const uint64_t heapUsage =
            budgetProperties.heapUsage[deviceLocalHeapIndex];
        const uint64_t freeBytes =
            heapBudget > heapUsage ? heapBudget - heapUsage : 0u;
        constexpr uint64_t minimumReserveBytes =
            uint64_t{512} * 1024u * 1024u;
        const uint64_t reserveBytes = std::min(
            heapBudget / 2u,
            std::max(minimumReserveBytes, heapBudget * 15u / 100u));
        const uint64_t usableBytes =
            freeBytes > reserveBytes ? freeBytes - reserveBytes : imageBytes;
        return std::min(fullDatasetBytes,
                        std::max(imageBytes,
                                 std::min(usableBytes, heapBudget / 4u)));
    }
    const uint64_t automaticBudget = largestDeviceLocalHeap / 20u;
    return std::min(fullDatasetBytes,
                    std::max(imageBytes, automaticBudget));
}

bool TrainingImageRuntime::refreshDeviceCache(
    bool reserveForDensification,
    uint32_t trainingIteration,
    TrainingBuffers& buffers) {
    if (!deviceImageCache_ || !deviceImageCache_->isInitialized()) {
        return false;
    }
    const bool allowGrowth = !reserveForDensification &&
        trainingIteration >= deviceCacheGrowthResumeIteration_;
    if (!deviceImageCache_->refreshMemoryBudget(allowGrowth,
                                                reserveForDensification)) {
        return false;
    }
    buffers.clearTargetColorDescriptor();
    pendingUploadValue_ = 0;
    const DeviceImageCacheStats stats = deviceImageCache_->stats();
    LOG_INFO("Resized device image cache to {} slots ({:.1f} MiB){}",
             stats.slotCount,
             static_cast<double>(stats.allocatedBytes) /
                 (1024.0 * 1024.0),
             reserveForDensification ? " before densification" : "");
    return true;
}

} // namespace vulkan3DGS
