#include "vulkan/device_image_cache.hpp"

#include "utils/logger.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace vulkan3DGS {

namespace {

vk::DeviceSize alignUp(vk::DeviceSize value, vk::DeviceSize alignment) {
    if (alignment <= 1u) {
        return value;
    }
    return ((value + alignment - 1u) / alignment) * alignment;
}

bool forceSynchronousImageUploads() {
#ifdef _WIN32
    char* value = nullptr;
    size_t length = 0;
    if (_dupenv_s(&value, &length, "VULKAN_3DGS_SYNC_IMAGE_UPLOAD") != 0 || !value) {
        return false;
    }
    const bool enabled = std::strcmp(value, "1") == 0;
    std::free(value);
    return enabled;
#else
    const char* value = std::getenv("VULKAN_3DGS_SYNC_IMAGE_UPLOAD");
    return value && std::strcmp(value, "1") == 0;
#endif
}

} // namespace

DeviceImageCache::~DeviceImageCache() {
    cleanup();
}

void DeviceImageCache::initialize(vk::Device device,
                                  vk::PhysicalDevice physicalDevice,
                                  vk::Queue transferQueue,
                                  uint32_t transferQueueFamilyIndex,
                                  uint32_t width,
                                  uint32_t height,
                                  uint32_t imageCount,
                                  uint64_t budgetBytes) {
    cleanup();
    if (width == 0u || height == 0u || imageCount == 0u) {
        throw std::runtime_error("DeviceImageCache requires non-empty images and source set");
    }

    device_ = device;
    physicalDevice_ = physicalDevice;
    transferQueue_ = transferQueue;
    transferQueueFamilyIndex_ = transferQueueFamilyIndex;
    imageCount_ = imageCount;
    const vk::PhysicalDeviceLimits limits = physicalDevice.getProperties().limits;
    imageBytes_ = static_cast<vk::DeviceSize>(width) * static_cast<vk::DeviceSize>(height) * sizeof(uint32_t);
    if (imageBytes_ > limits.maxStorageBufferRange) {
        throw std::runtime_error("Training target image exceeds maxStorageBufferRange");
    }
    slotStride_ = alignUp(imageBytes_, std::max<vk::DeviceSize>(limits.minStorageBufferOffsetAlignment, 1u));

    const uint64_t safeBudget = std::max<uint64_t>(budgetBytes, slotStride_);
    const uint64_t budgetSlots = safeBudget / slotStride_;
    const uint32_t slotCount = static_cast<uint32_t>(
        std::min<uint64_t>(std::max<uint64_t>(budgetSlots, 1u), imageCount));
    const vk::DeviceSize allocationSize = slotStride_ * slotCount;
    buffer_.create(device,
                   physicalDevice,
                   transferQueue,
                   transferQueueFamilyIndex,
                   nullptr,
                   allocationSize,
                   vk::BufferUsageFlagBits::eStorageBuffer,
                   vk::MemoryPropertyFlagBits::eDeviceLocal);

    slots_.assign(slotCount, Slot{});
    stats_ = {};
    stats_.budgetBytes = safeBudget;
    stats_.allocatedBytes = allocationSize;
    stats_.slotCount = slotCount;
    stats_.totalImages = imageCount;
    stats_.mode = slotCount >= imageCount
        ? DeviceImageCacheMode::Full
        : (slotCount > 1u ? DeviceImageCacheMode::Partial : DeviceImageCacheMode::Streaming);
    initializeUploadResources(physicalDevice, transferQueue, transferQueueFamilyIndex);
    stats_.asynchronousUploads = asynchronousUploads_;
    const vk::PhysicalDeviceMemoryProperties memory = physicalDevice.getMemoryProperties();
    uint64_t largestHeap = 0;
    for (uint32_t typeIndex = 0; typeIndex < memory.memoryTypeCount; ++typeIndex) {
        const vk::MemoryType& type = memory.memoryTypes[typeIndex];
        if ((type.propertyFlags & vk::MemoryPropertyFlagBits::eDeviceLocal) == vk::MemoryPropertyFlagBits::eDeviceLocal &&
            memory.memoryHeaps[type.heapIndex].size > largestHeap) {
            largestHeap = memory.memoryHeaps[type.heapIndex].size;
            deviceLocalHeapIndex_ = type.heapIndex;
        }
    }
    initialized_ = true;
    refreshMemoryBudget(false, false);
}

void DeviceImageCache::cleanup() {
    try {
        waitForUploads();
    } catch (const std::exception& error) {
        LOG_WARN("Failed to wait for device image uploads during cleanup: {}", error.what());
    }
    destroyUploadResources();
    buffer_.cleanup();
    slots_.clear();
    imageSlots_.clear();
    imageBytes_ = 0;
    slotStride_ = 0;
    stagingStride_ = 0;
    nextUploadValue_ = 0;
    uploadCounter_ = 0;
    useCounter_ = 0;
    stats_ = {};
    device_ = nullptr;
    physicalDevice_ = nullptr;
    transferQueue_ = nullptr;
    transferQueueFamilyIndex_ = 0;
    imageCount_ = 0;
    deviceLocalHeapIndex_ = 0;
    initialized_ = false;
}

DeviceImageBinding DeviceImageCache::getOrUpload(ImageId id, const ImageRgba8& image) {
    if (!initialized_) {
        throw std::runtime_error("DeviceImageCache is not initialized");
    }
    const vk::DeviceSize sourceBytes = static_cast<vk::DeviceSize>(image.width) *
                                       static_cast<vk::DeviceSize>(image.height) * sizeof(uint32_t);
    if (sourceBytes != imageBytes_ || image.pixels.size() != sourceBytes) {
        throw std::runtime_error("DeviceImageCache image dimensions do not match its slots");
    }

    uint32_t slotIndex = 0;
    const auto existing = imageSlots_.find(id);
    if (existing != imageSlots_.end()) {
        slotIndex = existing->second;
        ++stats_.hits;
    } else {
        ++stats_.misses;
        slotIndex = selectSlot();
        Slot& slot = slots_[slotIndex];
        if (slot.imageId) {
            imageSlots_.erase(*slot.imageId);
            ++stats_.evictions;
        }
        slot.readyValue = asynchronousUploads_
            ? uploadAsync(slotIndex, image)
            : (buffer_.upload(image.pixels.data(), imageBytes_, slotStride_ * slotIndex), 0u);
        slot.imageId = id;
        imageSlots_.insert_or_assign(id, slotIndex);
        ++stats_.uploads;
        stats_.residentImages = static_cast<uint32_t>(imageSlots_.size());
    }

    slots_[slotIndex].lastUse = ++useCounter_;
    DeviceImageBinding binding{};
    binding.descriptor.setBuffer(buffer_.getBuffer())
                      .setOffset(slotStride_ * slotIndex)
                      .setRange(imageBytes_);
    if (asynchronousUploads_ && slots_[slotIndex].readyValue > 0) {
        const uint64_t completedValue = device_.getSemaphoreCounterValue(uploadSemaphore_);
        if (slots_[slotIndex].readyValue > completedValue) {
            binding.readyValue = slots_[slotIndex].readyValue;
        }
    }
    return binding;
}

bool DeviceImageCache::refreshMemoryBudget(bool allowGrowth, bool reserveForDensification) {
    if (!physicalDevice_ || slots_.empty()) {
        return false;
    }

    const auto extensions = physicalDevice_.enumerateDeviceExtensionProperties();
    const bool hasMemoryBudget = std::any_of(
        extensions.begin(), extensions.end(),
        [](const vk::ExtensionProperties& extension) {
            return std::strcmp(extension.extensionName.data(), vk::EXTMemoryBudgetExtensionName) == 0;
        });

    uint64_t heapBudget = 0;
    uint64_t heapUsage = 0;
    if (hasMemoryBudget) {
        vk::PhysicalDeviceMemoryBudgetPropertiesEXT budgetProperties{};
        vk::PhysicalDeviceMemoryProperties2 memoryProperties{};
        memoryProperties.pNext = &budgetProperties;
        physicalDevice_.getMemoryProperties2(&memoryProperties);
        heapBudget = budgetProperties.heapBudget[deviceLocalHeapIndex_];
        heapUsage = budgetProperties.heapUsage[deviceLocalHeapIndex_];
    } else {
        const vk::PhysicalDeviceMemoryProperties memory = physicalDevice_.getMemoryProperties();
        heapBudget = memory.memoryHeaps[deviceLocalHeapIndex_].size;
    }

    stats_.memoryBudgetAvailable = hasMemoryBudget;
    stats_.heapBudgetBytes = heapBudget;
    stats_.heapUsageBytes = heapUsage;
    const uint64_t currentBytes = stats_.allocatedBytes;
    const uint64_t freeBytes = heapBudget > heapUsage ? heapBudget - heapUsage : 0;
    const uint64_t fixedReserve = reserveForDensification
        ? 1024ull * 1024ull * 1024ull
        : 512ull * 1024ull * 1024ull;
    const uint64_t proportionalReserve = reserveForDensification
        ? heapBudget / 4u
        : heapBudget * 15u / 100u;
    const uint64_t reserveBytes = std::min(heapBudget / 2u,
                                          std::max(fixedReserve, proportionalReserve));
    const uint64_t reclaimableBytes = hasMemoryBudget ? currentBytes + freeBytes : heapBudget;
    uint64_t targetBytes = reclaimableBytes > reserveBytes ? reclaimableBytes - reserveBytes : slotStride_;
    targetBytes = std::min<uint64_t>(targetBytes, heapBudget / 4u);
    targetBytes = std::min<uint64_t>(targetBytes, slotStride_ * imageCount_);
    targetBytes = std::max<uint64_t>(targetBytes, slotStride_);
    if (!allowGrowth) {
        targetBytes = std::min(targetBytes, currentBytes);
    }

    const uint32_t targetSlots = static_cast<uint32_t>(std::clamp<uint64_t>(
        targetBytes / slotStride_, 1u, imageCount_));
    const uint32_t currentSlots = static_cast<uint32_t>(slots_.size());
    const uint32_t growThreshold = std::max(2u, currentSlots / 4u);
    const uint32_t shrinkThreshold = std::max(1u, currentSlots / 5u);
    const bool shouldGrow = targetSlots > currentSlots && targetSlots - currentSlots >= growThreshold;
    const bool shouldShrink = targetSlots < currentSlots && currentSlots - targetSlots >= shrinkThreshold;
    stats_.budgetBytes = targetBytes;
    if (!shouldGrow && !shouldShrink) {
        return false;
    }

    resizeSlots(targetSlots, targetBytes);
    return true;
}

DeviceImageCacheStats DeviceImageCache::stats() const {
    DeviceImageCacheStats result = stats_;
    if (asynchronousUploads_ && device_ && uploadSemaphore_) {
        const uint64_t completedValue = device_.getSemaphoreCounterValue(uploadSemaphore_);
        result.pendingUploads = static_cast<uint32_t>(std::count_if(
            uploadSlots_.begin(), uploadSlots_.end(),
            [completedValue](const UploadSlot& slot) {
                return slot.completionValue > completedValue;
            }));
    }
    return result;
}

void DeviceImageCache::waitForUploads() {
    if (!asynchronousUploads_ || !device_ || !uploadSemaphore_ || nextUploadValue_ == 0) {
        return;
    }
    vk::SemaphoreWaitInfo waitInfo{};
    waitInfo.setSemaphores(uploadSemaphore_).setValues(nextUploadValue_);
    const vk::Result result = device_.waitSemaphores(waitInfo, std::numeric_limits<uint64_t>::max());
    if (result != vk::Result::eSuccess) {
        throw std::runtime_error("Failed to wait for device image uploads");
    }
}

uint32_t DeviceImageCache::selectSlot() {
    for (uint32_t index = 0; index < slots_.size(); ++index) {
        if (!slots_[index].imageId) {
            return index;
        }
    }

    uint32_t selected = 0;
    uint64_t oldestUse = std::numeric_limits<uint64_t>::max();
    for (uint32_t index = 0; index < slots_.size(); ++index) {
        if (slots_[index].lastUse < oldestUse) {
            oldestUse = slots_[index].lastUse;
            selected = index;
        }
    }
    return selected;
}

void DeviceImageCache::resizeSlots(uint32_t slotCount, uint64_t targetBudgetBytes) {
    waitForUploads();
    Buffer replacement;
    const vk::DeviceSize allocationSize = slotStride_ * slotCount;
    replacement.create(device_,
                       physicalDevice_,
                       transferQueue_,
                       transferQueueFamilyIndex_,
                       nullptr,
                       allocationSize,
                       vk::BufferUsageFlagBits::eStorageBuffer,
                       vk::MemoryPropertyFlagBits::eDeviceLocal);

    stats_.evictions += imageSlots_.size();
    imageSlots_.clear();
    slots_.assign(slotCount, Slot{});
    buffer_ = std::move(replacement);
    stats_.budgetBytes = targetBudgetBytes;
    stats_.allocatedBytes = allocationSize;
    stats_.slotCount = slotCount;
    stats_.residentImages = 0;
    stats_.mode = slotCount >= imageCount_
        ? DeviceImageCacheMode::Full
        : (slotCount > 1u ? DeviceImageCacheMode::Partial : DeviceImageCacheMode::Streaming);
    ++stats_.budgetResizes;
}

void DeviceImageCache::initializeUploadResources(vk::PhysicalDevice physicalDevice,
                                                 vk::Queue transferQueue,
                                                 uint32_t transferQueueFamilyIndex) {
    if (forceSynchronousImageUploads()) {
        LOG_WARN("Using synchronous device image uploads because VULKAN_3DGS_SYNC_IMAGE_UPLOAD=1");
        asynchronousUploads_ = false;
        return;
    }

    vk::PhysicalDeviceTimelineSemaphoreFeatures timelineFeatures{};
    vk::PhysicalDeviceFeatures2 features{};
    features.setPNext(&timelineFeatures);
    physicalDevice.getFeatures2(&features);
    if (!timelineFeatures.timelineSemaphore) {
        asynchronousUploads_ = false;
        return;
    }

    constexpr uint32_t kUploadRingSize = 3u;
    const vk::PhysicalDeviceLimits limits = physicalDevice.getProperties().limits;
    stagingStride_ = alignUp(imageBytes_, std::max<vk::DeviceSize>(limits.nonCoherentAtomSize, 4u));
    const vk::DeviceSize stagingBytes = stagingStride_ * kUploadRingSize;

    vk::BufferCreateInfo bufferInfo{};
    bufferInfo.setSize(stagingBytes)
              .setUsage(vk::BufferUsageFlagBits::eTransferSrc)
              .setSharingMode(vk::SharingMode::eExclusive);
    stagingBuffer_ = device_.createBuffer(bufferInfo);

    const vk::MemoryRequirements requirements = device_.getBufferMemoryRequirements(stagingBuffer_);
    vk::MemoryAllocateInfo allocation{};
    allocation.setAllocationSize(requirements.size)
              .setMemoryTypeIndex(findHostMemoryType(physicalDevice,
                                                     requirements.memoryTypeBits,
                                                     stagingCoherent_));
    stagingMemory_ = device_.allocateMemory(allocation);
    device_.bindBufferMemory(stagingBuffer_, stagingMemory_, 0);
    stagingMapped_ = device_.mapMemory(stagingMemory_, 0, stagingBytes);

    vk::CommandPoolCreateInfo poolInfo{};
    poolInfo.setQueueFamilyIndex(transferQueueFamilyIndex)
            .setFlags(vk::CommandPoolCreateFlagBits::eTransient |
                      vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
    uploadCommandPool_ = device_.createCommandPool(poolInfo);

    vk::CommandBufferAllocateInfo commandAllocation{};
    commandAllocation.setCommandPool(uploadCommandPool_)
                     .setLevel(vk::CommandBufferLevel::ePrimary)
                     .setCommandBufferCount(kUploadRingSize);
    const std::vector<vk::CommandBuffer> commandBuffers = device_.allocateCommandBuffers(commandAllocation);
    uploadSlots_.resize(kUploadRingSize);
    for (uint32_t index = 0; index < kUploadRingSize; ++index) {
        uploadSlots_[index].commandBuffer = commandBuffers[index];
    }

    vk::SemaphoreTypeCreateInfo timelineInfo{};
    timelineInfo.setSemaphoreType(vk::SemaphoreType::eTimeline).setInitialValue(0);
    vk::SemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.setPNext(&timelineInfo);
    uploadSemaphore_ = device_.createSemaphore(semaphoreInfo);

    transferQueue_ = transferQueue;
    stats_.stagingBytes = stagingBytes;
    asynchronousUploads_ = true;
}

void DeviceImageCache::destroyUploadResources() {
    if (!device_) {
        return;
    }
    if (stagingMapped_ && stagingMemory_) {
        device_.unmapMemory(stagingMemory_);
        stagingMapped_ = nullptr;
    }
    if (uploadSemaphore_) {
        device_.destroySemaphore(uploadSemaphore_);
        uploadSemaphore_ = nullptr;
    }
    if (uploadCommandPool_) {
        device_.destroyCommandPool(uploadCommandPool_);
        uploadCommandPool_ = nullptr;
    }
    uploadSlots_.clear();
    if (stagingBuffer_) {
        device_.destroyBuffer(stagingBuffer_);
        stagingBuffer_ = nullptr;
    }
    if (stagingMemory_) {
        device_.freeMemory(stagingMemory_);
        stagingMemory_ = nullptr;
    }
    asynchronousUploads_ = false;
}

uint64_t DeviceImageCache::uploadAsync(uint32_t slotIndex, const ImageRgba8& image) {
    const uint32_t uploadSlotIndex = static_cast<uint32_t>(uploadCounter_++ % uploadSlots_.size());
    UploadSlot& uploadSlot = uploadSlots_[uploadSlotIndex];
    if (uploadSlot.completionValue > 0) {
        const uint64_t completedValue = device_.getSemaphoreCounterValue(uploadSemaphore_);
        if (completedValue < uploadSlot.completionValue) {
            vk::SemaphoreWaitInfo waitInfo{};
            waitInfo.setSemaphores(uploadSemaphore_).setValues(uploadSlot.completionValue);
            const vk::Result result = device_.waitSemaphores(waitInfo, std::numeric_limits<uint64_t>::max());
            if (result != vk::Result::eSuccess) {
                throw std::runtime_error("Failed to recycle device image staging slot");
            }
            ++stats_.uploadWaits;
        }
    }

    const vk::DeviceSize stagingOffset = stagingStride_ * uploadSlotIndex;
    std::memcpy(static_cast<std::byte*>(stagingMapped_) + stagingOffset,
                image.pixels.data(),
                static_cast<size_t>(imageBytes_));
    if (!stagingCoherent_) {
        vk::MappedMemoryRange range{};
        range.setMemory(stagingMemory_).setOffset(stagingOffset).setSize(stagingStride_);
        device_.flushMappedMemoryRanges(range);
    }

    uploadSlot.commandBuffer.reset();
    vk::CommandBufferBeginInfo beginInfo{};
    beginInfo.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    uploadSlot.commandBuffer.begin(beginInfo);
    vk::BufferCopy copy{};
    copy.setSrcOffset(stagingOffset)
        .setDstOffset(slotStride_ * slotIndex)
        .setSize(imageBytes_);
    uploadSlot.commandBuffer.copyBuffer(stagingBuffer_, buffer_.getBuffer(), copy);
    uploadSlot.commandBuffer.end();

    const uint64_t signalValue = ++nextUploadValue_;
    vk::TimelineSemaphoreSubmitInfo timelineSubmit{};
    timelineSubmit.setSignalSemaphoreValues(signalValue);
    vk::SubmitInfo submit{};
    submit.setPNext(&timelineSubmit)
          .setCommandBuffers(uploadSlot.commandBuffer)
          .setSignalSemaphores(uploadSemaphore_);
    transferQueue_.submit(submit);
    uploadSlot.completionValue = signalValue;
    return signalValue;
}

uint32_t DeviceImageCache::findHostMemoryType(vk::PhysicalDevice physicalDevice,
                                              uint32_t memoryTypeBits,
                                              bool& coherent) const {
    const vk::PhysicalDeviceMemoryProperties memory = physicalDevice.getMemoryProperties();
    for (uint32_t index = 0; index < memory.memoryTypeCount; ++index) {
        const vk::MemoryPropertyFlags flags = memory.memoryTypes[index].propertyFlags;
        if ((memoryTypeBits & (1u << index)) != 0 &&
            (flags & (vk::MemoryPropertyFlagBits::eHostVisible |
                      vk::MemoryPropertyFlagBits::eHostCoherent)) ==
                (vk::MemoryPropertyFlagBits::eHostVisible |
                 vk::MemoryPropertyFlagBits::eHostCoherent)) {
            coherent = true;
            return index;
        }
    }
    for (uint32_t index = 0; index < memory.memoryTypeCount; ++index) {
        const vk::MemoryPropertyFlags flags = memory.memoryTypes[index].propertyFlags;
        if ((memoryTypeBits & (1u << index)) != 0 &&
            (flags & vk::MemoryPropertyFlagBits::eHostVisible) == vk::MemoryPropertyFlagBits::eHostVisible) {
            coherent = false;
            return index;
        }
    }
    throw std::runtime_error("DeviceImageCache could not find host-visible staging memory");
}

} // namespace vulkan3DGS
