#include "training/core/training_resident_images.hpp"

#include "image/image_decoder.hpp"

#include <chrono>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace vulkan3DGS {

std::unique_ptr<TrainingResidentImages>
TrainingResidentImages::prepare(vk::Device device, vk::PhysicalDevice physical, vk::Queue transfer,
                                uint32_t transferFamily, uint32_t computeFamily,
                                const TrainingDataset& dataset, vk::Queue compute) {
    if (dataset.empty()) throw std::invalid_argument("GPU Resident requires a nonempty dataset");
    const auto started = std::chrono::steady_clock::now();
    auto result = std::make_unique<TrainingResidentImages>();
    result->physical_ = physical;
    for (const auto& extension : physical.enumerateDeviceExtensionProperties())
        result->memoryBudgetAvailable_ |=
            std::strcmp(extension.extensionName.data(), VK_EXT_MEMORY_BUDGET_EXTENSION_NAME) == 0;
    result->trainingCount_ = dataset.size();
    result->images_.reserve(dataset.size() + dataset.validationSize());
    CommandPool uploads;
    uploads.create(device, transferFamily, vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
    const auto command = uploads.allocateCommandBuffer();
    auto fence = device.createFenceUnique({});
    const auto upload = [&](const TrainingCameraFrame& frame) {
        ImageSourceDesc source{};
        source.path = frame.imagePath;
        source.expectedWidth = frame.width;
        source.expectedHeight = frame.height;
        source.format = ImagePixelFormat::Rgba8Unorm;
        source.colorSpace = ImageColorSpace::LinearUnorm;
        const auto image = ImageDecoder::decodeRgba8(source);
        if (image.width != frame.width || image.height != frame.height)
            throw std::runtime_error("GPU Resident image dimensions do not match camera metadata");
        const uint64_t bytes = uint64_t(frame.width) * frame.height * sizeof(uint32_t);
        if (bytes == 0 || bytes > physical.getProperties().limits.maxStorageBufferRange)
            throw std::runtime_error("GPU Resident image exceeds storage buffer limits");
        Buffer buffer;
        buffer.create(device, physical, transfer, transferFamily, nullptr, bytes,
                      vk::BufferUsageFlagBits::eStorageBuffer |
                          vk::BufferUsageFlagBits::eTransferDst |
                          vk::BufferUsageFlagBits::eTransferSrc,
                      vk::MemoryPropertyFlagBits::eDeviceLocal, {transferFamily, computeFamily});
        Buffer staging;
        staging.create(device, physical, transfer, transferFamily, image.pixels.data(), bytes,
                       vk::BufferUsageFlagBits::eTransferSrc,
                       vk::MemoryPropertyFlagBits::eHostVisible |
                           vk::MemoryPropertyFlagBits::eHostCoherent,
                       {transferFamily});
        command.reset();
        command.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        command.copyBuffer(staging.getBuffer(), buffer.getBuffer(), vk::BufferCopy{0, 0, bytes});
        command.end();
        transfer.submit(vk::SubmitInfo{}.setCommandBuffers(command), *fence);
        (void)device.waitForFences(*fence, true, UINT64_MAX);
        device.resetFences(*fence);
        result->bytes_ += buffer.getSize();
        result->images_.push_back(std::move(buffer));
    };
    for (const auto& frame : dataset.frames)
        upload(frame);
    for (const auto& frame : dataset.validationFrames)
        upload(frame);
    auto ready = device.createSemaphoreUnique({});
    transfer.submit(vk::SubmitInfo{}.setSignalSemaphores(*ready));
    const vk::PipelineStageFlags waitStage = vk::PipelineStageFlagBits::eAllCommands;
    if (!compute) compute = device.getQueue(computeFamily, 0);
    compute.submit(vk::SubmitInfo{}.setWaitSemaphores(*ready).setWaitDstStageMask(waitStage),
                   *fence);
    (void)device.waitForFences(*fence, true, UINT64_MAX);
    result->preparationSeconds_ =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    return result;
}

vk::DescriptorBufferInfo TrainingResidentImages::trainingImage(size_t frame) {
    if (frame >= trainingCount_) throw std::out_of_range("GPU Resident training frame");
    ++hits_;
    return images_[frame].getDescriptorInfo();
}

DeviceImageCacheStats TrainingResidentImages::stats() const {
    DeviceImageCacheStats result{};
    result.mode = DeviceImageCacheMode::Full;
    result.allocatedBytes = bytes_;
    result.budgetBytes = bytes_;
    result.slotCount = static_cast<uint32_t>(images_.size());
    result.totalImages = result.slotCount;
    result.residentImages = result.slotCount;
    result.hits = hits_;
    result.uploads = images_.size();
    if (memoryBudgetAvailable_) {
        vk::PhysicalDeviceMemoryBudgetPropertiesEXT budget;
        vk::PhysicalDeviceMemoryProperties2 memory;
        memory.pNext = &budget;
        physical_.getMemoryProperties2(&memory);
        uint64_t largest = 0;
        for (uint32_t i = 0; i < memory.memoryProperties.memoryHeapCount; ++i) {
            const auto& heap = memory.memoryProperties.memoryHeaps[i];
            if ((heap.flags & vk::MemoryHeapFlagBits::eDeviceLocal) && heap.size > largest) {
                largest = heap.size;
                result.heapBudgetBytes = budget.heapBudget[i];
                result.heapUsageBytes = budget.heapUsage[i];
                result.memoryBudgetAvailable = true;
            }
        }
    }
    return result;
}

} // namespace vulkan3DGS
