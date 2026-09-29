#include "training/core/training_validation_service.hpp"

#include "utils/logger.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace vulkan3DGS {
namespace {

uint32_t findHostVisibleMemoryType(vk::PhysicalDevice physicalDevice, uint32_t typeFilter) {
    const vk::PhysicalDeviceMemoryProperties properties = physicalDevice.getMemoryProperties();
    constexpr vk::MemoryPropertyFlags required =
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
    for (uint32_t index = 0; index < properties.memoryTypeCount; ++index) {
        if ((typeFilter & (1u << index)) != 0u &&
            (properties.memoryTypes[index].propertyFlags & required) == required) {
            return index;
        }
    }
    throw std::runtime_error("Failed to find host-visible validation readback memory");
}

} // namespace

TrainingValidationService::~TrainingValidationService() noexcept {
    try {
        cleanup();
    } catch (...) {
        std::fputs("TrainingValidationService cleanup failed during destruction\n", stderr);
    }
}

void TrainingValidationService::initialize(vk::Device device, vk::PhysicalDevice physicalDevice) {
    if (slots_[0].buffer) return;
    device_ = device;
    physicalDevice_ = physicalDevice;
    try {
        for (ReadbackSlot& readback : slots_) {
            vk::BufferCreateInfo bufferInfo{};
            bufferInfo.setSize(sizeof(TrainingValidationGpuResult))
                .setUsage(vk::BufferUsageFlagBits::eTransferDst)
                .setSharingMode(vk::SharingMode::eExclusive);
            readback.buffer = device_.createBuffer(bufferInfo);

            const vk::MemoryRequirements requirements =
                device_.getBufferMemoryRequirements(readback.buffer);
            vk::MemoryAllocateInfo allocationInfo{};
            allocationInfo.setAllocationSize(requirements.size)
                .setMemoryTypeIndex(
                    findHostVisibleMemoryType(physicalDevice_, requirements.memoryTypeBits));
            readback.memory = device_.allocateMemory(allocationInfo);
            device_.bindBufferMemory(readback.buffer, readback.memory, 0);
            readback.mapped =
                device_.mapMemory(readback.memory, 0, sizeof(TrainingValidationGpuResult));
            readback.fence = device_.createFence(vk::FenceCreateInfo{});
        }
    } catch (...) {
        cleanup();
        throw;
    }
    nextSlot_ = 0;
}

void TrainingValidationService::cleanup() {
    for (ReadbackSlot& readback : slots_) {
        if (readback.pending && readback.fence && device_) {
            (void)device_.waitForFences(readback.fence, VK_TRUE, UINT64_MAX);
        }
        if (readback.fence && device_) device_.destroyFence(readback.fence);
        if (readback.mapped && readback.memory && device_) {
            device_.unmapMemory(readback.memory);
        }
        if (readback.buffer && device_) device_.destroyBuffer(readback.buffer);
        if (readback.memory && device_) device_.freeMemory(readback.memory);
        readback = {};
    }
    nextSlot_ = 0;
    device_ = nullptr;
    physicalDevice_ = nullptr;
    resetStats();
}

std::optional<TrainingValidationService::ReadbackTicket>
TrainingValidationService::acquire(TrainingValidationReadbackMetadata metadata) {
    for (size_t offset = 0; offset < slots_.size(); ++offset) {
        const size_t index = (nextSlot_ + offset) % slots_.size();
        ReadbackSlot& readback = slots_[index];
        if (readback.pending) continue;
        nextSlot_ = (index + 1u) % slots_.size();
        readback.iteration = metadata.iteration;
        readback.tileItemCount = metadata.tileItemCount;
        readback.gaussianCount = metadata.gaussianCount;
        return ReadbackTicket{index};
    }
    return std::nullopt;
}

void TrainingValidationService::recordCopy(vk::CommandBuffer commandBuffer, vk::Buffer sourceBuffer,
                                           ReadbackTicket ticket) const {
    vk::BufferCopy copyRegion{};
    copyRegion.setSize(sizeof(TrainingValidationGpuResult));
    commandBuffer.copyBuffer(sourceBuffer, slot(ticket).buffer, copyRegion);
}

vk::Fence TrainingValidationService::submissionFence(ReadbackTicket ticket) const {
    return slot(ticket).fence;
}

void TrainingValidationService::markSubmitted(ReadbackTicket ticket) {
    slot(ticket).pending = true;
}

void TrainingValidationService::collect(TrainingExtent extent, uint32_t gaussianCapacity) {
    for (ReadbackSlot& readback : slots_) {
        if (!readback.pending) continue;
        const vk::Result status = device_.getFenceStatus(readback.fence);
        if (status == vk::Result::eNotReady) continue;
        if (status != vk::Result::eSuccess) {
            LOG_WARN("Failed to poll training validation readback fence: {}",
                     vk::to_string(status));
            continue;
        }

        TrainingValidationGpuResult result{};
        std::memcpy(&result, readback.mapped, sizeof(result));
        if (result.validationIteration != readback.iteration) {
            LOG_WARN("Ignoring stale training validation result: expected iteration {}, got {}",
                     readback.iteration, result.validationIteration);
        } else {
            consumeResult(result, readback.tileItemCount, readback.gaussianCount, extent,
                          gaussianCapacity);
        }
        readback.pending = false;
    }
}

void TrainingValidationService::resetStats() {
    validationStats_ = {};
    candidateProfileStats_ = {};
}

void TrainingValidationService::resetCandidateProfileStats() {
    candidateProfileStats_ = {};
}

void TrainingValidationService::consumeResult(const TrainingValidationGpuResult& result,
                                              uint32_t tileItemCount, uint32_t gaussianCount,
                                              TrainingExtent extent, uint32_t gaussianCapacity) {
    TrainingValidationStats stats{};
    stats.tileItemCount = tileItemCount;
    const uint32_t pixelCount = extent.width * extent.height;
    if (pixelCount == 0u || gaussianCount == 0u) {
        stats.valid = false;
        validationStats_ = stats;
        LOG_WARN("Training validation failed: empty pixel or gaussian count");
        return;
    }

    stats.meanLoss = result.lossSum;
    stats.maxLoss = result.maxLoss;
    stats.invalidLossCount = result.invalidLossCount;
    stats.invalidRenderedPixelCount = result.invalidRenderedPixelCount;
    stats.nonFiniteGaussianCount = result.nonFiniteGaussianCount;
    stats.nonFinitePositionCount = result.nonFinitePositionCount;
    stats.nonFiniteOpacityCount = result.nonFiniteOpacityCount;
    stats.nonFiniteRawScaleCount = result.nonFiniteRawScaleCount;
    stats.nonFiniteActivatedScaleCount = result.nonFiniteActivatedScaleCount;
    stats.nonFiniteRotationCount = result.nonFiniteRotationCount;
    stats.nonFiniteSHCount = result.nonFiniteSHCount;
    stats.firstNonFiniteGaussianIndex = result.firstNonFiniteGaussianIndex;
    if (result.validRenderedPixelCount > 0u) {
        const float count = static_cast<float>(result.validRenderedPixelCount);
        stats.meanRenderedAlpha = result.alphaSum / count;
        stats.meanRenderedLuminance = result.luminanceSum / count;
    }

    const float pixels = static_cast<float>(pixelCount);
    stats.meanProcessedCandidatesPerPixel = result.processedCandidateSum / pixels;
    stats.meanContributorsPerPixel = result.contributorSum / pixels;
    stats.maxProcessedCandidatesPerPixel = result.maxProcessedCandidates;
    for (size_t bucket = 0; bucket < 4u; ++bucket) {
        const auto glmIndex = static_cast<glm::length_t>(bucket);
        stats.processedCandidateHistogram[bucket] = result.processedCandidateBucketsLow[glmIndex];
        stats.processedCandidateHistogram[bucket + 4u] =
            result.processedCandidateBucketsHigh[glmIndex];
    }
    stats.renderedNonEmpty =
        stats.meanRenderedAlpha > 1e-5f || std::abs(stats.meanRenderedLuminance) > 1e-5f;

    ++candidateProfileStats_.sampleCount;
    const float samples = static_cast<float>(candidateProfileStats_.sampleCount);
    candidateProfileStats_.meanProcessedCandidatesPerPixel +=
        (stats.meanProcessedCandidatesPerPixel -
         candidateProfileStats_.meanProcessedCandidatesPerPixel) /
        samples;
    candidateProfileStats_.meanContributorsPerPixel +=
        (stats.meanContributorsPerPixel - candidateProfileStats_.meanContributorsPerPixel) /
        samples;
    candidateProfileStats_.maxProcessedCandidatesPerPixel =
        std::max(candidateProfileStats_.maxProcessedCandidatesPerPixel,
                 stats.maxProcessedCandidatesPerPixel);
    for (size_t bucket = 0; bucket < stats.processedCandidateHistogram.size(); ++bucket) {
        const float fraction =
            static_cast<float>(stats.processedCandidateHistogram[bucket]) / pixels;
        candidateProfileStats_.meanPixelFractionByProcessedBucket[bucket] +=
            (fraction - candidateProfileStats_.meanPixelFractionByProcessedBucket[bucket]) /
            samples;
    }

    stats.valid = stats.invalidLossCount == 0u && stats.invalidRenderedPixelCount == 0u &&
                  stats.nonFiniteGaussianCount == 0u && stats.renderedNonEmpty &&
                  stats.tileItemCount > 0u && gaussianCount <= gaussianCapacity;
    validationStats_ = stats;
    if (!stats.valid) {
        LOG_WARN("Training validation issue at iteration {}: loss={} invalidLoss={} "
                 "invalidPixels={} nonFiniteGaussians={} firstNonFinite={} categories(position={}, "
                 "opacity={}, rawScale={}, activatedScale={}, rotation={}, sh={}) tileItems={} "
                 "alpha={} luminance={}",
                 result.validationIteration, stats.meanLoss, stats.invalidLossCount,
                 stats.invalidRenderedPixelCount, stats.nonFiniteGaussianCount,
                 stats.firstNonFiniteGaussianIndex, stats.nonFinitePositionCount,
                 stats.nonFiniteOpacityCount, stats.nonFiniteRawScaleCount,
                 stats.nonFiniteActivatedScaleCount, stats.nonFiniteRotationCount,
                 stats.nonFiniteSHCount, stats.tileItemCount, stats.meanRenderedAlpha,
                 stats.meanRenderedLuminance);
    }
}

TrainingValidationService::ReadbackSlot& TrainingValidationService::slot(ReadbackTicket ticket) {
    if (ticket.slotIndex >= slots_.size()) {
        throw std::out_of_range("Training validation readback ticket is invalid");
    }
    return slots_[ticket.slotIndex];
}

const TrainingValidationService::ReadbackSlot&
TrainingValidationService::slot(ReadbackTicket ticket) const {
    if (ticket.slotIndex >= slots_.size()) {
        throw std::out_of_range("Training validation readback ticket is invalid");
    }
    return slots_[ticket.slotIndex];
}

} // namespace vulkan3DGS
