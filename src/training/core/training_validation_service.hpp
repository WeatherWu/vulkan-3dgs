#pragma once

#include "training/core/training_types.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

#include <vulkan/vulkan.hpp>

namespace vulkan3DGS {

struct TrainingValidationReadbackMetadata {
    uint32_t iteration = 0;
    uint32_t tileItemCount = 0;
    uint32_t gaussianCount = 0;
};

class TrainingValidationService {
public:
    struct ReadbackTicket {
        size_t slotIndex = 0;
    };

    TrainingValidationService() = default;
    ~TrainingValidationService() noexcept;

    TrainingValidationService(const TrainingValidationService&) = delete;
    TrainingValidationService& operator=(const TrainingValidationService&) = delete;

    void initialize(vk::Device device, vk::PhysicalDevice physicalDevice);
    void cleanup();

    [[nodiscard]] std::optional<ReadbackTicket>
    acquire(TrainingValidationReadbackMetadata metadata);
    void recordCopy(vk::CommandBuffer commandBuffer, vk::Buffer sourceBuffer,
                    ReadbackTicket ticket) const;
    [[nodiscard]] vk::Fence submissionFence(ReadbackTicket ticket) const;
    void markSubmitted(ReadbackTicket ticket);
    void collect(TrainingExtent extent, uint32_t gaussianCapacity);
    void consumeResult(const TrainingValidationGpuResult& result, uint32_t tileItemCount,
                       uint32_t gaussianCount, TrainingExtent extent, uint32_t gaussianCapacity);

    void resetStats();
    void resetCandidateProfileStats();

    [[nodiscard]] const TrainingValidationStats& stats() const noexcept {
        return validationStats_;
    }
    [[nodiscard]] const TrainingCandidateProfileStats& candidateProfileStats() const noexcept {
        return candidateProfileStats_;
    }

private:
    struct ReadbackSlot {
        vk::Buffer buffer = nullptr;
        vk::DeviceMemory memory = nullptr;
        void* mapped = nullptr;
        vk::Fence fence = nullptr;
        uint32_t iteration = 0;
        uint32_t tileItemCount = 0;
        uint32_t gaussianCount = 0;
        bool pending = false;
    };

    [[nodiscard]] ReadbackSlot& slot(ReadbackTicket ticket);
    [[nodiscard]] const ReadbackSlot& slot(ReadbackTicket ticket) const;

    vk::Device device_ = nullptr;
    vk::PhysicalDevice physicalDevice_ = nullptr;
    std::array<ReadbackSlot, 3> slots_{};
    size_t nextSlot_ = 0;
    TrainingValidationStats validationStats_{};
    TrainingCandidateProfileStats candidateProfileStats_{};
};

} // namespace vulkan3DGS
