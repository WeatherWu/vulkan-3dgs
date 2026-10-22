#pragma once

#include "training/core/training_checkpoint_layout.hpp"
#include <vulkan/vulkan.hpp>

namespace vulkan3DGS {

inline constexpr std::array<uint32_t, 12> kCheckpointBindings = {3,  4,  5,  14, 43, 44,
                                                                 45, 46, 47, 48, 49, 50};

struct TrainingCheckpointResources {
    std::array<vk::DescriptorBufferInfo, 12> descriptors{};
    TrainingCheckpointCapacity capacity{};
};

} // namespace vulkan3DGS
