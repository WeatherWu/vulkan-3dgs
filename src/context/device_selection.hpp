#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace vulkan3DGS {

struct GpuSelectionCandidate {
    uint32_t vulkanIndex = 0;
    std::string name;
    std::string uuid;
};

struct GpuSelectionResult {
    std::optional<size_t> candidateIndex;
    std::string error;

    explicit operator bool() const { return candidateIndex.has_value(); }
};

GpuSelectionResult selectGpuCandidate(const std::vector<GpuSelectionCandidate>& candidates,
                                      const std::string& selector);

} // namespace vulkan3DGS
