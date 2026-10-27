#pragma once
#include <cstdint>
#include <filesystem>
#include <stdexcept>

namespace vulkan3DGS {
// Non-persisted diagnostic boundary, separate from the normal training schedule.
struct TrainingSmokeOptions {
    uint32_t stopIteration = 0;
    std::filesystem::path report;
    void validate() const {
        if (!stopIteration || stopIteration >= 30000 || report.empty())
            throw std::invalid_argument(
                "Training smoke requires 1..29999 steps and a new report path");
    }
};
} // namespace vulkan3DGS
