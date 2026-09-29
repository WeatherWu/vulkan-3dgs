#pragma once

#include "training/core/training_types.hpp"

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace vulkan3DGS {

class TrainingFrameScheduler {
public:
    void configure(const TrainingScheduleConfig& config);
    void setFrameCount(size_t frameCount);
    void restartSequence();
    void clear();

    [[nodiscard]] size_t selectForIteration();
    void advanceAfterIteration();
    void setCurrentFrame(size_t frameIndex);
    [[nodiscard]] std::vector<size_t> upcomingFrames(size_t maximumCount) const;

    [[nodiscard]] bool isComplete(uint32_t trainingIteration) const noexcept;
    [[nodiscard]] size_t currentFrame() const noexcept {
        return currentFrame_;
    }
    [[nodiscard]] size_t frameCount() const noexcept {
        return frameCount_;
    }
    [[nodiscard]] uint32_t totalIterations() const noexcept {
        return config_.totalIterations;
    }
    [[nodiscard]] const TrainingScheduleConfig& config() const noexcept {
        return config_;
    }

private:
    TrainingScheduleConfig config_{};
    size_t frameCount_ = 0;
    size_t currentFrame_ = 0;
    std::mt19937 rng_{1u};
    std::vector<size_t> randomFrameStack_;
};

} // namespace vulkan3DGS
