#include "training/core/training_frame_scheduler.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace vulkan3DGS {

void TrainingFrameScheduler::configure(
    const TrainingScheduleConfig& config) {
    config_ = config;
    restartSequence();
}

void TrainingFrameScheduler::setFrameCount(size_t frameCount) {
    frameCount_ = frameCount;
    currentFrame_ = 0;
    randomFrameStack_.clear();
}

void TrainingFrameScheduler::restartSequence() {
    rng_.seed(config_.randomSeed);
    randomFrameStack_.clear();
}

void TrainingFrameScheduler::clear() {
    frameCount_ = 0;
    currentFrame_ = 0;
    randomFrameStack_.clear();
}

size_t TrainingFrameScheduler::selectForIteration() {
    if (frameCount_ == 0u ||
        config_.imageSelectionMode != TrainingImageSelectionMode::Random) {
        return currentFrame_;
    }
    if (randomFrameStack_.empty()) {
        randomFrameStack_.resize(frameCount_);
        std::iota(randomFrameStack_.begin(), randomFrameStack_.end(), 0u);
        std::shuffle(randomFrameStack_.begin(), randomFrameStack_.end(), rng_);
    }
    currentFrame_ = randomFrameStack_.back();
    randomFrameStack_.pop_back();
    return currentFrame_;
}

void TrainingFrameScheduler::advanceAfterIteration() {
    if (frameCount_ > 0u &&
        config_.imageSelectionMode == TrainingImageSelectionMode::Sequential) {
        currentFrame_ = (currentFrame_ + 1u) % frameCount_;
    }
}

void TrainingFrameScheduler::setCurrentFrame(size_t frameIndex) {
    if (frameCount_ == 0u) {
        throw std::runtime_error(
            "Cannot set training frame index without a loaded dataset");
    }
    if (frameIndex >= frameCount_) {
        throw std::runtime_error("Training frame index is out of range");
    }
    currentFrame_ = frameIndex;
}

std::vector<size_t> TrainingFrameScheduler::upcomingFrames(
    size_t maximumCount) const {
    std::vector<size_t> result;
    result.reserve(maximumCount);
    if (frameCount_ == 0u || maximumCount == 0u) return result;

    if (config_.imageSelectionMode == TrainingImageSelectionMode::Random) {
        for (auto it = randomFrameStack_.rbegin();
             it != randomFrameStack_.rend() && result.size() < maximumCount;
             ++it) {
            result.push_back(*it);
        }
        return result;
    }

    for (size_t offset = 1u; offset <= maximumCount; ++offset) {
        result.push_back((currentFrame_ + offset) % frameCount_);
    }
    return result;
}

bool TrainingFrameScheduler::isComplete(
    uint32_t trainingIteration) const noexcept {
    return config_.totalIterations > 0u &&
           trainingIteration >= config_.totalIterations;
}

} // namespace vulkan3DGS
