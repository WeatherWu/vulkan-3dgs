#include "training/core/training_frame_scheduler.hpp"

#include <array>
#include <iostream>
#include <set>
#include <stdexcept>
#include <vector>

using namespace vulkan3DGS;

namespace {

bool testSequentialSchedule() {
    TrainingFrameScheduler scheduler;
    TrainingScheduleConfig config{};
    config.imageSelectionMode = TrainingImageSelectionMode::Sequential;
    config.totalIterations = 0;
    scheduler.configure(config);
    scheduler.setFrameCount(3u);

    if (scheduler.selectForIteration() != 0u) return false;
    if (scheduler.upcomingFrames(2u) != std::vector<size_t>{1u, 2u}) {
        return false;
    }
    scheduler.advanceAfterIteration();
    if (scheduler.currentFrame() != 1u) return false;
    scheduler.advanceAfterIteration();
    scheduler.advanceAfterIteration();
    return scheduler.currentFrame() == 0u &&
           !scheduler.isComplete(100000u);
}

bool testRandomWithoutReplacementAndRestart() {
    TrainingScheduleConfig config{};
    config.imageSelectionMode = TrainingImageSelectionMode::Random;
    config.totalIterations = 8;
    config.randomSeed = 73;

    TrainingFrameScheduler first;
    TrainingFrameScheduler second;
    first.configure(config);
    second.configure(config);
    first.setFrameCount(4u);
    second.setFrameCount(4u);

    std::array<size_t, 8> sequence{};
    for (size_t index = 0; index < sequence.size(); ++index) {
        sequence[index] = first.selectForIteration();
        if (sequence[index] != second.selectForIteration()) return false;
    }
    if (std::set<size_t>(sequence.begin(), sequence.begin() + 4).size() != 4u ||
        std::set<size_t>(sequence.begin() + 4, sequence.end()).size() != 4u) {
        return false;
    }
    if (!first.isComplete(8u) || first.isComplete(7u)) return false;

    first.restartSequence();
    first.setFrameCount(4u);
    second.restartSequence();
    second.setFrameCount(4u);
    const size_t selected = first.selectForIteration();
    if (selected != second.selectForIteration()) return false;
    const std::vector<size_t> upcoming = first.upcomingFrames(2u);
    if (upcoming.size() != 2u) return false;
    return upcoming[0] == first.selectForIteration() &&
           upcoming[1] == first.selectForIteration();
}

bool testExplicitFrameValidation() {
    TrainingFrameScheduler scheduler;
    bool rejectedEmpty = false;
    try {
        scheduler.setCurrentFrame(0u);
    } catch (const std::runtime_error&) {
        rejectedEmpty = true;
    }
    scheduler.setFrameCount(2u);
    scheduler.setCurrentFrame(1u);
    bool rejectedRange = false;
    try {
        scheduler.setCurrentFrame(2u);
    } catch (const std::runtime_error&) {
        rejectedRange = true;
    }
    return rejectedEmpty && rejectedRange && scheduler.currentFrame() == 1u;
}

} // namespace

int main() {
    if (!testSequentialSchedule()) {
        std::cerr << "Sequential frame scheduling failed\n";
        return 1;
    }
    if (!testRandomWithoutReplacementAndRestart()) {
        std::cerr << "Random frame scheduling failed\n";
        return 1;
    }
    if (!testExplicitFrameValidation()) {
        std::cerr << "Explicit frame validation failed\n";
        return 1;
    }
    return 0;
}
