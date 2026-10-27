#pragma once
#include "training/app/training_smoke_options.hpp"
#include "training/core/training_scheduling_diagnostics.hpp"
#include "training/core/training_types.hpp"

namespace vulkan3DGS {
std::string trainingSmokeBuildJson();
// Worker-owned bounded diagnostic. Reserves output before any smoke step; does not configure
// training.
class TrainingSmokeRun {
public:
    TrainingSmokeRun(TrainingSmokeOptions options, std::string identityJson);
    ~TrainingSmokeRun() noexcept;
    bool reached(uint32_t completedIteration) const {
        return completedIteration >= options_.stopIteration;
    }
    void observe(const TrainingStepObservation& sample);
    void finish(std::string_view reason, std::string_view error,
                std::optional<uint64_t> validationErrors, std::optional<double> loopWallMs,
                const std::optional<TrainingStateFiniteCheck>& finalState);
    bool written() const {
        return report_.written();
    }
    const std::filesystem::path& path() const {
        return options_.report;
    }

private:
    TrainingSmokeOptions options_;
    std::string identity_;
    TrainingDiagnosticReport report_;
    std::vector<TrainingStepObservation> samples_;
};
} // namespace vulkan3DGS
