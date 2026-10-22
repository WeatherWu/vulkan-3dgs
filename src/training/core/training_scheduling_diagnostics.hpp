#pragma once

#include <vulkan/vulkan.hpp>

#include <array>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace vulkan3DGS {

// Diagnostic-only output: reserve exclusively before any measured work.
class TrainingDiagnosticReport {
public:
    explicit TrainingDiagnosticReport(const std::filesystem::path& path);
    void write(std::string_view json);
    bool written() const noexcept {
        return written_;
    }

private:
    std::unique_ptr<FILE, decltype(&std::fclose)> file_{nullptr, &std::fclose};
    bool written_ = false;
};

std::string schedulingJsonQuote(std::string_view value);
struct SchedulingDiagnosticConfig {
    std::filesystem::path output;
    uint32_t samples = 128, startIteration = 0;
    bool synthetic = false;
};
std::optional<SchedulingDiagnosticConfig>
schedulingDiagnosticConfig(const char* output, const char* samples, const char* startIteration);
std::optional<double> schedulingTimestampInterval(uint64_t begin, uint64_t end, uint32_t validBits,
                                                  double periodNs);
struct SchedulingGpuIntervals {
    std::optional<double> prepareDurationMs, prepareToMainGapMs;
};
// Query order is prepare begin / prepare end / main begin.
SchedulingGpuIntervals schedulingGpuIntervals(const std::array<uint64_t, 3>& ticks,
                                              const std::array<bool, 3>& available,
                                              uint32_t validBits, double periodNs);
// Caller supplies instance configuration, not layer availability/default guesses.
// Omission is deliberately reported as null, not false.
std::string schedulingRuntimeMetadataJson(std::optional<bool> vulkanValidationLayersEnabled);

struct SchedulingStepMetadata {
    uint32_t iteration = 0, frame = 0, gaussians = 0, tileItems = 0, capacity = 0;
    uint32_t width = 0, height = 0, tileSize = 0, activeSHDegree = 0;
    uint32_t ssimMode = 0, pixelMode = 0, compositeMode = 0;
    uint64_t uploadWaitValue = 0;
    bool fixedBenchmark = false, adc = false, validation = false, opacityReset = false;
    bool resized = false, optimizerEnabled = false, optimizerCapture = false;
    bool ordinary() const noexcept {
        return optimizerEnabled && !adc && !validation && !opacityReset && !resized &&
               !fixedBenchmark && !optimizerCapture;
    }
};
struct SchedulingStepSample {
    SchedulingStepMetadata metadata;
    double prepareSubmitCallMs = 0, prepareFenceWaitMs = 0, countReadMs = 0;
    double capacityCheckMs = 0, resizeMs = 0, postFenceToMainSubmitMs = 0;
    double mainRecordMs = 0, mainSubmitCallMs = 0;
    std::optional<double> gpuPrepareToMainGapMs;
    std::optional<double> gpuPrepareDurationMs;
    std::string loggerLevel;
};
std::string schedulingDiagnosticJson(std::string_view identity,
                                     const std::vector<SchedulingStepSample>& samples,
                                     uint32_t requested, bool complete);

// Owned by TrainingProfilingService; no UI/settings/normal GPU resource ownership.
class TrainingSchedulingDiagnostics {
public:
    static std::unique_ptr<TrainingSchedulingDiagnostics>
    fromEnvironment(vk::Device device, vk::PhysicalDevice physical, uint32_t family,
                    std::optional<bool> vulkanValidationLayersEnabled = std::nullopt);
    TrainingSchedulingDiagnostics(vk::Device device, vk::PhysicalDevice physical, uint32_t family,
                                  SchedulingDiagnosticConfig config,
                                  std::optional<bool> vulkanValidationLayersEnabled = std::nullopt);
    ~TrainingSchedulingDiagnostics() noexcept;
    bool wantsStep(const SchedulingStepMetadata& metadata) const noexcept;
    void recordPrepareBegin(vk::CommandBuffer command);
    void recordPrepareEnd(vk::CommandBuffer command);
    void recordMainBegin(vk::CommandBuffer command);
    // Called after the ORIGINAL main fence has completed; adds no GPU wait.
    void finishStep(SchedulingStepSample sample) noexcept;

private:
    vk::Device device_;
    SchedulingDiagnosticConfig config_;
    vk::UniqueQueryPool queries_;
    uint32_t validBits_ = 0;
    double periodNs_ = 0;
    std::string identity_;
    std::vector<SchedulingStepSample> samples_;
    TrainingDiagnosticReport report_;
    bool failed_ = false;
};

} // namespace vulkan3DGS
