#include "training/core/training_scheduling_diagnostics.hpp"
#include "utils/logger.hpp"

#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace vulkan3DGS;
namespace {
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template <class F> void reject(F&& operation) {
    bool rejected = false;
    try {
        operation();
    } catch (const std::exception&) {
        rejected = true;
    }
    check(rejected, "Invalid scheduling diagnostic input was accepted");
}
void configuration() {
    check(!schedulingDiagnosticConfig(nullptr, nullptr, nullptr), "Diagnostics enabled by default");
    reject([] { schedulingDiagnosticConfig(nullptr, "128", nullptr); });
    for (auto invalid : {"0", "-1", "2049", "nan", "12x"})
        reject([&] { schedulingDiagnosticConfig("report.json", invalid, nullptr); });
    reject([] { schedulingDiagnosticConfig("report.json", "1", "-1"); });
    const auto defaults = schedulingDiagnosticConfig("report.json", nullptr, nullptr);
    check(defaults && defaults->samples == 128 && defaults->startIteration == 0,
          "Scheduling default configuration");
    const auto bounded = schedulingDiagnosticConfig("report.json", "2048", "8000");
    check(bounded && bounded->samples == 2048 && bounded->startIteration == 8000,
          "Scheduling explicit configuration");
}
void timestampsAndGroups() {
    check(!schedulingTimestampInterval(0, 1, 0, 1), "Unavailable timestamp accepted");
    check(!schedulingTimestampInterval(0, 1, 65, 1), "Invalid timestamp width accepted");
    check(!schedulingTimestampInterval(0, 1, 64, 0), "Zero timestamp period accepted");
    check(!schedulingTimestampInterval(0, 1, 64, std::numeric_limits<double>::quiet_NaN()),
          "NaN timestamp period accepted");
    check(!schedulingTimestampInterval(9, 8, 64, 1), "Backwards timestamp accepted");
    const auto wrap = schedulingTimestampInterval(250, 5, 8, 1000);
    check(wrap && std::abs(*wrap - 0.011) < 1e-12, "Timestamp wrap not handled");
    const auto full = schedulingGpuIntervals({10, 1010, 3010}, {true, true, true}, 64, 1000);
    check(full.prepareDurationMs && *full.prepareDurationMs == 1 && full.prepareToMainGapMs &&
              *full.prepareToMainGapMs == 2,
          "Three-query boundaries");
    const auto missingBegin =
        schedulingGpuIntervals({10, 1010, 3010}, {false, true, true}, 64, 1000);
    check(!missingBegin.prepareDurationMs && missingBegin.prepareToMainGapMs,
          "Missing prepare begin should not erase a valid gap");
    const auto missingEnd = schedulingGpuIntervals({10, 1010, 3010}, {true, false, true}, 64, 1000);
    check(!missingEnd.prepareDurationMs && !missingEnd.prepareToMainGapMs,
          "Missing shared endpoint");
    const auto missingMain =
        schedulingGpuIntervals({10, 1010, 3010}, {true, true, false}, 64, 1000);
    check(missingMain.prepareDurationMs && !missingMain.prepareToMainGapMs,
          "Missing main endpoint");
    const auto wrapBoth = schedulingGpuIntervals({250, 5, 15}, {true, true, true}, 8, 1000);
    check(wrapBoth.prepareDurationMs && std::abs(*wrapBoth.prepareDurationMs - 0.011) < 1e-12 &&
              wrapBoth.prepareToMainGapMs && std::abs(*wrapBoth.prepareToMainGapMs - 0.010) < 1e-12,
          "Three-query timestamp wrap");
    const auto reversed = schedulingGpuIntervals({100, 99, 105}, {true, true, true}, 64, 1);
    check(!reversed.prepareDurationMs && reversed.prepareToMainGapMs,
          "Backwards prepare not rejected");
    SchedulingStepSample ordinary;
    ordinary.metadata.optimizerEnabled = true;
    ordinary.gpuPrepareToMainGapMs = 0.2;
    ordinary.gpuPrepareDurationMs = 0.7;
    SchedulingStepSample special;
    special.metadata.adc = special.metadata.validation = special.metadata.resized = true;
    const auto json = schedulingDiagnosticJson("\"device\":\"test\"", {ordinary, special}, 2, true);
    check(json.find("\"status\":\"COMPLETE\"") != std::string::npos, "Complete report missing");
    check(json.find("\"ordinary\":{\"samples\":1,\"gpu_gap_samples\":1,\"mean_gap_ms\":0.2,"
                    "\"gpu_prepare_samples\":1,\"mean_prepare_duration_ms\":0.7}") !=
              std::string::npos,
          "Ordinary group polluted by special step");
    check(json.find("\"ADC\":{\"samples\":1,\"gpu_gap_samples\":0,\"mean_gap_ms\":null,"
                    "\"gpu_prepare_samples\":0,\"mean_prepare_duration_ms\":null}") !=
              std::string::npos,
          "Unavailable GPU gap reported as zero");
    check(json.find("\"gpu_driven_enabled\":false") != std::string::npos,
          "False GPU-driven admission");
    check(json.find("\"schema\":\"s4_scheduling_timing_v2\"") != std::string::npos &&
              json.find("\"overlap_measured\":false") != std::string::npos,
          "Version/scope not explicit");
    ordinary.metadata.opacityReset = true;
    check(!ordinary.metadata.ordinary(), "Reset step classified ordinary");
    ordinary.metadata.opacityReset = false;
    ordinary.metadata.optimizerCapture = true;
    check(!ordinary.metadata.ordinary(), "Capture step classified ordinary");
    ordinary.countReadMs = -1;
    reject([&] { schedulingDiagnosticJson("\"device\":\"test\"", {ordinary}, 1, true); });
    ordinary.countReadMs = 0;
    ordinary.gpuPrepareDurationMs = std::numeric_limits<double>::quiet_NaN();
    reject([&] { schedulingDiagnosticJson("\"device\":\"test\"", {ordinary}, 1, true); });
}
void runtimeMetadata() {
    auto& logger = Logger::get_instance();
    const auto old = logger.level();
    logger.set_level(LogLevel::WARN_VULKAN_3DGS);
    const auto off = schedulingRuntimeMetadataJson(false);
    logger.set_level(old);
    check(off.find("\"vulkan_validation_layers_enabled\":false") != std::string::npos &&
              off.find("\"logger_level_at_initialization\":\"WARN\"") != std::string::npos,
          "Caller disabled validation/logger state was not preserved");
    const auto unknown = schedulingRuntimeMetadataJson(std::nullopt);
    check(unknown.find("\"vulkan_validation_layers_enabled\":null") != std::string::npos &&
              unknown.find("\"vulkan_validation_state_source\":\"not_supplied\"") !=
                  std::string::npos,
          "Unknown validation was guessed false");
    check(schedulingRuntimeMetadataJson(true).find("\"vulkan_validation_layers_enabled\":true") !=
              std::string::npos,
          "Caller enabled validation state");
    check(unknown.find("\"cpp_build_config\":\"") != std::string::npos &&
              unknown.find("\"cpp_compiler\":\"") != std::string::npos &&
              unknown.find("\"cpp_ndebug_defined\":") != std::string::npos &&
              unknown.find("\"cpp_debug_macro_defined\":") != std::string::npos,
          "C++ configuration/macros not reported");
}
void reportFile() {
    const auto directory =
        std::filesystem::temp_directory_path() /
        ("v3dgs-scheduling-" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    const auto path = directory / "timing.json";
    try {
        {
            TrainingDiagnosticReport report(path);
            report.write("{\"status\":\"test\"}\n");
            reject([&] { report.write("{}"); });
        }
        reject([&] { TrainingDiagnosticReport duplicate(path); });
        std::ifstream input(path);
        std::string text;
        std::getline(input, text);
        check(text == "{\"status\":\"test\"}", "Report overwrite changed original");
    } catch (...) {
        std::filesystem::remove_all(directory);
        throw;
    }
    std::filesystem::remove_all(directory);
    check(schedulingJsonQuote("a\"\\\n") == "\"a\\\"\\\\\\u000a\"", "JSON escaping");
}
} // namespace
int main() {
    try {
        configuration();
        timestampsAndGroups();
        runtimeMetadata();
        reportFile();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "Scheduling diagnostic CPU contracts passed\n";
}
