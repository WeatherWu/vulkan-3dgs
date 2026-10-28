#include "training/app/training_smoke.hpp"
#include "training/core/training_frame_scheduler.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>

using namespace vulkan3DGS;
namespace {
void check(bool value, const char* reason) {
    if (!value) throw std::runtime_error(reason);
}
struct ReportFile {
    std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        ("v3d-smoke-" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".json");
    ~ReportFile() {
        std::error_code error;
        std::filesystem::remove(path, error);
    }
    std::string read() const {
        std::ifstream f(path);
        return {std::istreambuf_iterator<char>(f), {}};
    }
};
TrainingStepObservation observation(uint32_t step) {
    TrainingStepObservation s;
    s.iteration = step;
    s.frame = step % 2;
    s.gaussiansBefore = s.gaussiansAfter = 7;
    s.tileItems = 10;
    s.tileCapacity = 12;
    s.optimizerUpdated = true;
    s.validation.iteration = 1;
    s.validation.meanLoss = .2f;
    s.profiling.cpu[static_cast<size_t>(TrainingCpuProfileStage::Total)] = {1, 1, step};
    return s;
}
void successfulRun(const std::filesystem::path& path) {
    TrainingFrameScheduler schedule;
    TrainingScheduleConfig config;
    config.totalIterations = 30000;
    schedule.configure(config);
    TrainingSmokeRun run({3, path}, "{\"synthetic_unit_test\":true}");
    for (uint32_t n = 1; n <= 3; ++n) {
        check(!run.reached(n - 1), "Smoke stopped before boundary");
        run.observe(observation(n));
    }
    check(run.reached(3) && !schedule.isComplete(3) && schedule.totalIterations() == 30000,
          "Smoke rewrote normal schedule");
    bool rejected = false;
    try {
        run.observe(observation(4));
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    check(rejected, "Smoke accepted an extra step");
    TrainingStateFiniteCheck final;
    final.lossValues = 1;
    final.gaussianValues = 420;
    final.meanLoss = .2;
    run.finish("diagnostic_step_limit", "", 0, 3, final);
    check(run.written(), "Smoke report not written");
}
} // namespace
int main(int argc, char** argv) {
    try {
        if (argc == 3 && std::string_view(argv[1]) == "--write-fixture") {
            successfulRun(argv[2]);
            return 0;
        }
        check(argc == 1, "Unknown smoke test argument");
        check(trainingSmokeBuildJson().find("cpp_build") != std::string::npos,
              "Smoke build identity missing");
        ReportFile success;
        successfulRun(success.path);
        const auto text = success.read();
        check(text.find("SMOKE_LIMIT_REACHED_NOT_QUALITY_OR_SPEED_ADMISSION") !=
                      std::string::npos &&
                  text.find("\"completed_steps\":3") != std::string::npos &&
                  text.find("\"normal_total_iterations\":30000") != std::string::npos &&
                  text.find("\"optimizer_updated\":true") != std::string::npos,
              "Smoke success/schedule report contract");
        bool rejected = false;
        try {
            TrainingSmokeRun duplicate({3, success.path}, "{}");
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        check(rejected && success.read() == text, "Smoke report overwrite accepted");
        ReportFile partial;
        {
            TrainingSmokeRun run({3, partial.path}, "{}");
            run.observe(observation(1));
            run.finish("user_pause", "", 0, 1, std::nullopt);
        }
        check(partial.read().find("STOP_OR_PARTIAL_SMOKE") != std::string::npos,
              "Early pause claimed smoke completion");
        ReportFile errors;
        {
            TrainingSmokeRun run({1, errors.path}, "{}");
            run.observe(observation(1));
            TrainingStateFiniteCheck f;
            f.lossValues = 1;
            f.gaussianValues = 420;
            run.finish("diagnostic_step_limit", "", 1, 1, f);
        }
        check(errors.read().find("STOP_OR_PARTIAL_SMOKE") != std::string::npos,
              "Validation errors passed smoke");
        ReportFile invalid;
        {
            TrainingSmokeRun run({1, invalid.path}, "{}");
            auto s = observation(1);
            s.validation.invalidLossCount = 1;
            rejected = false;
            try {
                run.observe(s);
            } catch (const std::runtime_error&) {
                rejected = true;
            }
            check(rejected, "Invalid loss accepted");
        }
        check(invalid.read().find("aborted_before_report") != std::string::npos,
              "Aborted smoke has no partial report");
        ReportFile skipped;
        {
            TrainingSmokeRun run({3, skipped.path}, "{}");
            rejected = false;
            try {
                run.observe(observation(2));
            } catch (const std::runtime_error&) {
                rejected = true;
            }
            check(rejected, "Skipped step accepted");
        }
        for (uint32_t mode = 0; mode < 3; ++mode) {
            ReportFile morton;
            TrainingSmokeRun run({2, morton.path}, "{}");
            auto first = observation(1);
            first.mortonOrderingRequested = true;
            first.mortonReordered = mode != 1;
            first.morton.reorderCount = mode == 1 ? 0u : 1u;
            rejected = false;
            try {
                run.observe(first);
                auto second = observation(2);
                second.mortonOrderingRequested = true;
                second.mortonReordered = mode == 2;
                second.morton.reorderCount = 1;
                run.observe(second);
            } catch (const std::runtime_error&) {
                rejected = true;
            }
            check(rejected == (mode != 0), "Smoke did not verify actual Z-order event schedule");
            run.finish("unit_test", "", 0, 0, std::nullopt);
            check(morton.read().find("\"morton_requested\":true") != std::string::npos,
                  "Smoke omitted actual Z-order metadata");
        }
        std::cout << "Smoke boundaries, report reservation, interruption and validation contracts "
                     "passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
