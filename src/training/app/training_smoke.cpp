#include "training/app/training_smoke.hpp"
#include <cmath>
#include <iomanip>
#include <sstream>

namespace vulkan3DGS {
namespace {
TrainingSmokeOptions checkedOptions(TrainingSmokeOptions options) {
    options.validate();
    return options;
}
void number(std::ostream& out, double value) {
    if (std::isfinite(value))
        out << value;
    else
        out << "null";
}
void profileJson(std::ostream& out, const TrainingProfilingStats& profile) {
    out << "{\"gpu_timestamps_available\":" << std::boolalpha << profile.gpuTimestampsAvailable;
    auto timings = [&](const char* name, const auto& values) {
        out << ",\"" << name << "\":[";
        for (size_t i = 0; i < values.size(); ++i) {
            if (i) out << ',';
            out << "{\"stage\":" << i << ",\"samples\":" << values[i].sampleCount
                << ",\"last_ms\":";
            number(out, values[i].lastMs);
            out << ",\"average_ms\":";
            number(out, values[i].averageMs);
            out << '}';
        }
        out << ']';
    };
    timings("cpu", profile.cpu);
    timings("gpu", profile.gpu);
    out << '}';
}
} // namespace
std::string trainingSmokeBuildJson() {
    return "{\"cpp_build\":" + schedulingJsonQuote(TRAINING_CPP_BUILD_CONFIG) +
           ",\"compiler\":" + schedulingJsonQuote(TRAINING_CPP_COMPILER) +
           ",\"shader_compiler\":" + schedulingJsonQuote(OPTIMIZER_SHADER_COMPILER) + "}";
}
TrainingSmokeRun::TrainingSmokeRun(TrainingSmokeOptions options, std::string identityJson)
    : options_(checkedOptions(std::move(options))), identity_(std::move(identityJson)),
      report_(options_.report) {
    samples_.reserve(options_.stopIteration);
}
TrainingSmokeRun::~TrainingSmokeRun() noexcept {
    if (!report_.written()) {
        try {
            finish("aborted_before_report", "Smoke run aborted before final report", std::nullopt,
                   std::nullopt, std::nullopt);
        } catch (...) {
            std::fputs("Training smoke fallback report could not be written.\n", stderr);
        }
    }
}
void TrainingSmokeRun::observe(const TrainingStepObservation& sample) {
    if (sample.iteration != samples_.size() + 1 || reached(static_cast<uint32_t>(samples_.size())))
        throw std::runtime_error(
            "Smoke requires exactly one completed fresh training step per observation");
    samples_.push_back(sample);
    const auto& v = sample.validation;
    if (v.iteration &&
        (v.invalidLossCount || v.invalidRenderedPixelCount || v.nonFiniteGaussianCount ||
         !std::isfinite(v.meanLoss) || !std::isfinite(v.maxLoss)))
        throw std::runtime_error("Smoke detected non-finite/invalid training validation data");
    if (sample.fasterSortRequested && !sample.fasterSortActive)
        throw std::runtime_error("Smoke requested Faster sorting but the actual path fell back");
}
void TrainingSmokeRun::finish(std::string_view reason, std::string_view error,
                              std::optional<uint64_t> errors, std::optional<double> wall,
                              const std::optional<TrainingStateFiniteCheck>& finalState) {
    const bool atLimit = samples_.size() == options_.stopIteration;
    const bool pass =
        atLimit && error.empty() && errors && !*errors && finalState && finalState->passed();
    std::ostringstream out;
    out << std::setprecision(12) << std::boolalpha;
    out << "{\"schema\":\"training_smoke_v1\",\"status\":"
        << schedulingJsonQuote(pass ? "SMOKE_LIMIT_REACHED_NOT_QUALITY_OR_SPEED_ADMISSION"
                                    : "STOP_OR_PARTIAL_SMOKE")
        << ",\"stop_reason\":" << schedulingJsonQuote(reason)
        << ",\"error\":" << schedulingJsonQuote(error)
        << ",\"requested_stop_iteration\":" << options_.stopIteration
        << ",\"completed_steps\":" << samples_.size()
        << ",\"normal_total_iterations\":30000,\"normal_schedule_modified\":false,\"natural_"
           "training_complete\":false"
        << ",\"worker_loop_wall_ms\":";
    if (wall)
        number(out, *wall);
    else
        out << "null";
    out << ",\"vulkan_validation_errors_total\":";
    if (errors)
        out << *errors;
    else
        out << "null";
    out << ",\"identity\":" << identity_
        << ",\"timing_admission\":false,\"full_training_speedup\":null,\"peak_vram_measured_"
           "bytes\":null,\"final_readback_excluded_"
           "from_loop_wall\":true,\"final_finite_check\":";
    if (finalState) {
        const auto& f = *finalState;
        out << "{\"passed\":" << f.passed() << ",\"loss_values\":" << f.lossValues
            << ",\"invalid_loss_values\":" << f.invalidLossValues
            << ",\"gaussian_values\":" << f.gaussianValues
            << ",\"invalid_gaussian_values\":" << f.invalidGaussianValues
            << ",\"invalid_activated_scales\":" << f.invalidActivatedScales << ",\"mean_loss\":";
        number(out, f.meanLoss);
        out << ",\"max_loss\":";
        number(out, f.maxLoss);
        out << '}';
    } else
        out << "null";
    uint32_t adc = 0, reset = 0, resize = 0, sh = 0, active = 0, optimizer = 0;
    for (size_t i = 0; i < samples_.size(); ++i) {
        const auto& s = samples_[i];
        adc += s.densification;
        reset += s.opacityReset;
        resize += s.resized;
        active += s.fasterSortActive;
        optimizer += s.optimizerUpdated;
        if (i && s.activeSHDegree != samples_[i - 1].activeSHDegree) ++sh;
    }
    out << ",\"event_counts\":{\"adc\":" << adc << ",\"opacity_reset\":" << reset
        << ",\"tile_resize\":" << resize << ",\"sh_changes\":" << sh
        << ",\"faster_active_steps\":" << active << ",\"optimizer_updates\":" << optimizer
        << "},\"steps\":[";
    for (size_t i = 0; i < samples_.size(); ++i) {
        const auto& s = samples_[i];
        const auto& v = s.validation;
        if (i) out << ',';
        out << "{\"iteration\":" << s.iteration << ",\"frame\":" << s.frame
            << ",\"gaussians_before\":" << s.gaussiansBefore
            << ",\"gaussians_after\":" << s.gaussiansAfter << ",\"items\":" << s.tileItems
            << ",\"capacity\":" << s.tileCapacity << ",\"active_sh\":" << s.activeSHDegree
            << ",\"faster_requested\":" << s.fasterSortRequested
            << ",\"faster_active\":" << s.fasterSortActive << ",\"participating_gaussians\":";
        if (s.fasterSortActive)
            out << s.participatingGaussians;
        else
            out << "null";
        out << ",\"optimizer_updated\":" << s.optimizerUpdated << ",\"adc\":" << s.densification
            << ",\"opacity_reset\":" << s.opacityReset << ",\"tile_resized\":" << s.resized
            << ",\"ssim_mode\":" << s.ssimMode << ",\"pixel_mode\":" << s.pixelMode
            << ",\"tile_mode\":" << s.tileMode << ",\"validation_iteration\":" << v.iteration
            << ",\"validation_valid\":";
        if (v.iteration)
            out << v.valid;
        else
            out << "null";
        out << ",\"mean_loss_at_validation\":";
        if (v.iteration)
            number(out, v.meanLoss);
        else
            out << "null";
        out << ",\"invalid_loss\":" << v.invalidLossCount
            << ",\"invalid_rendered_pixels\":" << v.invalidRenderedPixelCount
            << ",\"non_finite_gaussians\":" << v.nonFiniteGaussianCount << ",\"cpu_step_ms\":";
        number(out, s.profiling.cpu[static_cast<size_t>(TrainingCpuProfileStage::Total)].lastMs);
        out << '}';
    }
    out << "],\"final_profiling\":";
    if (samples_.empty())
        out << "null";
    else
        profileJson(out, samples_.back().profiling);
    out << "}\n";
    report_.write(out.str());
}
} // namespace vulkan3DGS
