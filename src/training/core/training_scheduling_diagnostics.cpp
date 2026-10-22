#include "training/core/training_scheduling_diagnostics.hpp"
#include "utils/logger.hpp"

#include <array>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace vulkan3DGS {
namespace {
uint32_t number(const char* text, uint32_t fallback, uint32_t minimum, uint32_t maximum) {
    if (!text || !*text) return fallback;
    const std::string_view value(text);
    uint32_t result = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || result < minimum ||
        result > maximum)
        throw std::runtime_error("S4-3 invalid diagnostic sample/start configuration");
    return result;
}
bool present(const char* value) {
    return value && *value;
}
const char* currentLoggerLevel() noexcept {
    switch (Logger::get_instance().level()) {
        case LogLevel::DEBUG_VULKAN_3DGS:
            return "DEBUG";
        case LogLevel::INFO_VULKAN_3DGS:
            return "INFO";
        case LogLevel::WARN_VULKAN_3DGS:
            return "WARN";
        case LogLevel::ERROR_VULKAN_3DGS:
            return "ERROR";
    }
    return "UNKNOWN";
}
std::string environment(const char* name) {
#ifdef _WIN32
    char* value = nullptr;
    size_t length = 0;
    if (_dupenv_s(&value, &length, name) != 0)
        throw std::runtime_error("S4-3 could not read environment");
    std::unique_ptr<char, decltype(&std::free)> owner(value, &std::free);
    return value ? value : "";
#else
    const char* value = std::getenv(name);
    return value ? value : "";
#endif
}
void optional(std::ostream& out, std::optional<double> value) {
    if (value)
        out << *value;
    else
        out << "null";
}
} // namespace

TrainingDiagnosticReport::TrainingDiagnosticReport(const std::filesystem::path& path) {
    if (path.empty()) throw std::runtime_error("S4-3 output path is empty");
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    FILE* raw = nullptr;
#ifdef _WIN32
    _wfopen_s(&raw, path.c_str(), L"wbx");
#else
    raw = std::fopen(path.c_str(), "wbx");
#endif
    if (!raw) throw std::runtime_error("S4-3 cannot exclusively create report: " + path.string());
    file_.reset(raw);
}
void TrainingDiagnosticReport::write(std::string_view json) {
    if (written_) throw std::runtime_error("S4-3 report already committed");
    if (std::fwrite(json.data(), 1, json.size(), file_.get()) != json.size() ||
        std::fflush(file_.get()) != 0)
        throw std::runtime_error("S4-3 report write failed; partial file is not a valid result");
    written_ = true;
    file_.reset();
}
std::string schedulingJsonQuote(std::string_view value) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\')
            out << '\\' << char(c);
        else if (c < 32)
            out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c);
        else
            out << char(c);
    }
    out << '"';
    return out.str();
}
std::optional<SchedulingDiagnosticConfig>
schedulingDiagnosticConfig(const char* output, const char* samples, const char* startIteration) {
    if (!present(output)) {
        if (present(samples) || present(startIteration))
            throw std::runtime_error("S4-3 sample/start options require S4 scheduling report path");
        return {};
    }
    return SchedulingDiagnosticConfig{std::filesystem::absolute(output),
                                      number(samples, 128, 1, 2048),
                                      number(startIteration, 0, 0, UINT32_MAX)};
}
std::optional<double> schedulingTimestampInterval(uint64_t begin, uint64_t end, uint32_t validBits,
                                                  double periodNs) {
    if (!validBits || validBits > 64 || !std::isfinite(periodNs) || periodNs <= 0) return {};
    const uint64_t mask = validBits == 64 ? UINT64_MAX : (uint64_t(1) << validBits) - 1;
    const uint64_t ticks = (end - begin) & mask;
    // Reject ambiguous backwards/over-half-period readings rather than inventing a gap.
    if (ticks > mask / 2) return {};
    const double ms = double(ticks) * periodNs / 1e6;
    return std::isfinite(ms) ? std::optional(ms) : std::nullopt;
}
SchedulingGpuIntervals schedulingGpuIntervals(const std::array<uint64_t, 3>& ticks,
                                              const std::array<bool, 3>& available,
                                              uint32_t validBits, double periodNs) {
    SchedulingGpuIntervals result;
    if (available[0] && available[1])
        result.prepareDurationMs =
            schedulingTimestampInterval(ticks[0], ticks[1], validBits, periodNs);
    if (available[1] && available[2])
        result.prepareToMainGapMs =
            schedulingTimestampInterval(ticks[1], ticks[2], validBits, periodNs);
    return result;
}
std::string schedulingRuntimeMetadataJson(std::optional<bool> vulkanValidationLayersEnabled) {
    std::ostringstream out;
    out << "\"cpp_build_config\":" << schedulingJsonQuote(TRAINING_CPP_BUILD_CONFIG)
        << ",\"cpp_compiler\":" << schedulingJsonQuote(TRAINING_CPP_COMPILER)
        << ",\"cpp_ndebug_defined\":"
#ifdef NDEBUG
        << "true"
#else
        << "false"
#endif
        << ",\"cpp_debug_macro_defined\":"
#ifdef _DEBUG
        << "true"
#else
        << "false"
#endif
        << ",\"logger_level_at_initialization\":" << schedulingJsonQuote(currentLoggerLevel())
        << ",\"vulkan_validation_layers_enabled\":";
    if (vulkanValidationLayersEnabled.has_value())
        out << std::boolalpha << *vulkanValidationLayersEnabled;
    else
        out << "null";
    out << ",\"vulkan_validation_state_source\":"
        << schedulingJsonQuote(vulkanValidationLayersEnabled.has_value()
                                   ? "caller_instance_configuration_after_availability_check"
                                   : "not_supplied")
        << ",\"vulkan_validation_state_limit\":\"does not introspect externally forced "
           "loader/layer overrides\"";
    return out.str();
}
std::string schedulingDiagnosticJson(std::string_view identity,
                                     const std::vector<SchedulingStepSample>& samples,
                                     uint32_t requested, bool complete) {
    std::ostringstream out;
    out << std::setprecision(10) << '{' << identity
        << ",\"schema\":\"s4_scheduling_timing_v2\",\"status\":\""
        << (complete ? "COMPLETE" : "PARTIAL") << "\",\"requested_samples\":" << requested
        << ",\"collected_samples\":" << samples.size()
        << ",\"gpu_driven_enabled\":false,\"extra_gpu_waits\":0,\"speedup\":null"
        << ",\"overlap_measured\":false,\"diagnostic_timestamp_queries_per_step\":3"
        << ",\"prepare_duration_interpretation\":\"same-queue prepare begin to end including "
           "original work, barriers, profiling and count readback copy; NOT CPU fence wait\""
        << ",\"validation_field_interpretation\":\"per-step validation is training numeric "
           "checking, "
           "NOT Vulkan validation layers; no validation or logger settings changed by probe\""
        << ",\"gap_interpretation\":\"instrumented same-queue timestamp interval after prepare "
           "readback copy and before main commands; opportunity upper bound, NOT pure GPU idle "
           "or guaranteed savings; may include other submissions/upload dependencies\""
        << ",\"report_write_policy\":\"bounded in memory; one final write outside GPU intervals; "
           "final write can affect CPU Total step; PARTIAL is not admission\""
        << ",\"samples\":[";
    for (size_t i = 0; i < samples.size(); ++i) {
        if (i) out << ',';
        const auto& s = samples[i];
        const auto& m = s.metadata;
        const std::array times{
            s.prepareSubmitCallMs, s.prepareFenceWaitMs, s.countReadMs,
            s.capacityCheckMs,     s.resizeMs,           s.postFenceToMainSubmitMs,
            s.mainRecordMs,        s.mainSubmitCallMs};
        for (double t : times)
            if (!std::isfinite(t) || t < 0) throw std::runtime_error("S4-3 invalid CPU timing");
        if (s.gpuPrepareToMainGapMs &&
            (!std::isfinite(*s.gpuPrepareToMainGapMs) || *s.gpuPrepareToMainGapMs < 0))
            throw std::runtime_error("S4-3 invalid GPU gap");
        if (s.gpuPrepareDurationMs &&
            (!std::isfinite(*s.gpuPrepareDurationMs) || *s.gpuPrepareDurationMs < 0))
            throw std::runtime_error("S4-3 invalid GPU prepare duration");
        out << "{\"iteration\":" << m.iteration << ",\"frame\":" << m.frame
            << ",\"gaussians\":" << m.gaussians << ",\"tile_items\":" << m.tileItems
            << ",\"tile_item_capacity\":" << m.capacity << ",\"extent\":[" << m.width << ','
            << m.height << "],\"tile_size\":" << m.tileSize << ",\"SH_degree\":" << m.activeSHDegree
            << ",\"ssim_mode\":" << m.ssimMode << ",\"pixel_mode\":" << m.pixelMode
            << ",\"composite_mode\":" << m.compositeMode
            << ",\"upload_wait_value\":" << m.uploadWaitValue << std::boolalpha
            << ",\"ordinary\":" << m.ordinary() << ",\"ADC\":" << m.adc
            << ",\"validation\":" << m.validation << ",\"opacity_reset\":" << m.opacityReset
            << ",\"resized\":" << m.resized << ",\"optimizer_enabled\":" << m.optimizerEnabled
            << ",\"optimizer_capture\":" << m.optimizerCapture
            << ",\"prepare_submit_call_ms\":" << s.prepareSubmitCallMs
            << ",\"prepare_fence_wait_ms\":" << s.prepareFenceWaitMs
            << ",\"count_read_ms\":" << s.countReadMs
            << ",\"capacity_check_ms\":" << s.capacityCheckMs
            << ",\"resize_ms_nested_in_capacity_check\":" << s.resizeMs
            << ",\"post_fence_to_main_submit_ms\":" << s.postFenceToMainSubmitMs
            << ",\"main_record_ms_nested_in_post_fence\":" << s.mainRecordMs
            << ",\"main_submit_call_ms\":" << s.mainSubmitCallMs
            << ",\"gpu_prepare_to_main_gap_ms\":";
        optional(out, s.gpuPrepareToMainGapMs);
        out << ",\"gpu_prepare_duration_ms\":";
        optional(out, s.gpuPrepareDurationMs);
        out << ",\"logger_level\":";
        if (s.loggerLevel.empty())
            out << "null";
        else
            out << schedulingJsonQuote(s.loggerLevel);
        out << '}';
    }
    out << "],\"category_groups_overlap\":true,\"groups\":{";
    for (uint32_t group = 0; group < 6; ++group) {
        if (group) out << ',';
        constexpr std::array names{"ordinary", "ADC",           "validation",
                                   "resize",   "opacity_reset", "all"};
        uint32_t count = 0, measured = 0, prepareMeasured = 0;
        double sum = 0, prepareSum = 0;
        for (const auto& s : samples) {
            const auto& m = s.metadata;
            const std::array selected{m.ordinary(), m.adc,          m.validation,
                                      m.resized,    m.opacityReset, true};
            if (!selected[group]) continue;
            ++count;
            if (s.gpuPrepareToMainGapMs) {
                ++measured;
                sum += *s.gpuPrepareToMainGapMs;
            }
            if (s.gpuPrepareDurationMs) {
                ++prepareMeasured;
                prepareSum += *s.gpuPrepareDurationMs;
            }
        }
        out << '"' << names[group] << "\":{\"samples\":" << count
            << ",\"gpu_gap_samples\":" << measured << ",\"mean_gap_ms\":";
        optional(out, measured ? std::optional(sum / measured) : std::nullopt);
        out << ",\"gpu_prepare_samples\":" << prepareMeasured << ",\"mean_prepare_duration_ms\":";
        optional(out, prepareMeasured ? std::optional(prepareSum / prepareMeasured) : std::nullopt);
        out << '}';
    }
    out << "}}\n";
    return out.str();
}
std::unique_ptr<TrainingSchedulingDiagnostics>
TrainingSchedulingDiagnostics::fromEnvironment(vk::Device device, vk::PhysicalDevice physical,
                                               uint32_t family,
                                               std::optional<bool> vulkanValidationLayersEnabled) {
    const auto path = environment("VULKAN_3DGS_S4_SCHEDULING_REPORT");
    const auto count = environment("VULKAN_3DGS_S4_SCHEDULING_SAMPLES");
    const auto start = environment("VULKAN_3DGS_S4_SCHEDULING_START");
    auto config = schedulingDiagnosticConfig(path.c_str(), count.c_str(), start.c_str());
    if (!config) return {};
    return std::make_unique<TrainingSchedulingDiagnostics>(device, physical, family, *config,
                                                           vulkanValidationLayersEnabled);
}
TrainingSchedulingDiagnostics::TrainingSchedulingDiagnostics(
    vk::Device device, vk::PhysicalDevice physical, uint32_t family,
    SchedulingDiagnosticConfig config, std::optional<bool> vulkanValidationLayersEnabled)
    : device_(device), config_(std::move(config)), report_(config_.output) {
    if (!config_.samples || config_.samples > 2048)
        throw std::runtime_error("S4-3 invalid bounded sample count");
    const auto families = physical.getQueueFamilyProperties();
    if (family >= families.size()) throw std::runtime_error("S4-3 invalid compute family");
    validBits_ = families[family].timestampValidBits;
    const auto properties = physical.getProperties();
    periodNs_ = properties.limits.timestampPeriod;
    if (validBits_ && periodNs_ > 0)
        queries_ = device.createQueryPoolUnique({{}, vk::QueryType::eTimestamp, 3});
    vk::PhysicalDeviceIDProperties ids;
    vk::PhysicalDeviceProperties2 props;
    props.pNext = &ids;
    physical.getProperties2(&props);
    std::ostringstream uuid;
    for (uint8_t byte : ids.deviceUUID)
        uuid << std::hex << std::setw(2) << std::setfill('0') << unsigned(byte);
    identity_ = "\"device\":" + schedulingJsonQuote(properties.deviceName.data()) +
                ",\"device_uuid\":" + schedulingJsonQuote(uuid.str()) +
                ",\"driver_version\":" + std::to_string(properties.driverVersion) +
                ",\"api_version\":" + std::to_string(properties.apiVersion) +
                ",\"queue_family\":" + std::to_string(family) +
                ",\"timestamp_valid_bits\":" + std::to_string(validBits_) +
                ",\"timestamp_period_ns\":" + std::to_string(periodNs_) +
                ",\"training_shader_compiler\":" + schedulingJsonQuote(OPTIMIZER_SHADER_COMPILER) +
                ",\"start_iteration\":" + std::to_string(config_.startIteration) +
                ",\"synthetic\":" + (config_.synthetic ? "true" : "false") + "," +
                schedulingRuntimeMetadataJson(vulkanValidationLayersEnabled);
    samples_.reserve(config_.samples);
    LOG_INFO("S4-3 scheduling diagnostics armed: {} samples from iteration {}, output {}. No "
             "scheduling change.",
             config_.samples, config_.startIteration, config_.output.string());
}
TrainingSchedulingDiagnostics::~TrainingSchedulingDiagnostics() noexcept {
    if (!report_.written() && !failed_) {
        try {
            report_.write(schedulingDiagnosticJson(identity_, samples_, config_.samples, false));
        } catch (const std::exception& error) {
            LOG_ERROR("S4-3 partial report: {}", error.what());
        }
    }
}
bool TrainingSchedulingDiagnostics::wantsStep(
    const SchedulingStepMetadata& metadata) const noexcept {
    return !failed_ && !report_.written() && !metadata.fixedBenchmark &&
           metadata.iteration >= config_.startIteration && samples_.size() < config_.samples;
}
void TrainingSchedulingDiagnostics::recordPrepareBegin(vk::CommandBuffer command) {
    if (!queries_) return;
    command.resetQueryPool(*queries_, 0, 3);
    command.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands, *queries_, 0);
}
void TrainingSchedulingDiagnostics::recordPrepareEnd(vk::CommandBuffer command) {
    if (queries_) command.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands, *queries_, 1);
}
void TrainingSchedulingDiagnostics::recordMainBegin(vk::CommandBuffer command) {
    if (queries_) command.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands, *queries_, 2);
}
void TrainingSchedulingDiagnostics::finishStep(SchedulingStepSample sample) noexcept {
    try {
        if (queries_) {
            struct Value {
                uint64_t ticks, available;
            };
            std::array<Value, 3> values{};
            const auto status = device_.getQueryPoolResults(
                *queries_, 0, 3, sizeof(values), values.data(), sizeof(Value),
                vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWithAvailability);
            if (status == vk::Result::eSuccess || status == vk::Result::eNotReady) {
                const auto intervals = schedulingGpuIntervals(
                    {values[0].ticks, values[1].ticks, values[2].ticks},
                    {values[0].available != 0, values[1].available != 0, values[2].available != 0},
                    validBits_, periodNs_);
                sample.gpuPrepareDurationMs = intervals.prepareDurationMs;
                sample.gpuPrepareToMainGapMs = intervals.prepareToMainGapMs;
            }
        }
        sample.loggerLevel = currentLoggerLevel();
        samples_.push_back(sample);
        if (samples_.size() == config_.samples) {
            report_.write(schedulingDiagnosticJson(identity_, samples_, config_.samples, true));
            LOG_INFO(
                "S4-3 scheduling diagnostic report complete: {}. Training continues unchanged.",
                config_.output.string());
        }
    } catch (const std::exception& error) {
        // Adam/main already completed. Diagnostic IO must not prevent normal iteration advance.
        failed_ = true;
        LOG_ERROR("S4-3 diagnostics disabled: {}. Report may be partial; training state was not "
                  "rolled back.",
                  error.what());
    }
}
} // namespace vulkan3DGS
