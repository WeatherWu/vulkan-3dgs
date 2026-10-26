#include "training_range_key_session.hpp"
#include "training_optimizer_r1_metrics.hpp"
#include "utils/logger.hpp"

#include <atomic>
#include <charconv>
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace range_keys;
namespace {
std::atomic<uint32_t> validationErrors{0};
VKAPI_ATTR vk::Bool32 VKAPI_CALL message(vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
    vk::DebugUtilsMessageTypeFlagsEXT, const vk::DebugUtilsMessengerCallbackDataEXT* data, void*) {
    if (severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eError) {
        ++validationErrors;
        std::cerr << "Range key validation: " << data->pMessage << '\n';
    }
    return VK_FALSE;
}
uint32_t integer(const char* value) {
    uint32_t number = 0;
    const std::string_view text(value);
    const auto result = std::from_chars(text.data(), text.data() + text.size(), number);
    check(result.ec == std::errc{} && result.ptr == text.data() + text.size(), "Invalid device index");
    return number;
}
class Context {
public:
    vk::UniqueInstance instance;
    vk::UniqueDebugUtilsMessengerEXT messenger;
    vk::PhysicalDevice physical;
    vk::PhysicalDeviceProperties properties;
    vk::UniqueDevice device;
    vk::Queue queue;
    uint32_t family = 0;
    bool validation = false, synchronization = false;
    std::string uuid;
    Context(bool requireValidation, bool noValidation, uint32_t index) {
        VULKAN_HPP_DEFAULT_DISPATCHER.init();
        const char* layer = "VK_LAYER_KHRONOS_validation";
        if (!noValidation)
            for (const auto& p : vk::enumerateInstanceLayerProperties())
                validation |= std::string_view(p.layerName.data()) == layer;
        vk::ApplicationInfo app{"private-range-keys", 1, "vulkan-3dgs", 1, VK_API_VERSION_1_2};
        vk::InstanceCreateInfo create;
        create.setPApplicationInfo(&app);
        std::vector<const char*> extensions;
        const auto feature = vk::ValidationFeatureEnableEXT::eSynchronizationValidation;
        vk::ValidationFeaturesEXT validationFeatures;
        if (validation) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
            for (const auto& p : vk::enumerateInstanceExtensionProperties(std::string(layer)))
                synchronization |= std::string_view(p.extensionName.data()) == VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME;
            if (synchronization) {
                extensions.push_back(VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME);
                validationFeatures.setEnabledValidationFeatures(feature);
                create.setPNext(&validationFeatures);
            }
            create.setPEnabledLayerNames(layer).setPEnabledExtensionNames(extensions);
        }
        check(!requireValidation || (validation && synchronization), "Required core/synchronization validation unavailable");
        instance = vk::createInstanceUnique(create);
        VULKAN_HPP_DEFAULT_DISPATCHER.init(*instance);
        if (validation) {
            vk::DebugUtilsMessengerCreateInfoEXT debug;
            debug.setMessageSeverity(vk::DebugUtilsMessageSeverityFlagBitsEXT::eError)
                .setMessageType(vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance);
            debug.pfnUserCallback = message;
            messenger = instance->createDebugUtilsMessengerEXTUnique(debug);
        }
        const auto devices = instance->enumeratePhysicalDevices();
        check(index < devices.size(), "Requested device unavailable"); physical = devices[index];
        bool push = false;
        for (const auto& p : physical.enumerateDeviceExtensionProperties())
            push |= std::string_view(p.extensionName.data()) == VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME;
        check(push, "Radix sorter requires VK_KHR_push_descriptor");
        const auto families = physical.getQueueFamilyProperties();
        while (family < families.size() && !(families[family].queueFlags & vk::QueueFlagBits::eCompute)) ++family;
        check(family < families.size() && families[family].timestampValidBits, "Compute timestamps unavailable");
        vk::PhysicalDeviceIDProperties ids;
        vk::PhysicalDeviceProperties2 props;
        props.pNext = &ids; physical.getProperties2(&props); properties = props.properties;
        std::ostringstream id;
        for (uint8_t byte : ids.deviceUUID) id << std::hex << std::setw(2) << std::setfill('0') << unsigned(byte);
        uuid = id.str();
        const float priority = 1;
        vk::DeviceQueueCreateInfo q{{}, family, 1, &priority};
        const char* extension = VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME;
        vk::PhysicalDeviceFeatures features; features.shaderInt64 = physical.getFeatures().shaderInt64;
        vk::DeviceCreateInfo deviceInfo;
        deviceInfo.setQueueCreateInfos(q).setPEnabledExtensionNames(extension).setPEnabledFeatures(&features);
        device = physical.createDeviceUnique(deviceInfo);
        VULKAN_HPP_DEFAULT_DISPATCHER.init(*device); queue = device->getQueue(family, 0);
    }
};

void evaluationJson(std::ostream& out, const optimizer_r1::Decision& d, bool target, bool absoluteGate) {
    std::string verdict = d.verdict;
    if (absoluteGate && verdict == "SHORT_GAIN_REQUIRES_NATIVE_REVIEW" && d.gainMs < .10)
        verdict = "STOP_BELOW_ABSOLUTE_GAIN_TARGET";
    if (!target) verdict = "NON_TARGET_DEVICE_NOT_ADMISSION";
    out << "{\"A1_mean_ms\":" << d.a1.mean << ",\"B_mean_ms\":" << d.b.mean
        << ",\"A2_mean_ms\":" << d.a2.mean << ",\"gain_ms\":" << d.gainMs
        << ",\"gain_percent\":" << d.gainPercent << ",\"baseline_drift_percent\":" << d.baselineDriftPercent
        << ",\"verdict\":" << schedulingJsonQuote(verdict) << '}';
}

std::string testCase(Context& context, const Inputs& input, bool timing) {
    Session session(*context.device, context.physical, context.queue, context.family, input);
    session.initialize();
    for (bool direct : {false, true, false}) {
        session.run(direct); session.verify();
        check(!validationErrors, "Range correctness core/synchronization validation failed");
    }
    std::ostringstream out;
    out.precision(10);
    out << "{\"items\":" << input.header.source.itemCount << ",\"tiles\":" << input.header.source.tileCount
        << ",\"capacity\":" << input.header.capacity
        << ",\"gpu_producer_verified\":true,\"ranges_exact_A_B_A\":true,\"readonly_tails_canaries_verified\":true"
        << ",\"direct_source\":\"" << (input.header.source.itemCount > 1 ? "GPU_SORTED_KEYS" : "ORIGINAL_KEY_BYPASS") << '"';
    if (timing) {
        std::array<std::vector<Sample>, 3> samples;
        for (size_t phase = 0; phase < 3; ++phase) {
            for (uint32_t i = 0; i < 2; ++i) session.run(phase == 1);
            for (uint32_t i = 0; i < 8; ++i) samples[phase].push_back(session.run(phase == 1));
        }
        session.verify();
        session.run(true); session.verify();
        check(!validationErrors, "Range timing validation failed");
        out << ",\"warmup_per_phase\":2,\"measured_per_phase\":8,\"phases\":[";
        std::array<std::array<double, 8>, 3> sequence{}, range{};
        for (size_t phase = 0; phase < 3; ++phase) {
            if (phase) out << ',';
            out << "{\"direct\":" << std::boolalpha << (phase == 1) << ",\"samples\":[";
            for (size_t i = 0; i < 8; ++i) {
                const auto& s = samples[phase][i];
                sequence[phase][i] = s.sequenceMs; range[phase][i] = s.rangeMs;
                if (i) out << ',';
                out << "{\"sort_gather_ranges_ms\":" << s.sequenceMs << ",\"ranges_ms\":" << s.rangeMs
                    << ",\"boundaries_ms\":" << s.boundaryMs << ",\"restore_cpu_submit_wait_ms\":" << s.restoreCpuMs
                    << ",\"cpu_submit_wait_ms\":" << s.submitWaitMs << '}';
            }
            out << "]}";
        }
        const bool target = context.properties.vendorID == 0x10DE &&
                            std::string_view(context.properties.deviceName.data()).find("RTX 3090") != std::string_view::npos;
        out << "],\"sequence_evaluation\":";
        evaluationJson(out, optimizer_r1::evaluate(sequence[0], sequence[1], sequence[2]), target, true);
        out << ",\"range_local_evaluation\":";
        evaluationJson(out, optimizer_r1::evaluate(range[0], range[1], range[2]), target, false);
    }
    out << '}';
    return out.str();
}
constexpr const char* usage = "training_range_keys_gpu_tests (--synthetic | --capture INPUT.scanline.bin --inputs INPUT.sort-input.bin) "
                             "--report NEW.json [--require-validation | --no-validation] [--timing] [--device-index N]";
} // namespace

int main(int argc, char** argv) {
    std::unique_ptr<TrainingDiagnosticReport> report;
    try {
        std::filesystem::path capturePath, inputPath, output;
        bool syntheticMode = false, timing = false, requireValidation = false, noValidation = false;
        uint32_t deviceIndex = 0;
        for (int i = 1; i < argc; ++i) {
            const std::string_view arg(argv[i]);
            if (arg == "--help") { std::cout << usage << '\n'; return 0; }
            if (arg == "--synthetic") syntheticMode = true;
            else if (arg == "--timing") timing = true;
            else if (arg == "--require-validation") requireValidation = true;
            else if (arg == "--no-validation") noValidation = true;
            else if (arg == "--capture" && i + 1 < argc) capturePath = argv[++i];
            else if (arg == "--inputs" && i + 1 < argc) inputPath = argv[++i];
            else if (arg == "--report" && i + 1 < argc) output = argv[++i];
            else if (arg == "--device-index" && i + 1 < argc) deviceIndex = integer(argv[++i]);
            else throw std::runtime_error(usage);
        }
        check(!output.empty() && !(requireValidation && noValidation), usage);
        check(syntheticMode ? (capturePath.empty() && inputPath.empty() && !timing)
                            : (!capturePath.empty() && !inputPath.empty()), usage);
        check(!timing || noValidation, "Timing requires explicit --no-validation and a separate validated run");
        output = std::filesystem::absolute(output);
        report = std::make_unique<TrainingDiagnosticReport>(output);
        Inputs input;
        if (!syntheticMode) {
            capturePath = std::filesystem::absolute(capturePath); inputPath = std::filesystem::absolute(inputPath);
            check(std::filesystem::file_size(capturePath) <= (256ull << 20), "Capture exceeds input byte budget");
            const auto source = readTrainingScanlineCapture(capturePath);
            input = readInputs(inputPath); verify(input, source);
        }
        Context context(requireValidation, noValidation, deviceIndex);
        Logger::get_instance().set_level(LogLevel::ERROR_VULKAN_3DGS);
        std::ostringstream cases;
        if (syntheticMode) {
            bool first = true;
            for (uint32_t n : {0u, 1u, 2u, 31u, 32u, 33u, 255u, 256u, 257u, 4095u, 4096u, 4097u}) {
                auto [data, source] = synthetic(n); verify(data, source);
                if (!first) cases << ',';
                first = false;
                cases << testCase(context, data, false);
            }
            auto [data, source] = synthetic(4097, true); verify(data, source);
            cases << ',' << testCase(context, data, false);
        } else cases << testCase(context, input, timing);
        context.device->waitIdle();
        check(!validationErrors, "Range validation errors");
        std::ostringstream json;
        json << "{\"schema\":\"range_sorted_keys_gpu_v1\",\"status\":\"PRIVATE_CORRECTNESS_PASSED_NOT_TRAINING_ADMISSION\""
             << ",\"device\":" << schedulingJsonQuote(context.properties.deviceName.data())
             << ",\"device_uuid\":" << schedulingJsonQuote(context.uuid)
             << ",\"vendor_id\":" << context.properties.vendorID << ",\"device_id\":" << context.properties.deviceID
             << ",\"driver_version\":" << context.properties.driverVersion
             << ",\"timestamp_valid_bits\":" << context.physical.getQueueFamilyProperties()[context.family].timestampValidBits
             << ",\"timestamp_period_ns\":" << context.properties.limits.timestampPeriod
             << ",\"device_index\":" << deviceIndex << ",\"validation_enabled\":" << std::boolalpha << context.validation
             << ",\"synchronization_validation_enabled\":" << context.synchronization
             << ",\"validation_errors\":" << validationErrors.load()
             << ",\"synthetic_inputs\":" << syntheticMode << ",\"timing_requested\":" << timing
             << ",\"cpp_build\":\"" RANGE_KEY_BUILD_CONFIG "\",\"compiler\":" << schedulingJsonQuote(RANGE_KEY_COMPILER)
             << ",\"shader_compiler\":" << schedulingJsonQuote(RANGE_KEY_SHADER_COMPILER)
             << ",\"sorter_header_sha256\":\"" RANGE_KEY_SORTER_HASH "\""
             << ",\"source_capture\":" << (syntheticMode ? "null" : schedulingJsonQuote(capturePath.string()))
             << ",\"source_capture_fnv1a64\":" << (syntheticMode ? "null" : schedulingJsonQuote(std::to_string(optimizerShaderHash(capturePath))))
             << ",\"raw_inputs\":" << (syntheticMode ? "null" : schedulingJsonQuote(inputPath.string()))
             << ",\"raw_inputs_fnv1a64\":" << (syntheticMode ? "null" : schedulingJsonQuote(std::to_string(optimizerShaderHash(inputPath))))
             << ",\"iteration\":" << (syntheticMode ? 0 : input.header.source.iteration)
             << ",\"frame\":" << (syntheticMode ? 0 : input.header.source.frame)
             << ",\"production_candidate_enabled\":false,\"training_advanced\":false,\"full_training_speedup\":null"
             << ",\"full_list_sequence_measured\":false,\"scope\":\"GPU depth sort + high-key gather + tile sort + item gather + "
                "boundary construction + range finalize. No projection/count/prefix/emit/composite/backward/Adam.\""
             << ",\"restore_policy\":\"Private seeds/work/ranges/radix scratch restored outside every measured submission; "
                "producer sorts are executed inside the measured sequence for A/B/A, not supplied by CPU. Cache-perturbed.\""
             << ",\"shader_fnv1a64\":{";
        bool first = true;
        for (const char* name : {"train_forward_tile_range_boundaries", "train_range_boundaries_sorted_keys_test",
                 "train_forward_tile_gather_high", "train_forward_tile_gather_items", "train_forward_tile_sort"}) {
            if (!first) json << ',';
            first = false;
            const auto path = std::filesystem::path(RANGE_KEY_RUNTIME_DIR) / "shaders" / (std::string(name) + ".comp.spv");
            json << schedulingJsonQuote(name) << ":\"" << optimizerShaderHash(path) << '"';
        }
        json << "},\"cases\":[" << cases.str() << "]}\n";
        report->write(json.str()); std::cout << "Private range-key comparison complete: " << output << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Private range-key comparison failed: " << error.what() << '\n';
        if (report && !report->written()) {
            try { report->write("{\"schema\":\"range_sorted_keys_gpu_v1\",\"status\":\"STOP\",\"error\":" +
                schedulingJsonQuote(error.what()) + ",\"validation_errors\":" + std::to_string(validationErrors.load()) +
                ",\"full_training_speedup\":null}\n"); } catch (...) {}
        }
        return 1;
    }
    return 0;
}
