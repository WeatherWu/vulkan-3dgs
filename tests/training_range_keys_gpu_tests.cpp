#include "training_range_key_session.hpp"
#include "training_full_sort_session.hpp"
#include "training_sort_downstream_session.hpp"
#include "training_faster_sort_integration.hpp"
#include "training_optimizer_r1_metrics.hpp"
#include "utils/logger.hpp"

#include <atomic>
#include <charconv>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <optional>

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
    bool validation = false, synchronization = false, atomicFloatEnabled = false;
    std::string uuid;
    Context(bool requireValidation, bool noValidation, uint32_t index,
            bool downstream = false, bool portable = false) {
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
        std::vector<const char*> deviceExtensions{VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME};
        vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT atomic;
        if (downstream) {
            bool available = false;
            for (const auto& extension : physical.enumerateDeviceExtensionProperties())
                available |= std::string_view(extension.extensionName.data()) == VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME;
            check(available, "Downstream requires VK_EXT_shader_atomic_float");
            vk::PhysicalDeviceFeatures2 supported; supported.pNext = &atomic;
            physical.getFeatures2(&supported);
            check(atomic.shaderBufferFloat32AtomicAdd, "Downstream requires buffer float32 atomic add");
            atomic = vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT{}; atomic.shaderBufferFloat32AtomicAdd = true;
            check(portable || full_sort::supportsDownstreamTensor(physical),
                  "Downstream Tensor requires native subgroup32 and original capability gate; no automatic fallback");
            deviceExtensions.push_back(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME); atomicFloatEnabled = true;
        }
        vk::PhysicalDeviceFeatures features; features.shaderInt64 = physical.getFeatures().shaderInt64;
        vk::DeviceCreateInfo deviceInfo;
        deviceInfo.setQueueCreateInfos(q).setPEnabledExtensionNames(deviceExtensions).setPEnabledFeatures(&features);
        if (downstream) deviceInfo.pNext = &atomic;
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
        std::array<std::vector<range_keys::Sample>, 3> samples;
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
void fullSampleJson(std::ostream& out, const full_sort::Sample& sample) {
    out << "{\"gpu_active_ms\":" << sample.activeMs << ",\"gpu_span_ms\":" << sample.spanMs
        << ",\"prepare_ms\":" << sample.prepareMs << ",\"main_ms\":" << sample.mainMs
        << ",\"ranges_ms\":" << sample.rangeMs << ",\"cpu_sequence_ms\":" << sample.cpuMs
        << ",\"restore_cpu_submit_wait_ms\":" << sample.restoreCpuMs << '}';
}
std::string testFullCase(Context& context, const Inputs& input, bool timing) {
    full_sort::verifyChainOracle(input);
    full_sort::Session session(*context.device, context.physical, context.queue, context.family, input);
    session.initialize();
    for (bool candidate : {false, true, false}) {
        session.run(candidate); session.verify();
        check(!validationErrors, "Full sort core/synchronization validation failed");
    }
    std::ostringstream out; out.precision(10);
    out << "{\"gaussians\":" << input.header.source.gaussianCount << ",\"participating_gaussians\":" << session.participatingCount()
        << ",\"items\":" << input.header.source.itemCount << ",\"tiles\":" << input.header.source.tileCount
        << ",\"capacity\":" << input.header.capacity << ",\"tile_effective_bits\":" << full_sort::tileBits(input.header.source.tileCount)
        << ",\"tile_radix_passes\":" << full_sort::tileBits(input.header.source.tileCount) / 8
        << ",\"exact_list_ranges_intermediates_A_B_A\":true,\"readonly_tails_canaries_verified\":true";
    if (timing) {
        constexpr uint32_t preconditioningPairs = 16;
        std::array<full_sort::Sample, preconditioningPairs * 2> preconditioning{};
        for (uint32_t pair = 0; pair < preconditioningPairs; ++pair) {
            preconditioning[pair * 2] = session.run(false);
            preconditioning[pair * 2 + 1] = session.run(true);
        }
        std::array<std::array<full_sort::Sample, 2>, 3> phaseWarmups{};
        std::array<std::vector<full_sort::Sample>, 3> samples;
        for (size_t phase = 0; phase < 3; ++phase) {
            for (auto& warmup : phaseWarmups[phase]) warmup = session.run(phase == 1);
            for (uint32_t i = 0; i < 8; ++i) samples[phase].push_back(session.run(phase == 1));
        }
        session.verify(); session.run(true); session.verify();
        check(!validationErrors, "Full sort timing validation errors");
        std::array<std::array<double, 8>, 3> active{}, span{}, cpu{};
        out << ",\"timing_protocol\":\"fixed_preconditioning_32_aba_2w8_v1\""
            << ",\"preconditioning_pairs\":" << preconditioningPairs << ",\"preconditioning_samples\":[";
        for (size_t i = 0; i < preconditioning.size(); ++i) {
            if (i) out << ',';
            out << "{\"candidate\":" << std::boolalpha << bool(i & 1u) << ",\"sample\":";
            fullSampleJson(out, preconditioning[i]); out << '}';
        }
        out << "],\"warmup_per_phase\":2,\"measured_per_phase\":8,\"phases\":[";
        for (size_t phase = 0; phase < 3; ++phase) {
            if (phase) out << ',';
            out << "{\"candidate\":" << std::boolalpha << (phase == 1) << ",\"warmup_samples\":[";
            for (size_t i = 0; i < phaseWarmups[phase].size(); ++i) {
                if (i) out << ',';
                fullSampleJson(out, phaseWarmups[phase][i]);
            }
            out << "],\"samples\":[";
            for (size_t i = 0; i < 8; ++i) {
                const auto& s = samples[phase][i];
                active[phase][i] = s.activeMs; span[phase][i] = s.spanMs; cpu[phase][i] = s.cpuMs;
                if (i) out << ',';
                fullSampleJson(out, s);
            }
            out << "]}";
        }
        const bool target = context.properties.vendorID == 0x10DE &&
            std::string_view(context.properties.deviceName.data()).find("RTX 3090") != std::string_view::npos;
        out << "],\"gpu_active_evaluation\":";
        evaluationJson(out, optimizer_r1::evaluate(active[0], active[1], active[2]), target, true);
        out << ",\"gpu_span_evaluation\":";
        evaluationJson(out, optimizer_r1::evaluate(span[0], span[1], span[2]), target, true);
        out << ",\"cpu_sequence_evaluation\":";
        evaluationJson(out, optimizer_r1::evaluate(cpu[0], cpu[1], cpu[2]), target, true);
    }
    out << '}'; return out.str();
}
std::pair<std::string, bool> testDownstreamCase(Context& context, const Inputs& input,
    const TrainingScanlineCapture& source, bool synthetic, bool portable) {
    full_sort::verifyChainOracle(input);
    full_sort::Session producer(*context.device, context.physical, context.queue, context.family, input);
    producer.initialize();
    full_sort::DownstreamSession consumer(*context.device, context.physical, context.queue, context.family,
                                         producer.outputView(), source, portable);
    std::array<full_sort::DownstreamSnapshot, 3> snapshots;
    std::array<uint64_t, 3> captureBlendDifferent{};
    for (size_t i = 0; i < snapshots.size(); ++i) {
        producer.runWithConsumers(i == 1, [&](auto command, const auto& view) { consumer.record(command, view); });
        producer.verify(); snapshots[i] = consumer.snapshot();
        check(!validationErrors, "Downstream core/synchronization validation failed");
        if (!synthetic) {
            check(source.blends.size() == snapshots[i].blends.size(), "Captured blend dimensions mismatch");
            for (size_t p = 0; p < source.blends.size(); ++p)
                captureBlendDifferent[i] += std::memcmp(&source.blends[p], &snapshots[i].blends[p], 16) != 0;
        }
    }
    const auto ab = full_sort::compareDownstream(snapshots[0], snapshots[1]);
    const auto repeat = full_sort::compareDownstream(snapshots[0], snapshots[2]);
    const auto ba = full_sort::compareDownstream(snapshots[2], snapshots[1]);
    const bool passed = ab.passed() && repeat.passed() && ba.passed() &&
        std::all_of(captureBlendDifferent.begin(), captureBlendDifferent.end(), [](auto n) { return n == 0; });
    std::ostringstream out; out.precision(10);
    out << "{\"passed\":" << std::boolalpha << passed << ",\"gaussians\":" << source.header.gaussianCount
        << ",\"participating_gaussians\":" << producer.participatingCount() << ",\"items\":" << source.header.itemCount
        << ",\"width\":" << source.header.width << ",\"height\":" << source.header.height
        << ",\"list_intermediates_readonly_guards_verified\":true,\"consumer_guards_and_readonly_pixel_gradients_verified\":true"
        << ",\"producer_consumer_same_main_command_buffer\":true,\"cpu_sorted_input_uploaded\":false"
        << ",\"capture_blend_checked\":" << !synthetic << ",\"capture_blend_differing_pixels_A1_B_A2\":["
        << captureBlendDifferent[0] << ',' << captureBlendDifferent[1] << ',' << captureBlendDifferent[2] << ']'
        << ",\"producer_logical_buffer_bytes\":" << producer.logicalBufferBytes()
        << ",\"consumer_logical_buffer_bytes\":" << consumer.logicalBufferBytes()
        << ",\"peak_vram_measured_bytes\":null,\"A1_B\":";
    full_sort::downstreamComparisonJson(out, ab); out << ",\"A1_A2_repeat\":";
    full_sort::downstreamComparisonJson(out, repeat); out << ",\"A2_B\":";
    full_sort::downstreamComparisonJson(out, ba); out << '}';
    return {out.str(), passed};
}
constexpr const char* usage = "training_range_keys_gpu_tests (--synthetic | --capture INPUT.scanline.bin --inputs INPUT.sort-input.bin) "
                             "--report NEW.json [--full-sort-chain] [--require-validation | --no-validation] [--timing] [--device-index N] "
                             "[--downstream [--portable-reference (synthetic only)]] [--integrated-sort]";
} // namespace

int main(int argc, char** argv) {
    std::unique_ptr<TrainingDiagnosticReport> report;
    bool fullChain = false;
    bool downstream = false, portableReference = false;
    bool integrated = false;
    std::string failureContext;
    try {
        std::filesystem::path capturePath, inputPath, output;
        bool syntheticMode = false, timing = false, requireValidation = false, noValidation = false;
        uint32_t deviceIndex = 0;
        for (int i = 1; i < argc; ++i) {
            const std::string_view arg(argv[i]);
            if (arg == "--help") { std::cout << usage << '\n'; return 0; }
            if (arg == "--synthetic") syntheticMode = true;
            else if (arg == "--full-sort-chain") fullChain = true;
            else if (arg == "--downstream") downstream = true;
            else if (arg == "--portable-reference") portableReference = true;
            else if (arg == "--integrated-sort") integrated = true;
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
        check(!downstream || (fullChain && requireValidation && !timing && !noValidation),
              "Downstream requires --full-sort-chain --require-validation without timing");
        check(!portableReference || (downstream && syntheticMode), "Portable reference is restricted to explicit downstream synthetic fixtures");
        check(!integrated || (fullChain && requireValidation && !timing && !noValidation && !downstream),
              "Integrated sort requires --full-sort-chain --require-validation without downstream/timing");
        check(syntheticMode ? (capturePath.empty() && inputPath.empty() && !timing)
                            : (!capturePath.empty() && !inputPath.empty()), usage);
        check(!timing || noValidation, "Timing requires explicit --no-validation and a separate validated run");
        output = std::filesystem::absolute(output);
        report = std::make_unique<TrainingDiagnosticReport>(output);
        Inputs input;
        std::optional<TrainingScanlineCapture> downstreamSource;
        if (!syntheticMode) {
            capturePath = std::filesystem::absolute(capturePath); inputPath = std::filesystem::absolute(inputPath);
            check(std::filesystem::file_size(capturePath) <= (256ull << 20), "Capture exceeds input byte budget");
            auto source = readTrainingScanlineCapture(capturePath);
            input = readInputs(inputPath); verify(input, source);
            if (downstream || integrated) downstreamSource.emplace(std::move(source));
        }
        Context context(requireValidation, noValidation, deviceIndex, downstream, portableReference);
        failureContext = ",\"device\":" + schedulingJsonQuote(context.properties.deviceName.data()) +
            ",\"device_uuid\":" + schedulingJsonQuote(context.uuid) +
            ",\"driver_version\":" + std::to_string(context.properties.driverVersion) +
            ",\"validation_enabled\":" + (context.validation ? "true" : "false") +
            ",\"synchronization_validation_enabled\":" + (context.synchronization ? "true" : "false");
        Logger::get_instance().set_level(LogLevel::ERROR_VULKAN_3DGS);
        std::ostringstream cases;
        bool downstreamPassed = true;
        if (integrated) {
            const auto result = runFasterSortIntegration(*context.device,context.physical,context.queue,context.family,
                syntheticMode ? nullptr : &*downstreamSource);
            check(!validationErrors,"Integrated sort core/synchronization validation failed");
            cases << result.substr(1,result.size()-2);
        } else if (downstream) {
            bool first = true;
            auto run = [&](const Inputs& data, const TrainingScanlineCapture& source) {
                verify(data, source);
                const auto [result, passed] = testDownstreamCase(context, data, source, syntheticMode, portableReference);
                if (!first) cases << ',';
                first = false; cases << result; downstreamPassed &= passed;
            };
            if (syntheticMode) {
                for (uint32_t n : {0u, 1u, 2u, 33u}) { auto [data, source] = full_sort::downstreamSynthetic(n, false); run(data, source); }
                for (uint32_t n : {33u, 257u}) { auto [data, source] = full_sort::downstreamSynthetic(n, true); run(data, source); }
            } else run(input, *downstreamSource);
        } else if (syntheticMode && fullChain) {
            bool first = true;
            auto run = [&](auto pair) {
                auto& [data, source] = pair; verify(data, source);
                if (!first) cases << ',';
                first = false; cases << testFullCase(context, data, false);
            };
            for (uint32_t n : {0u, 1u, 2u, 31u, 32u, 33u, 255u, 256u, 257u, 4095u, 4096u, 4097u, 65537u})
                run(full_sort::synthetic(n));
            run(full_sort::synthetic(33, true));
            run(full_sort::synthetic(33, false, 2, true));
            run(full_sort::synthetic(2, false, 65536));
            run(full_sort::synthetic(2, false, 65537));
        } else if (syntheticMode) {
            bool first = true;
            for (uint32_t n : {0u, 1u, 2u, 31u, 32u, 33u, 255u, 256u, 257u, 4095u, 4096u, 4097u}) {
                auto [data, source] = synthetic(n); verify(data, source);
                if (!first) cases << ',';
                first = false;
                cases << testCase(context, data, false);
            }
            auto [data, source] = synthetic(4097, true); verify(data, source);
            cases << ',' << testCase(context, data, false);
        } else cases << (fullChain ? testFullCase(context, input, timing) : testCase(context, input, timing));
        context.device->waitIdle();
        check(!validationErrors, "Range validation errors");
        std::ostringstream json;
        json << "{\"schema\":" << schedulingJsonQuote(integrated ? "faster_sort_integration_gpu_v1" : downstream ? "faster_sort_downstream_gpu_v1" : fullChain ? "faster_sort_chain_gpu_v1" : "range_sorted_keys_gpu_v1")
             << ",\"status\":" << schedulingJsonQuote(integrated ? "PRIVATE_RENDERER_INTEGRATION_PASSED_NOT_LIVE_TRAINING_ADMISSION" : downstream ? (downstreamPassed ? (portableReference ?
                 "PRIVATE_PORTABLE_FIXTURE_PASSED_NOT_TENSOR_ADMISSION" : "PRIVATE_DOWNSTREAM_PASSED_NOT_TRAINING_ADMISSION") :
                 "STOP_DOWNSTREAM_NUMERICAL_MISMATCH") : "PRIVATE_CORRECTNESS_PASSED_NOT_TRAINING_ADMISSION")
             << ",\"device\":" << schedulingJsonQuote(context.properties.deviceName.data())
             << ",\"device_uuid\":" << schedulingJsonQuote(context.uuid)
             << ",\"vendor_id\":" << context.properties.vendorID << ",\"device_id\":" << context.properties.deviceID
             << ",\"driver_version\":" << context.properties.driverVersion
             << ",\"timestamp_valid_bits\":" << context.physical.getQueueFamilyProperties()[context.family].timestampValidBits
             << ",\"timestamp_period_ns\":" << context.properties.limits.timestampPeriod
             << ",\"min_storage_buffer_offset_alignment\":" << context.properties.limits.minStorageBufferOffsetAlignment
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
             << (integrated ? ",\"full_list_sequence_measured\":false,\"scope\":\"Actual GaussianForwardRenderer original/candidate/original list and composite. Private TrainingBuffers; optional buffer adoption exercise. No worker, loss, backward or Adam.\""
                              ",\"renderer_candidate_exercised\":true,\"live_training_exercised\":false"
                          : downstream ? ",\"full_list_sequence_measured\":false,\"scope\":\"Correctness-only GPU list production -> original composite -> original backward with fixed captured pixelGrad. No loss/projection backward/Adam/ADC.\""
                              ",\"candidate_bundle\":\"Stable compact + Nv depth sort + reordered prefix/emit + effective-bit tile KV sort + direct Gaussian IDs + direct-key ranges\""
                              ",\"key_storage_bits\":32,\"physical_uint16_storage\":false,\"stable_compaction\":true"
                              ",\"downstream_timing_measured\":false,\"fixed_pixel_gradient\":true,\"loss_recomputed\":false,\"optimizer_updated\":false"
                          : fullChain ? ",\"full_list_sequence_measured\":true,\"scope\":\"GPU count + prefixes + stable compaction + depth sort + emit + tile sort + ranges, including original prepare count readback and candidate Nv readback. No projection/composite/loss/backward/Adam/ADC.\""
                             ",\"candidate_bundle\":\"Stable compact + Nv depth sort + reordered prefix/emit + effective-bit tile KV sort + direct Gaussian IDs + direct-key ranges\""
                             ",\"key_storage_bits\":32,\"physical_uint16_storage\":false,\"stable_compaction\":true"
                             ",\"cpu_sequence_includes\":\"Prepare/main recording, submissions, original fences, count readback/checks. Excludes restore, downloads/verification and timestamp query retrieval.\""
                             ",\"gpu_active_includes\":\"Prepare and main GPU intervals including count copies; CPU scheduling gap excluded and separately reported in gpu_span_ms and cpu_sequence_ms.\""
                          : ",\"full_list_sequence_measured\":false,\"scope\":\"GPU depth sort + high-key gather + tile sort + item gather + "
                "boundary construction + range finalize. No projection/count/prefix/emit/composite/backward/Adam.\""
                )
             << (integrated ? ",\"restore_policy\":\"Private TrainingBuffers restored per A/B/A; renderer workspaces retained across growth/adoption fixtures. No timing collected.\""
                            : downstream ? ",\"restore_policy\":\"Private producer/consumer workspace reset per A/B/A; pixelGrad readonly. GPU list producers feed consumers directly; no combined timing collected.\""
                            : ",\"restore_policy\":\"Private seeds/work/ranges/radix scratch restored outside every measured submission; "
                "producer sorts are executed inside the measured sequence for A/B/A, not supplied by CPU. Cache-perturbed.\"");
        if (downstream) {
            vk::PhysicalDeviceSubgroupProperties subgroup;
            vk::PhysicalDeviceProperties2 properties; properties.pNext = &subgroup;
            context.physical.getProperties2(&properties);
            json << ",\"portable_reference\":" << portableReference
                 << ",\"backward_reference\":" << schedulingJsonQuote(portableReference ? "Direct synthetic fixture only" : "Tensor Subgroup Global Baseline")
                 << ",\"shader_buffer_float32_atomic_add_enabled\":" << context.atomicFloatEnabled
                 << ",\"native_subgroup_size\":" << subgroup.subgroupSize
                 << ",\"max_shared_memory_bytes\":" << context.properties.limits.maxComputeSharedMemorySize
                 << ",\"color_blend_contract\":\"Bitwise equality\""
                 << ",\"gradient_contract\":\"Existing compareRowGradients: abs 1e-5 + rel 1e-3 per component; finite, zero padding, identical Gaussian count\"";
        }
        json << ",\"shader_fnv1a64\":{";
        bool first = true;
        std::vector<const char*> shaderNames{"train_forward_tile_range_boundaries", "train_range_boundaries_sorted_keys_test",
                 "train_forward_tile_gather_high", "train_forward_tile_gather_items", "train_forward_tile_sort"};
        if (fullChain) for (const char* name : {"train_forward_tile_count", "train_forward_tile_emit", "train_forward_tile_clear",
                "train_full_sort_prepare_test", "train_full_sort_emit_test", "train_forward_gaussian_prefix_ranges",
                "train_forward_gaussian_prefix_scratch", "train_forward_gaussian_prefix_top",
                "train_forward_gaussian_prefix_add_scratch", "train_forward_gaussian_prefix_add_ranges"}) shaderNames.push_back(name);
        if (integrated) shaderNames.push_back("train_forward_workgroup");
        if (downstream) {
            shaderNames.push_back("train_forward_workgroup");
            shaderNames.push_back(portableReference ? "train_backward_pixel_to_2dgs" : "train_backward_vksplat_tensor_subgroup_global_grad");
        }
        for (const char* name : shaderNames) {
            if (!first) json << ',';
            first = false;
            const auto path = std::filesystem::path(RANGE_KEY_RUNTIME_DIR) / "shaders" / (std::string(name) + ".comp.spv");
            json << schedulingJsonQuote(name) << ":\"" << optimizerShaderHash(path) << '"';
        }
        json << "},\"cases\":[" << cases.str() << "]}\n";
        report->write(json.str());
        if (downstream && !downstreamPassed) { std::cerr << "Private downstream comparison failed: see report\n"; return 1; }
        std::cout << (integrated ? "Private renderer sorting integration complete: " : downstream ? "Private sort downstream comparison complete: " : fullChain ? "Private full-sort comparison complete: " :
            "Private range-key comparison complete: ") << output << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Private range-key comparison failed: " << error.what() << '\n';
        if (report && !report->written()) {
            try { report->write("{\"schema\":" + schedulingJsonQuote(integrated ? "faster_sort_integration_gpu_v1" : downstream ? "faster_sort_downstream_gpu_v1" : fullChain ? "faster_sort_chain_gpu_v1" : "range_sorted_keys_gpu_v1") + ",\"status\":\"STOP\",\"error\":" +
                schedulingJsonQuote(error.what()) + ",\"validation_errors\":" + std::to_string(validationErrors.load()) +
                failureContext + ",\"full_training_speedup\":null}\n"); } catch (...) {}
        }
        return 1;
    }
    return 0;
}
