#include "training_sort_baseline_session.hpp"
#include "utils/logger.hpp"
#include "utils/system_memory.hpp"

#include <atomic>
#include <charconv>
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace sort_replay;
namespace {
std::atomic<uint32_t> validationErrors{0};
VKAPI_ATTR vk::Bool32 VKAPI_CALL validationMessage(vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
    vk::DebugUtilsMessageTypeFlagsEXT, const vk::DebugUtilsMessengerCallbackDataEXT* data, void*) {
    if (severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eError) {
        ++validationErrors;
        std::cerr << "Sort replay validation: " << data->pMessage << '\n';
    }
    return VK_FALSE;
}
uint32_t integer(const char* value) {
    uint32_t result = 0;
    const std::string_view text(value);
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), result);
    check(parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size(), "Invalid device index");
    return result;
}
void sampleJson(std::ostream& out, const Sample& sample) {
    out << '{';
    auto interval = [&](const char* name, uint32_t a, uint32_t b) {
        out << '"' << name << "\":";
        if (sample.cumulative[a] && sample.cumulative[b])
            out << *sample.cumulative[b] - *sample.cumulative[a];
        else out << "null";
        out << ',';
    };
    interval("count_clear_ms", 0, 1);
    interval("prefix_ms", 1, 2);
    interval("emit_ms", 3, 4);
    interval("depth_sort_ms", 4, 5);
    interval("high_gather_ms", 5, 6);
    interval("tile_sort_ms", 6, 7);
    interval("item_gather_ms", 7, 8);
    interval("ranges_ms", 8, 9);
    interval("range_boundaries_ms", 8, uint32_t(TrainingSortTimestamp::RangeBoundaryDispatchEnd));
    interval("range_boundary_barrier_ms", uint32_t(TrainingSortTimestamp::RangeBoundaryDispatchEnd),
             uint32_t(TrainingSortTimestamp::RangeBoundaryBarrierEnd));
    interval("range_finalize_ms", uint32_t(TrainingSortTimestamp::RangeBoundaryBarrierEnd),
             uint32_t(TrainingSortTimestamp::RangeFinalizeDispatchEnd));
    interval("range_finalize_barrier_ms", uint32_t(TrainingSortTimestamp::RangeFinalizeDispatchEnd), 9);
    interval("sort_ranges_ms", 4, 9);
    interval("prepare_gpu_ms", 0, 2);
    interval("main_gpu_ms", 3, 9);
    interval("gpu_span_including_host_gap_ms", 0, 9);
    out << "\"restore_cpu_submit_wait_ms\":" << sample.restoreCpuMs
        << ",\"prepare_cpu_submit_wait_ms\":" << sample.prepareCpuMs
        << ",\"count_read_cpu_ms\":" << sample.countReadCpuMs
        << ",\"main_cpu_submit_wait_ms\":" << sample.mainCpuMs << '}';
}
struct WorkingDirectory {
    std::filesystem::path before = std::filesystem::current_path();
    ~WorkingDirectory() { std::error_code error; std::filesystem::current_path(before, error); }
};
constexpr const char* usage =
    "training_sort_baseline_gpu_tests --capture INPUT.scanline.bin --report NEW.json "
    "[--inputs NEW.sort-input.bin] [--timing] [--require-validation | --no-validation] "
    "[--device-index N]";
} // namespace

int main(int argc, char** argv) {
    std::unique_ptr<TrainingDiagnosticReport> report, rawFile;
    try {
        std::filesystem::path input, output, rawPath;
        bool timing = false, requireValidation = false, noValidation = false;
        uint32_t deviceIndex = 0;
        for (int i = 1; i < argc; ++i) {
            const std::string_view arg(argv[i]);
            if (arg == "--help") { std::cout << usage << '\n'; return 0; }
            if (arg == "--timing") timing = true;
            else if (arg == "--require-validation") requireValidation = true;
            else if (arg == "--no-validation") noValidation = true;
            else if (arg == "--capture" && i + 1 < argc) input = argv[++i];
            else if (arg == "--report" && i + 1 < argc) output = argv[++i];
            else if (arg == "--inputs" && i + 1 < argc) rawPath = argv[++i];
            else if (arg == "--device-index" && i + 1 < argc) deviceIndex = integer(argv[++i]);
            else throw std::runtime_error(usage);
        }
        check(!input.empty() && !output.empty() && !(requireValidation && noValidation), usage);
        input = std::filesystem::absolute(input);
        output = std::filesystem::absolute(output);
        if (!rawPath.empty()) rawPath = std::filesystem::absolute(rawPath);
        check(std::filesystem::file_size(input) <= (256ull << 20), "Capture exceeds 256 MiB input budget");
        report = std::make_unique<TrainingDiagnosticReport>(output);
        if (!rawPath.empty()) rawFile = std::make_unique<TrainingDiagnosticReport>(rawPath);
        const auto source = readTrainingScanlineCapture(input);
        const auto& h = source.header;
        check(h.gaussianCount && h.gaussianCount <= 2000000 && h.itemCount <= 16000000,
              "Replay workload exceeds bounded Gaussian/item limit");
        check(availableSystemMemoryBytes() > uint64_t(h.gaussianCount) * 600 +
              uint64_t(h.itemCount) * 80 + (256ull << 20), "Insufficient CPU memory headroom");
        VULKAN_HPP_DEFAULT_DISPATCHER.init();
        const char* layer = "VK_LAYER_KHRONOS_validation";
        bool validation = false, synchronization = false;
        if (!noValidation)
            for (const auto& available : vk::enumerateInstanceLayerProperties())
                validation |= std::string_view(available.layerName.data()) == layer;
        vk::ValidationFeatureEnableEXT enabled = vk::ValidationFeatureEnableEXT::eSynchronizationValidation;
        vk::ValidationFeaturesEXT validationFeatures;
        vk::ApplicationInfo app{"sort-baseline-replay", 1, "vulkan-3dgs", 1, VK_API_VERSION_1_2};
        vk::InstanceCreateInfo instanceInfo;
        instanceInfo.setPApplicationInfo(&app);
        std::vector<const char*> instanceExtensions;
        if (validation) {
            instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
            for (const auto& ext : vk::enumerateInstanceExtensionProperties(std::string(layer)))
                synchronization |= std::string_view(ext.extensionName.data()) == VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME;
            if (synchronization) {
                instanceExtensions.push_back(VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME);
                validationFeatures.setEnabledValidationFeatures(enabled);
                instanceInfo.setPNext(&validationFeatures);
            }
            instanceInfo.setPEnabledLayerNames(layer).setPEnabledExtensionNames(instanceExtensions);
        }
        check(!requireValidation || (validation && synchronization),
              "Required core/synchronization validation unavailable");
        auto instance = vk::createInstanceUnique(instanceInfo);
        VULKAN_HPP_DEFAULT_DISPATCHER.init(*instance);
        vk::UniqueDebugUtilsMessengerEXT messenger;
        if (validation) {
            vk::DebugUtilsMessengerCreateInfoEXT debug;
            debug.setMessageSeverity(vk::DebugUtilsMessageSeverityFlagBitsEXT::eError)
                .setMessageType(vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
                                vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance);
            debug.pfnUserCallback = validationMessage;
            messenger = instance->createDebugUtilsMessengerEXTUnique(debug);
        }
        const auto devices = instance->enumeratePhysicalDevices();
        check(deviceIndex < devices.size(), "Requested device index unavailable");
        const auto physical = devices[deviceIndex];
        bool pushDescriptors = false;
        for (const auto& ext : physical.enumerateDeviceExtensionProperties())
            pushDescriptors |= std::string_view(ext.extensionName.data()) == VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME;
        check(pushDescriptors, "Sorter requires VK_KHR_push_descriptor");
        const auto families = physical.getQueueFamilyProperties();
        uint32_t family = 0;
        while (family < families.size() && !(families[family].queueFlags & vk::QueueFlagBits::eCompute)) ++family;
        check(family < families.size() && families[family].timestampValidBits, "Compute timestamps unavailable");
        vk::PhysicalDeviceIDProperties ids;
        vk::PhysicalDeviceProperties2 properties;
        properties.pNext = &ids;
        physical.getProperties2(&properties);
        const auto& limits = properties.properties.limits;
        check(limits.maxComputeWorkGroupInvocations >= 512 && limits.maxComputeWorkGroupSize[0] >= 512 &&
                  limits.maxComputeSharedMemorySize >= 32768 &&
                  uint64_t(h.gaussianCount) * 32 <= limits.maxStorageBufferRange &&
                  uint64_t(h.width) * h.height * 64 <= limits.maxStorageBufferRange,
              "Replay workspace/workgroup exceeds device limits");
        checkOptimizerCaptureBudget(physical, h.gaussianCount,
            uint64_t(h.gaussianCount) * 2048 + uint64_t(h.width) * h.height * 256 + (64ull << 20));
        const float priority = 1;
        vk::DeviceQueueCreateInfo queueInfo{{}, family, 1, &priority};
        const char* deviceExtension = VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME;
        vk::PhysicalDeviceFeatures features;
        features.shaderInt64 = physical.getFeatures().shaderInt64;
        vk::DeviceCreateInfo deviceInfo;
        deviceInfo.setQueueCreateInfos(queueInfo).setPEnabledExtensionNames(deviceExtension).setPEnabledFeatures(&features);
        auto device = physical.createDeviceUnique(deviceInfo);
        VULKAN_HPP_DEFAULT_DISPATCHER.init(*device);
        Logger::get_instance().set_level(LogLevel::ERROR_VULKAN_3DGS);
        WorkingDirectory directory;
        std::filesystem::current_path(SORT_REPLAY_RUNTIME_DIR);
        uint32_t capacity = 0;
        std::string binary;
        std::ostringstream samples;
        samples.precision(10);
        {
            BaselineSession session(*device, physical, device->getQueue(family, 0), family);
            session.initialize(source);
            const auto raw = session.captureEmit();
            verify(raw, source);
            capacity = session.capacity();
            if (rawFile) binary = serialize(raw);
            session.run(false);
            session.verifySorted(source);
            check(!validationErrors, "Raw emit/list correctness validation failed");
            if (timing) {
                bool first = true;
                for (uint32_t phase = 0; phase < 4; ++phase) {
                    const bool segmented = phase < 3;
                    for (uint32_t i = 0; i < 2; ++i) session.run(segmented);
                    for (uint32_t i = 0; i < 8; ++i) {
                        const auto sample = session.run(segmented);
                        if (!first) samples << ',';
                        first = false;
                        samples << "{\"phase\":" << phase << ",\"segmented\":" << std::boolalpha << segmented << ",\"timing\":";
                        sampleJson(samples, sample);
                        samples << '}';
                    }
                }
                session.verifySorted(source);
            }
        }
        check(!validationErrors, "Core/synchronization validation errors");
        std::ostringstream uuid, json;
        for (uint8_t value : ids.deviceUUID) uuid << std::hex << std::setw(2) << std::setfill('0') << unsigned(value);
        json.precision(10);
        json << "{\"schema\":\"sort_baseline_replay_v1\",\"status\":\"BASELINE_REPLAY_PASSED_NOT_CANDIDATE_ADMISSION\","
             << "\"device\":" << schedulingJsonQuote(properties.properties.deviceName.data())
             << ",\"device_uuid\":" << schedulingJsonQuote(uuid.str())
             << ",\"device_index\":" << deviceIndex << ",\"vendor_id\":" << properties.properties.vendorID
             << ",\"device_id\":" << properties.properties.deviceID
             << ",\"driver_version\":" << properties.properties.driverVersion
             << ",\"timestamp_valid_bits\":" << families[family].timestampValidBits
             << ",\"timestamp_period_ns\":" << limits.timestampPeriod
             << ",\"sort_timestamp_count\":" << kTrainingSortTimestampCount
             << ",\"validation_enabled\":" << std::boolalpha << validation
             << ",\"synchronization_validation_enabled\":" << synchronization
             << ",\"validation_errors\":" << validationErrors.load()
             << ",\"cpp_build\":\"" SORT_REPLAY_BUILD_CONFIG "\",\"compiler\":"
             << schedulingJsonQuote(SORT_REPLAY_COMPILER)
             << ",\"shader_compiler\":" << schedulingJsonQuote(SORT_REPLAY_SHADER_COMPILER)
             << ",\"sorter_header_sha256\":\"" SORT_REPLAY_SORTER_HASH "\""
             << ",\"source_capture\":" << schedulingJsonQuote(input.string())
             << ",\"source_capture_fnv1a64\":\"" << optimizerShaderHash(input) << "\""
             << ",\"iteration\":" << h.iteration << ",\"frame\":" << h.frame
             << ",\"gaussians\":" << h.gaussianCount << ",\"items\":" << h.itemCount
             << ",\"width\":" << h.width << ",\"height\":" << h.height
             << ",\"replay_capacity\":" << capacity << ",\"historical_live_capacity\":null"
             << ",\"raw_inputs\":" << (rawFile ? schedulingJsonQuote(rawPath.string()) : "null")
             << ",\"gpu_count_emit_CPU_oracle_and_sorted_capture_match\":true,\"readonly_projected_match\":true"
             << ",\"training_advanced\":false,\"candidate_enabled\":false,\"speedup\":null"
             << ",\"scope\":\"Original renderer on private buffers from frozen projected inputs; "
                "count/clear/prefix/readback + emit/depth-sort/high-gather/tile-sort/item-gather/ranges. "
                "No projection, composite, loss, backward, Adam or live training.\""
             << ",\"restore_policy\":\"Mutable renderer arrays cleared outside each measured pair of original-style "
                "prepare/main submissions; sorter initializes its own scratch. Cache-perturbed replay; "
                "allocation warmed, no capacity growth measured.\""
             << ",\"warmup_per_phase\":2,\"measured_per_phase\":8,\"timing_requested\":" << timing
             << ",\"timing_layout\":\"phases0/1/2 are segmented A/A/A; phase3 omits internal sort timestamps. "
                "No candidate B; 32 measured samples when timing requested.\""
             << ",\"shader_fnv1a64\":{";
        bool first = true;
        for (const char* name : {"train_forward_tile_clear", "train_forward_tile_count", "train_forward_tile_emit",
                 "train_forward_gaussian_prefix_ranges", "train_forward_gaussian_prefix_scratch",
                 "train_forward_gaussian_prefix_top", "train_forward_gaussian_prefix_add_scratch",
                 "train_forward_gaussian_prefix_add_ranges", "train_forward_tile_gather_high",
                 "train_forward_tile_gather_items", "train_forward_tile_range_boundaries", "train_forward_tile_sort"}) {
            if (!first) json << ',';
            first = false;
            json << schedulingJsonQuote(name) << ":\"" << optimizerShaderHash(std::filesystem::path("shaders") /
                (std::string(name) + ".comp.spv")) << '"';
        }
        json << "},\"samples\":[" << samples.str() << "]}\n";
        if (rawFile) rawFile->write(binary);
        report->write(json.str());
        std::cout << "Sort baseline replay complete: " << output << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Sort baseline replay failed: " << error.what() << '\n';
        if (report && !report->written()) {
            try { report->write("{\"schema\":\"sort_baseline_replay_v1\",\"status\":\"STOP\",\"error\":" +
                schedulingJsonQuote(error.what()) + ",\"speedup\":null}\n"); } catch (...) {}
        }
        return 1;
    }
    return 0;
}
