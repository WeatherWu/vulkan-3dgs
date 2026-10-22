#include "training_optimizer_r1_metrics.hpp"
#include "training_scheduling_sort_session.hpp"

#include <atomic>
#include <charconv>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace scheduling_test;
namespace {
std::atomic<uint32_t> validationErrors{0};
VKAPI_ATTR vk::Bool32 VKAPI_CALL message(vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
                                         vk::DebugUtilsMessageTypeFlagsEXT,
                                         const vk::DebugUtilsMessengerCallbackDataEXT* data,
                                         void*) {
    if (severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eError) {
        ++validationErrors;
        std::cerr << "S4-3 Vulkan validation: " << data->pMessage << '\n';
    }
    return VK_FALSE;
}
uint32_t integer(const char* value) {
    uint32_t number = 0;
    const std::string_view text(value);
    const auto result = std::from_chars(text.data(), text.data() + text.size(), number);
    check(result.ec == std::errc{} && result.ptr == text.data() + text.size(),
          "Invalid integer option");
    return number;
}
void samplesJson(std::ostream& out, const std::vector<Sample>& samples) {
    out << '[';
    for (size_t i = 0; i < samples.size(); ++i) {
        if (i) out << ',';
        out << "{\"stage_ms\":";
        if (samples[i].workMs)
            out << *samples[i].workMs;
        else
            out << "null";
        out << ",\"restore_ms\":";
        if (samples[i].restoreMs)
            out << *samples[i].restoreMs;
        else
            out << "null";
        out << ",\"cpu_submit_wait_ms\":" << samples[i].cpuSubmitWaitMs << '}';
    }
    out << ']';
}
std::string testCase(vk::Device device, vk::PhysicalDevice physical, vk::Queue queue,
                     uint32_t family, uint32_t count, uint32_t capacity, bool ties, bool timing) {
    const auto shader = std::filesystem::path(SCHEDULING_RUNTIME_DIR) /
                        "shaders/train_scheduling_sort_test.comp.spv";
    SortSession session(device, physical, queue, family, count, capacity, ties, shader);
    for (bool indirect : {false, true, false}) {
        session.dispatch(indirect);
        session.verify();
        check(!validationErrors, "Sort correctness synchronization validation failed");
    }
    std::ostringstream out;
    out.precision(10);
    out << "{\"count\":" << count << ",\"capacity\":" << capacity
        << ",\"all_ties\":" << std::boolalpha << ties
        << ",\"bitwise_CPU_oracle_A_B_A\":true,\"tails_seeds_canaries_verified\":true"
        << ",\"direct_partition_groups_per_sort\":" << (count > 1 ? (count + 4095u) / 4096u : 0u)
        << ",\"indirect_partition_groups_per_sort\":" << (capacity + 4095u) / 4096u;
    if (timing) {
        std::array<std::vector<Sample>, 3> phases;
        for (size_t p = 0; p < 3; ++p) {
            for (uint32_t i = 0; i < 2; ++i)
                session.dispatch(p == 1);
            for (uint32_t i = 0; i < 8; ++i)
                phases[p].push_back(session.dispatch(p == 1));
        }
        session.verify(); // No downloads between timed phases.
        session.dispatch(true);
        session.verify();
        check(!validationErrors, "Sort timing synchronization validation failed");
        out << ",\"warmup_per_phase\":2,\"measured_per_phase\":8,\"A1\":";
        samplesJson(out, phases[0]);
        out << ",\"B\":";
        samplesJson(out, phases[1]);
        out << ",\"A2\":";
        samplesJson(out, phases[2]);
        bool available = true;
        std::array<std::array<double, 8>, 3> times{};
        for (size_t p = 0; p < 3; ++p)
            for (size_t i = 0; i < 8; ++i) {
                available &= phases[p][i].workMs.has_value();
                times[p][i] = phases[p][i].workMs.value_or(0);
            }
        if (available) {
            const auto decision = optimizer_r1::evaluate(times[0], times[1], times[2]);
            const bool stable = decision.verdict != "INCONCLUSIVE_SAMPLE_OUTLIER" &&
                                decision.verdict != "INCONCLUSIVE_BASELINE_DRIFT";
            out << ",\"A1_mean_ms\":" << decision.a1.mean << ",\"B_mean_ms\":" << decision.b.mean
                << ",\"A2_mean_ms\":" << decision.a2.mean
                << ",\"baseline_drift_percent\":" << decision.baselineDriftPercent
                << ",\"indirect_incremental_stage_ms\":" << -decision.gainMs
                << ",\"cost_sample_verdict\":\""
                << (stable ? "COST_SAMPLE_ONLY_REQUIRES_TRAINING_GAP_BUDGET" : decision.verdict)
                << '"';
        } else
            out << ",\"cost_sample_verdict\":\"TIMESTAMPS_UNAVAILABLE\"";
    }
    out << '}';
    return out.str();
}
void timingSmoke(vk::Device device, vk::PhysicalDevice physical, vk::Queue queue, uint32_t family,
                 const std::filesystem::path& reportPath, bool instanceValidationEnabled) {
    SchedulingDiagnosticConfig config{reportPath, 2, 0, true};
    TrainingSchedulingDiagnostics diagnostics(device, physical, family, config,
                                              instanceValidationEnabled);
    Buffer work;
    work.create(device, physical, queue, family, nullptr, 16,
                vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
                vk::MemoryPropertyFlagBits::eDeviceLocal, {family});
    CommandPool pool;
    pool.create(device, family, vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
    auto command = pool.allocateCommandBuffer();
    auto fence = device.createFenceUnique({});
    auto submit = [&] {
        command.end();
        device.resetFences(*fence);
        vk::SubmitInfo info;
        info.setCommandBuffers(command);
        queue.submit(info, *fence);
        (void)device.waitForFences(*fence, true, UINT64_MAX);
    };
    SchedulingStepMetadata fixed;
    fixed.fixedBenchmark = true;
    check(!diagnostics.wantsStep(fixed), "Fixed Benchmark included in ordinary timing sampling");
    for (uint32_t i = 0; i < 2; ++i) {
        SchedulingStepSample sample;
        sample.metadata.iteration = i + 1;
        sample.metadata.optimizerEnabled = true;
        sample.metadata.adc = sample.metadata.validation = sample.metadata.resized = i == 1;
        check(diagnostics.wantsStep(sample.metadata), "Timing sampler rejected bounded step");
        command.reset();
        command.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        diagnostics.recordPrepareBegin(command);
        command.fillBuffer(work.getBuffer(), 0, 16, i + 1u);
        diagnostics.recordPrepareEnd(command);
        submit();
        command.reset();
        command.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        diagnostics.recordMainBegin(command);
        submit();
        diagnostics.finishStep(sample);
    }
    check(!diagnostics.wantsStep({}), "Timing sampler did not stop at bound");
    std::ifstream input(reportPath);
    const std::string text((std::istreambuf_iterator<char>(input)), {});
    check(text.find("\"status\":\"COMPLETE\"") != std::string::npos &&
              text.find("\"schema\":\"s4_scheduling_timing_v2\"") != std::string::npos &&
              text.find("\"collected_samples\":2") != std::string::npos &&
              text.find("\"synthetic\":true") != std::string::npos,
          "Timing smoke report not committed");
    if (physical.getQueueFamilyProperties()[family].timestampValidBits)
        check(text.find("\"gpu_prepare_to_main_gap_ms\":null") == std::string::npos &&
                  text.find("\"gpu_prepare_duration_ms\":null") == std::string::npos,
              "Timestamp-capable smoke did not collect all three GPU queries");
    check(text.find(instanceValidationEnabled
                        ? "\"vulkan_validation_layers_enabled\":true"
                        : "\"vulkan_validation_layers_enabled\":false") != std::string::npos,
          "Standalone instance state not passed into sampler");
    check(text.find("\"logger_level\":null") == std::string::npos, "Runtime logger level missing");
    OptimizerCaptureTransfer transfer(device, physical, queue, family);
    std::array<uint32_t, 4> actual{};
    transfer.download(work.getDescriptorInfo(), 0, std::as_writable_bytes(std::span(actual)));
    check(std::all_of(actual.begin(), actual.end(), [](uint32_t value) { return value == 2u; }),
          "Timestamp probe changed ordinary work");
    check(!validationErrors, "Scheduling timestamp smoke validation failed");
}
} // namespace

int main(int argc, char** argv) {
    std::unique_ptr<TrainingDiagnosticReport> report;
    std::string identity;
    bool validation = false, syncValidation = false;
    try {
        bool synthetic = false, requireValidation = false, haveCount = false, haveCapacity = false;
        uint32_t count = 0, capacity = 0, deviceIndex = 0;
        std::filesystem::path path;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--synthetic")
                synthetic = true;
            else if (arg == "--require-validation")
                requireValidation = true;
            else if (arg == "--report" && i + 1 < argc)
                path = argv[++i];
            else if ((arg == "--count" || arg == "--capacity" || arg == "--device-index") &&
                     i + 1 < argc) {
                const auto value = integer(argv[++i]);
                if (arg == "--count") {
                    count = value;
                    haveCount = true;
                } else if (arg == "--capacity") {
                    capacity = value;
                    haveCapacity = true;
                } else
                    deviceIndex = value;
            } else
                throw std::runtime_error(
                    "Usage: training_scheduling_gpu_tests (--synthetic | --count N --capacity C) "
                    "--report NEW.json [--require-validation] [--device-index N]");
        }
        check(!path.empty(), "A NEW --report path is required");
        check(synthetic ? (!haveCount && !haveCapacity)
                        : (haveCount && haveCapacity && count <= capacity && capacity &&
                           capacity <= (1u << 22)),
              "Choose synthetic or valid bounded count/capacity (capacity <=4194304)");
        path = std::filesystem::absolute(path);
        report = std::make_unique<TrainingDiagnosticReport>(path);
        VULKAN_HPP_DEFAULT_DISPATCHER.init();
        const char* layer = "VK_LAYER_KHRONOS_validation";
        for (const auto& p : vk::enumerateInstanceLayerProperties())
            validation |= std::strcmp(p.layerName.data(), layer) == 0;
        std::vector<const char*> extensions;
        vk::ValidationFeatureEnableEXT feature =
            vk::ValidationFeatureEnableEXT::eSynchronizationValidation;
        vk::ValidationFeaturesEXT validationFeatures;
        vk::ApplicationInfo app{"s4-scheduling", 1, "vulkan-3dgs", 1, VK_API_VERSION_1_2};
        vk::InstanceCreateInfo instanceInfo;
        instanceInfo.setPApplicationInfo(&app);
        if (validation) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
            for (const auto& p : vk::enumerateInstanceExtensionProperties(std::string(layer)))
                syncValidation |= std::strcmp(p.extensionName.data(),
                                              VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME) == 0;
            if (syncValidation) {
                extensions.push_back(VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME);
                validationFeatures.setEnabledValidationFeatures(feature);
                instanceInfo.setPNext(&validationFeatures);
            }
            instanceInfo.setPEnabledLayerNames(layer).setPEnabledExtensionNames(extensions);
        }
        check(!requireValidation || (validation && syncValidation),
              "Required core/synchronization validation unavailable");
        auto instance = vk::createInstanceUnique(instanceInfo);
        VULKAN_HPP_DEFAULT_DISPATCHER.init(*instance);
        vk::UniqueDebugUtilsMessengerEXT messenger;
        if (validation) {
            vk::DebugUtilsMessengerCreateInfoEXT info;
            info.setMessageSeverity(vk::DebugUtilsMessageSeverityFlagBitsEXT::eError)
                .setMessageType(vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
                                vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance);
            info.pfnUserCallback = message;
            messenger = instance->createDebugUtilsMessengerEXTUnique(info);
        }
        const auto devices = instance->enumeratePhysicalDevices();
        check(deviceIndex < devices.size(), "Requested GPU index unavailable");
        const auto physical = devices[deviceIndex];
        bool pushDescriptors = false;
        for (const auto& p : physical.enumerateDeviceExtensionProperties())
            pushDescriptors |=
                std::strcmp(p.extensionName.data(), VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME) == 0;
        check(pushDescriptors, "Sorter requires VK_KHR_push_descriptor");
        const auto families = physical.getQueueFamilyProperties();
        uint32_t family = 0;
        while (family < families.size() &&
               !(families[family].queueFlags & vk::QueueFlagBits::eCompute))
            ++family;
        check(family < families.size(), "Compute queue unavailable");
        vk::PhysicalDeviceIDProperties ids;
        vk::PhysicalDeviceProperties2 properties;
        properties.pNext = &ids;
        physical.getProperties2(&properties);
        std::ostringstream uuid;
        for (uint8_t byte : ids.deviceUUID)
            uuid << std::hex << std::setw(2) << std::setfill('0') << unsigned(byte);
        identity =
            "\"device\":" + schedulingJsonQuote(properties.properties.deviceName.data()) +
            ",\"device_uuid\":" + schedulingJsonQuote(uuid.str()) +
            ",\"device_index\":" + std::to_string(deviceIndex) +
            ",\"driver_version\":" + std::to_string(properties.properties.driverVersion) + "," +
            schedulingRuntimeMetadataJson(validation) + ",\"helper_shader_fnv1a64\":\"" +
            std::to_string(optimizerShaderHash(std::filesystem::path(SCHEDULING_RUNTIME_DIR) /
                                               "shaders/train_scheduling_sort_test.comp.spv")) +
            "\"" + ",\"sorter_header_sha256\":\"" SCHEDULING_SORTER_HASH "\"";
        const float priority = 1;
        vk::DeviceQueueCreateInfo queueInfo{{}, family, 1, &priority};
        const char* extension = VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME;
        vk::PhysicalDeviceFeatures features;
        features.shaderInt64 = physical.getFeatures().shaderInt64;
        vk::DeviceCreateInfo deviceInfo;
        deviceInfo.setQueueCreateInfos(queueInfo)
            .setPEnabledExtensionNames(extension)
            .setPEnabledFeatures(&features);
        auto device = physical.createDeviceUnique(deviceInfo);
        VULKAN_HPP_DEFAULT_DISPATCHER.init(*device);
        const auto queue = device->getQueue(family, 0);
        std::ostringstream cases;
        if (synthetic) {
            auto smokePath = path;
            smokePath.replace_extension(".timing_smoke.json");
            timingSmoke(*device, physical, queue, family, smokePath, validation);
            uint32_t index = 0;
            for (auto [n, c] : std::array<std::pair<uint32_t, uint32_t>, 7>{{{0, 17},
                                                                             {1, 17},
                                                                             {255, 255},
                                                                             {256, 512},
                                                                             {257, 1024},
                                                                             {4097, 8194},
                                                                             {8193, 16386}}}) {
                if (index++) cases << ',';
                cases << testCase(*device, physical, queue, family, n, c, false, false);
                std::cerr << "S4-3 sort correctness passed N=" << n << " capacity=" << c << '\n';
            }
            cases << ',' << testCase(*device, physical, queue, family, 257, 1024, true, false);
        } else
            cases << testCase(*device, physical, queue, family, count, capacity, false, true);
        device->waitIdle();
        check(!validationErrors, "Core/synchronization validation error(s)");
        const std::string json =
            "{" + identity +
            ",\"schema\":\"s4_indirect_sort_cost_v1\",\"status\":\"DIAGNOSTICS_PASSED_NOT_TRAINING_"
            "ADMISSION\""
            ",\"synthetic_inputs\":true,\"gpu_driven_training_enabled\":false,\"speedup\":null,"
            "\"count_source\":\"GPU fill in synthetic setup, fixed during A/B/A; not a renderer "
            "capture\","
            "\"stage\":\"two 32-bit stable sorts + high-key gather + start/count ranges; "
            "NOT final item gather, emit, forward or backward\","
            "\"restore_policy\":\"ALL keys/indices/tails/ranges/scratch restored before each "
            "invocation; "
            "separate fenced restore excluded from stage; cache-perturbed isolated cost\","
            "\"capacity_dispatch\":\"library groups use maxElementCount, not runtime count; "
            "helper gather/range groups also capacity-based in B\","
            "\"validation_enabled\":" +
            std::string(validation ? "true" : "false") +
            ",\"synchronization_validation_enabled\":" + (syncValidation ? "true" : "false") +
            ",\"validation_errors\":" + std::to_string(validationErrors.load()) + ",\"cases\":[" +
            cases.str() + "]}\n";
        report->write(json);
        std::cout << json;
    } catch (const std::exception& error) {
        std::cerr << "S4-3 scheduling diagnostics failed: " << error.what() << '\n';
        if (report && !report->written()) {
            try {
                report->write("{" + (identity.empty() ? "" : identity + ',') +
                              "\"status\":\"STOP\",\"error\":" + schedulingJsonQuote(error.what()) +
                              ",\"validation_errors\":" + std::to_string(validationErrors.load()) +
                              ",\"speedup\":null}\n");
            } catch (...) {}
        }
        return 1;
    }
    return 0;
}
