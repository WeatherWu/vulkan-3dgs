#include "training/core/training_optimizer_capture_gpu.hpp"
#include "training/core/training_scheduling_diagnostics.hpp"
#include "training/core/training_step_executor.hpp"

#include <array>
#include <atomic>
#include <iostream>
#include <sstream>

using namespace vulkan3DGS;
namespace {
std::atomic<uint32_t> errors{0};
VKAPI_ATTR vk::Bool32 VKAPI_CALL message(vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
                                         vk::DebugUtilsMessageTypeFlagsEXT,
                                         const vk::DebugUtilsMessengerCallbackDataEXT* data,
                                         void*) {
    if (severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eError) {
        ++errors;
        std::cerr << "Diagnostic transfer validation: " << data->pMessage << '\n';
    }
    return VK_FALSE;
}
void check(bool okay, const char* reason) {
    if (!okay) throw std::runtime_error(reason);
}
bool transferOnly(vk::QueueFlags flags) {
    return (flags & vk::QueueFlagBits::eTransfer) &&
           !(flags & (vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute));
}
void exercise(vk::Device device, vk::PhysicalDevice physical, uint32_t computeFamily,
              uint32_t downloadFamily) {
    const auto compute = device.getQueue(computeFamily, 0);
    const auto download = device.getQueue(downloadFamily, 0);
    constexpr uint32_t bytes = 4096, pattern = 0x3f123456;
    Buffer buffer;
    buffer.create(device, physical, download, downloadFamily, nullptr, bytes,
                  vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
                  vk::MemoryPropertyFlagBits::eDeviceLocal, {computeFamily, downloadFamily});
    CommandPool pool;
    pool.create(device, computeFamily, vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
    const auto command = pool.allocateCommandBuffer();
    command.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    command.fillBuffer(buffer.getBuffer(), 0, bytes, pattern);
    command.end();
    vk::SubmitInfo submit;
    submit.setCommandBuffers(command);
    compute.submit(submit);

    TrainingStepExecutor executor;
    executor.initialize(device, compute, computeFamily);
    // Exercise the same producer barrier/semaphore handoff used by smoke fingerprints.
    executor.synchronizeDiagnosticDownloads(download);
    OptimizerCaptureTransfer transfer(device, physical, download, downloadFamily);
    std::array<uint32_t, bytes / 4> actual;
    transfer.download(buffer.getDescriptorInfo(), 0, std::as_writable_bytes(std::span(actual)));
    for (auto value : actual)
        check(value == pattern, "Producer-to-download content mismatch");

    // Reuse staging through host read, host write, device read and device write cycles.
    std::array<uint32_t, 257> expected;
    for (uint32_t round = 0; round < 3; ++round) {
        for (size_t i = 0; i < expected.size(); ++i)
            expected[i] = uint32_t(i) * 7919u + round;
        transfer.upload(buffer.getDescriptorInfo(), 12, std::as_bytes(std::span(expected)));
        std::array<uint32_t, 257> restored;
        transfer.download(buffer.getDescriptorInfo(), 12,
                          std::as_writable_bytes(std::span(restored)));
        check(expected == restored, "Diagnostic upload/download round-trip mismatch");
    }
}
} // namespace

int main(int argc, char** argv) {
    std::unique_ptr<TrainingDiagnosticReport> report;
    try {
        std::filesystem::path path;
        std::string gpu;
        bool requireTransferOnly = false;
        for (int i = 1; i < argc; ++i) {
            const std::string_view arg(argv[i]);
            if (arg == "--help") {
                std::cout << "training_diagnostic_transfer_gpu_tests --report NEW.json "
                             "[--gpu NAME_SUBSTRING] [--require-transfer-only]\n";
                return 0;
            }
            if (arg == "--require-transfer-only")
                requireTransferOnly = true;
            else if (arg == "--report" && i + 1 < argc)
                path = argv[++i];
            else if (arg == "--gpu" && i + 1 < argc)
                gpu = argv[++i];
            else
                throw std::runtime_error("Unknown or incomplete argument");
        }
        check(!path.empty(), "Use --report NEW.json");
        report = std::make_unique<TrainingDiagnosticReport>(path);
        VULKAN_HPP_DEFAULT_DISPATCHER.init();
        const char* layer = "VK_LAYER_KHRONOS_validation";
        bool available = false, sync = false;
        for (const auto& l : vk::enumerateInstanceLayerProperties())
            available |= std::string_view(l.layerName.data()) == layer;
        check(available, "Required validation layer unavailable");
        for (const auto& e : vk::enumerateInstanceExtensionProperties(std::string(layer)))
            sync |= std::string_view(e.extensionName.data()) ==
                    VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME;
        check(sync, "Required synchronization validation unavailable");
        const std::array<const char*, 2> extensions{VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
                                                    VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME};
        const auto feature = vk::ValidationFeatureEnableEXT::eSynchronizationValidation;
        vk::ValidationFeaturesEXT features;
        features.setEnabledValidationFeatures(feature);
        vk::ApplicationInfo app{"diagnostic-transfer-regression", 1, "vulkan-3dgs", 1,
                                VK_API_VERSION_1_2};
        vk::InstanceCreateInfo info;
        info.setPApplicationInfo(&app)
            .setPEnabledLayerNames(layer)
            .setPEnabledExtensionNames(extensions)
            .setPNext(&features);
        auto instance = vk::createInstanceUnique(info);
        VULKAN_HPP_DEFAULT_DISPATCHER.init(*instance);
        vk::DebugUtilsMessengerCreateInfoEXT debug;
        debug.setMessageSeverity(vk::DebugUtilsMessageSeverityFlagBitsEXT::eError)
            .setMessageType(vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
                            vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral);
        debug.pfnUserCallback = message;
        auto messenger = instance->createDebugUtilsMessengerEXTUnique(debug);
        vk::PhysicalDevice physical;
        uint32_t matches = 0;
        for (auto candidate : instance->enumeratePhysicalDevices()) {
            const auto properties = candidate.getProperties();
            if (std::string_view(properties.deviceName.data()).find(gpu) == std::string_view::npos)
                continue;
            physical = candidate;
            ++matches;
        }
        check(matches == 1, "Select exactly one physical device with --gpu");
        const auto families = physical.getQueueFamilyProperties();
        uint32_t computeFamily = uint32_t(families.size());
        bool dedicated = false;
        std::vector<uint32_t> tested;
        for (uint32_t i = 0; i < families.size(); ++i) {
            const auto flags = families[i].queueFlags;
            if (flags & vk::QueueFlagBits::eCompute) computeFamily = i;
            if (flags & (vk::QueueFlagBits::eTransfer | vk::QueueFlagBits::eCompute |
                         vk::QueueFlagBits::eGraphics))
                tested.push_back(i);
            dedicated |= transferOnly(flags);
        }
        check(computeFamily < families.size(), "Compute queue unavailable");
        check(!requireTransferOnly || dedicated, "Required transfer-only queue unavailable");
        const float priority = 1;
        std::vector<vk::DeviceQueueCreateInfo> queues;
        for (auto family : tested)
            queues.emplace_back(vk::DeviceQueueCreateFlags{}, family, 1, &priority);
        vk::DeviceCreateInfo deviceInfo;
        deviceInfo.setQueueCreateInfos(queues);
        auto device = physical.createDeviceUnique(deviceInfo);
        VULKAN_HPP_DEFAULT_DISPATCHER.init(*device);
        for (auto family : tested)
            exercise(*device, physical, computeFamily, family);
        check(errors == 0, "Diagnostic transfer core/synchronization validation failed");
        std::ostringstream out;
        out << "{\"schema\":\"diagnostic_transfer_gpu_v1\",\"status\":\"PASS\",\"device\":"
            << schedulingJsonQuote(physical.getProperties().deviceName.data())
            << ",\"core_validation_enabled\":true,\"synchronization_validation_enabled\":true"
            << ",\"validation_errors\":" << errors.load()
            << ",\"transfer_only_covered\":" << (dedicated ? "true" : "false")
            << ",\"compute_family\":" << computeFamily << ",\"tested_queue_families\":[";
        for (size_t i = 0; i < tested.size(); ++i) {
            if (i) out << ',';
            out << "{\"index\":" << tested[i]
                << ",\"flags\":" << uint32_t(VkQueueFlags(families[tested[i]].queueFlags)) << '}';
        }
        out << "],\"window_created\":false,\"training_advanced\":false}\n";
        report->write(out.str());
        std::cout << "Diagnostic transfer: content matched; zero validation errors; transfer-only="
                  << dedicated << '\n';
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        if (report && !report->written()) try {
                report->write("{\"status\":\"STOP\",\"error\":" + schedulingJsonQuote(e.what()) +
                              ",\"validation_errors\":" + std::to_string(errors.load()) + "}\n");
            } catch (...) {}
        return 1;
    }
}
