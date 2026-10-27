#include "graphics/renderpass.hpp"
#include "training/core/training_scheduling_diagnostics.hpp"
#include <array>
#include <atomic>
#include <cstring>
#include <iostream>
#include <sstream>

using namespace vulkan3DGS;
namespace {
std::atomic<uint32_t> phase{0};
std::array<std::atomic<uint32_t>, 2> errors{}, depthHazards{};
VKAPI_ATTR vk::Bool32 VKAPI_CALL message(vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
                                         vk::DebugUtilsMessageTypeFlagsEXT,
                                         const vk::DebugUtilsMessengerCallbackDataEXT* data,
                                         void*) {
    if (severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eError) {
        const uint32_t index = phase.load();
        ++errors[index];
        const std::string_view text(data->pMessage);
        if ((text.find("WRITE_AFTER_WRITE") != text.npos ||
             text.find("WRITE-AFTER-WRITE") != text.npos) &&
            text.find("DEPTH_STENCIL_ATTACHMENT_WRITE") != text.npos)
            ++depthHazards[index];
        if (index == 1) std::cerr << "Fixed render-pass validation: " << data->pMessage << '\n';
    }
    return VK_FALSE;
}
void check(bool okay, const char* reason) {
    if (!okay) throw std::runtime_error(reason);
}
struct Attachment {
    vk::UniqueDeviceMemory memory;
    vk::UniqueImage image;
    vk::UniqueImageView view;
};
Attachment attachment(vk::Device device, vk::PhysicalDevice physical, vk::Format format,
                      bool depth) {
    Attachment out;
    vk::ImageCreateInfo info;
    info.setImageType(vk::ImageType::e2D)
        .setFormat(format)
        .setExtent({16, 16, 1})
        .setMipLevels(1)
        .setArrayLayers(1)
        .setSamples(vk::SampleCountFlagBits::e1)
        .setTiling(vk::ImageTiling::eOptimal)
        .setInitialLayout(vk::ImageLayout::eUndefined)
        .setUsage(depth ? vk::ImageUsageFlagBits::eDepthStencilAttachment
                        : vk::ImageUsageFlagBits::eColorAttachment);
    out.image = device.createImageUnique(info);
    const auto requirements = device.getImageMemoryRequirements(*out.image);
    const auto memory = physical.getMemoryProperties();
    uint32_t type = 0;
    while (type < memory.memoryTypeCount &&
           (!(requirements.memoryTypeBits & (1u << type)) ||
            !(memory.memoryTypes[type].propertyFlags & vk::MemoryPropertyFlagBits::eDeviceLocal)))
        ++type;
    check(type < memory.memoryTypeCount, "Device-local image memory unavailable");
    out.memory = device.allocateMemoryUnique({requirements.size, type});
    device.bindImageMemory(*out.image, *out.memory, 0);
    vk::ImageViewCreateInfo view;
    view.setImage(*out.image)
        .setViewType(vk::ImageViewType::e2D)
        .setFormat(format)
        .setSubresourceRange(
            {depth ? vk::ImageAspectFlagBits::eDepth : vk::ImageAspectFlagBits::eColor, 0, 1, 0,
             1});
    out.view = device.createImageViewUnique(view);
    return out;
}
// Negative control deliberately omits late-depth-store synchronization; renderer uses its own pass.
vk::UniqueRenderPass legacyPass(vk::Device device) {
    std::array<vk::AttachmentDescription, 2> attachments;
    for (size_t i = 0; i < 2; ++i)
        attachments[i]
            .setFormat(i ? RenderPass::DepthFormat : vk::Format::eR8G8B8A8Unorm)
            .setSamples(vk::SampleCountFlagBits::e1)
            .setLoadOp(vk::AttachmentLoadOp::eClear)
            .setStoreOp(i ? vk::AttachmentStoreOp::eDontCare : vk::AttachmentStoreOp::eStore)
            .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
            .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
            .setInitialLayout(vk::ImageLayout::eUndefined)
            .setFinalLayout(i ? vk::ImageLayout::eDepthStencilAttachmentOptimal
                              : vk::ImageLayout::eColorAttachmentOptimal);
    vk::AttachmentReference color{0, vk::ImageLayout::eColorAttachmentOptimal},
        depth{1, vk::ImageLayout::eDepthStencilAttachmentOptimal};
    vk::SubpassDescription subpass;
    subpass.setPipelineBindPoint(vk::PipelineBindPoint::eGraphics)
        .setColorAttachments(color)
        .setPDepthStencilAttachment(&depth);
    vk::SubpassDependency dependency;
    const auto stages = vk::PipelineStageFlagBits::eColorAttachmentOutput |
                        vk::PipelineStageFlagBits::eEarlyFragmentTests;
    dependency.setSrcSubpass(vk::SubpassExternal)
        .setDstSubpass(0)
        .setSrcStageMask(stages)
        .setDstStageMask(stages)
        .setDstAccessMask(vk::AccessFlagBits::eColorAttachmentWrite |
                          vk::AccessFlagBits::eDepthStencilAttachmentWrite);
    vk::RenderPassCreateInfo info;
    info.setAttachments(attachments).setSubpasses(subpass).setDependencies(dependency);
    return device.createRenderPassUnique(info);
}
void run(vk::Device device, vk::PhysicalDevice physical, vk::Queue queue, uint32_t family,
         bool fixed) {
    phase.store(fixed ? 1u : 0u);
    RenderPass actual;
    vk::UniqueRenderPass legacy;
    vk::RenderPass pass;
    if (fixed) {
        actual.initialize(device, vk::Format::eR8G8B8A8Unorm,
                          vk::ImageLayout::eColorAttachmentOptimal);
        pass = actual.getRenderPass();
    } else {
        legacy = legacyPass(device);
        pass = *legacy;
    }
    auto depth = attachment(device, physical, RenderPass::DepthFormat, true);
    std::array<Attachment, 2> colors;
    std::array<vk::UniqueFramebuffer, 2> framebuffers;
    for (size_t i = 0; i < 2; ++i) {
        colors[i] = attachment(device, physical, vk::Format::eR8G8B8A8Unorm, false);
        const std::array<vk::ImageView, 2> views{*colors[i].view, *depth.view};
        vk::FramebufferCreateInfo info;
        info.setRenderPass(pass).setAttachments(views).setWidth(16).setHeight(16).setLayers(1);
        framebuffers[i] = device.createFramebufferUnique(info);
    }
    auto pool = device.createCommandPoolUnique({{}, family});
    const auto commands =
        device.allocateCommandBuffers({*pool, vk::CommandBufferLevel::ePrimary, 8});
    auto fence = device.createFenceUnique({});
    std::array<vk::ClearValue, 2> clear;
    clear[0].setColor(vk::ClearColorValue(std::array<float, 4>{0, 0, 0, 1}));
    clear[1].setDepthStencil({1, 0});
    for (size_t i = 0; i < commands.size(); ++i) {
        const auto command = commands[i];
        command.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        vk::RenderPassBeginInfo begin;
        begin.setRenderPass(pass)
            .setFramebuffer(*framebuffers[i % 2])
            .setRenderArea({{0, 0}, {16, 16}})
            .setClearValues(clear);
        command.beginRenderPass(begin, vk::SubpassContents::eInline);
        command.endRenderPass();
        command.end();
        vk::SubmitInfo submit;
        submit.setCommandBuffers(command);
        queue.submit(submit, i + 1 == commands.size() ? *fence : vk::Fence{});
    }
    check(device.waitForFences(*fence, true, UINT64_MAX) == vk::Result::eSuccess,
          "Render-pass test fence failed");
}
} // namespace
int main(int argc, char** argv) {
    std::unique_ptr<TrainingDiagnosticReport> report;
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--help") {
            std::cout << "graphics_depth_reuse_gpu_tests --report NEW.json\n";
            return 0;
        }
        check(argc == 3 && std::string_view(argv[1]) == "--report", "Use --report NEW.json");
        report = std::make_unique<TrainingDiagnosticReport>(argv[2]);
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
        vk::ApplicationInfo app{"depth-reuse-regression", 1, "vulkan-3dgs", 1, VK_API_VERSION_1_2};
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
        const auto physicalDevices = instance->enumeratePhysicalDevices();
        check(!physicalDevices.empty(), "No Vulkan physical device available");
        const auto physical = physicalDevices.front();
        const auto families = physical.getQueueFamilyProperties();
        uint32_t family = 0;
        while (family < families.size() &&
               !(families[family].queueFlags & vk::QueueFlagBits::eGraphics))
            ++family;
        check(family < families.size(), "Graphics queue unavailable");
        const float priority = 1;
        vk::DeviceQueueCreateInfo queueInfo{{}, family, 1, &priority};
        vk::DeviceCreateInfo deviceInfo;
        deviceInfo.setQueueCreateInfos(queueInfo);
        auto device = physical.createDeviceUnique(deviceInfo);
        VULKAN_HPP_DEFAULT_DISPATCHER.init(*device);
        const auto queue = device->getQueue(family, 0);
        run(*device, physical, queue, family, false);
        run(*device, physical, queue, family, true);
        const bool passed = depthHazards[0] > 0 && errors[1] == 0;
        std::ostringstream out;
        out << "{\"schema\":\"depth_attachment_reuse_gpu_v1\",\"status\":\""
            << (passed ? "PASS" : "STOP")
            << "\",\"device\":" << schedulingJsonQuote(physical.getProperties().deviceName.data())
            << ",\"core_validation_enabled\":true,\"synchronization_validation_enabled\":true,"
               "\"submissions_per_path\":8"
            << ",\"legacy_validation_errors_expected\":" << errors[0].load()
            << ",\"legacy_depth_waw_hazards\":" << depthHazards[0].load()
            << ",\"fixed_validation_errors\":" << errors[1].load()
            << ",\"window_created\":false,\"training_advanced\":false}\n";
        report->write(out.str());
        check(passed, "Depth reuse regression failed; see report");
        std::cout << "Depth attachment reuse: legacy hazard reproduced; fixed render pass has zero "
                     "errors\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        if (report && !report->written()) try {
                report->write("{\"status\":\"STOP\",\"error\":" + schedulingJsonQuote(e.what()) +
                              "}\n");
            } catch (...) {}
        return 1;
    }
}
