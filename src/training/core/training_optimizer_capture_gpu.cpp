#include "training/core/training_optimizer_capture_gpu.hpp"
#include "utils/system_memory.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace vulkan3DGS {
namespace {
void visibility(vk::CommandBuffer command, vk::AccessFlags destination) {
    // ALL_COMMANDS covers only stages supported by this queue, including transfer-only queues.
    vk::MemoryBarrier barrier{vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
                              destination};
    command.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands |
                                vk::PipelineStageFlagBits::eHost,
                            vk::PipelineStageFlagBits::eAllCommands, {}, barrier, {}, {});
}
} // namespace
OptimizerCaptureTransfer::OptimizerCaptureTransfer(vk::Device device, vk::PhysicalDevice physical,
                                                   vk::Queue queue, uint32_t family)
    : device_(device), queue_(queue) {
    staging_.create(device, physical, queue, family, nullptr, optimizerCaptureChunkBytes,
                    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
                    vk::MemoryPropertyFlagBits::eHostVisible |
                        vk::MemoryPropertyFlagBits::eHostCoherent,
                    {family});
    pool_.create(device, family, vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
    command_ = pool_.allocateCommandBuffer();
    fence_ = device.createFenceUnique({});
}
void OptimizerCaptureTransfer::download(vk::DescriptorBufferInfo source, uint64_t offset,
                                        std::span<std::byte> bytes) {
    if (bytes.empty()) return;
    if (bytes.size() > optimizerCaptureChunkBytes || offset > source.range ||
        bytes.size() > source.range - offset)
        throw std::runtime_error("Optimizer download bounds");
    command_.reset();
    command_.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    visibility(command_, vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite);
    command_.copyBuffer(source.buffer, staging_.getBuffer(),
                        vk::BufferCopy{source.offset + offset, 0, bytes.size()});
    vk::MemoryBarrier host{vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eHostRead};
    command_.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost,
                             {}, host, {}, {});
    command_.end();
    device_.resetFences(*fence_);
    vk::SubmitInfo submit;
    submit.setCommandBuffers(command_);
    queue_.submit(submit, *fence_);
    (void)device_.waitForFences(*fence_, true, UINT64_MAX);
    staging_.download(bytes.data(), bytes.size());
}
void OptimizerCaptureTransfer::upload(vk::DescriptorBufferInfo destination, uint64_t offset,
                                      std::span<const std::byte> bytes) {
    if (bytes.empty()) return;
    if (bytes.size() > optimizerCaptureChunkBytes || offset > destination.range ||
        bytes.size() > destination.range - offset)
        throw std::runtime_error("Optimizer upload bounds");
    staging_.upload(bytes.data(), bytes.size());
    command_.reset();
    command_.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    visibility(command_, vk::AccessFlagBits::eTransferWrite | vk::AccessFlagBits::eTransferRead);
    command_.copyBuffer(staging_.getBuffer(), destination.buffer,
                        vk::BufferCopy{0, destination.offset + offset, bytes.size()});
    visibility(command_, vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite);
    command_.end();
    device_.resetFences(*fence_);
    vk::SubmitInfo submit;
    submit.setCommandBuffers(command_);
    queue_.submit(submit, *fence_);
    (void)device_.waitForFences(*fence_, true, UINT64_MAX);
}
void checkOptimizerCaptureBudget(vk::PhysicalDevice physical, uint32_t count, uint64_t deviceBytes,
                                 const std::filesystem::path& output) {
    const auto limits = physical.getProperties().limits;
    if (uint64_t(count) * 480 > limits.maxStorageBufferRange ||
        limits.maxComputeWorkGroupInvocations < 256 || limits.maxComputeWorkGroupSize[0] < 256 ||
        (uint64_t(count) + 255) / 256 > limits.maxComputeWorkGroupCount[0])
        throw std::runtime_error("Optimizer capture exceeds original fused device limits");
    if (optimizerCaptureBytes(count) > optimizerCaptureFileBudget)
        throw std::runtime_error("Optimizer capture exceeds 8 GiB file budget");
    const uint64_t ram = availableSystemMemoryBytes();
    if (ram < (128ull << 20))
        throw std::runtime_error("Optimizer capture cannot verify 128 MiB RAM headroom");
    bool memoryBudget = false;
    for (const auto& ext : physical.enumerateDeviceExtensionProperties())
        memoryBudget |=
            std::strcmp(ext.extensionName.data(), VK_EXT_MEMORY_BUDGET_EXTENSION_NAME) == 0;
    if (!memoryBudget)
        throw std::runtime_error(
            "Optimizer capture requires measurable VK_EXT_memory_budget; unavailable");
    vk::PhysicalDeviceMemoryBudgetPropertiesEXT budget;
    vk::PhysicalDeviceMemoryProperties2 memory;
    memory.pNext = &budget;
    physical.getMemoryProperties2(&memory);
    uint64_t available = 0;
    for (uint32_t i = 0; i < memory.memoryProperties.memoryHeapCount; ++i) {
        if (memory.memoryProperties.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal)
            available = std::max(available, budget.heapBudget[i] > budget.heapUsage[i]
                                                ? budget.heapBudget[i] - budget.heapUsage[i]
                                                : 0);
    }
    if (available < deviceBytes + (256ull << 20))
        throw std::runtime_error(
            "Optimizer capture insufficient measured VRAM headroom (plus 256 MiB reserve)");
    if (!output.empty()) {
        if (std::filesystem::exists(output))
            throw std::runtime_error("Optimizer capture output already exists");
        auto manifest = output;
        manifest.replace_extension(".json");
        if (std::filesystem::exists(manifest))
            throw std::runtime_error("Optimizer capture manifest already exists");
        auto parent = output.parent_path();
        if (parent.empty()) parent = ".";
        std::filesystem::create_directories(parent);
        if (std::filesystem::space(parent).available < optimizerCaptureBytes(count) + (64ull << 20))
            throw std::runtime_error("Optimizer capture insufficient disk space");
    }
}
TrainingOptimizerCaptureSession::TrainingOptimizerCaptureSession(vk::Device device,
                                                                 vk::PhysicalDevice physical,
                                                                 vk::Queue queue, uint32_t family,
                                                                 OptimizerCaptureHeader header,
                                                                 const std::filesystem::path& path)
    : header_(header), path_(path), transfer_(device, physical, queue, family) {
    checkOptimizerCaptureBudget(physical, header.gaussianCount,
                                uint64_t(header.gaussianCount) * 840, path);
    for (size_t i = 0; i < pre_.size(); ++i)
        pre_[i].create(device, physical, queue, family, nullptr,
                       std::max<uint64_t>(16, uint64_t(header.gaussianCount) * header.strides[i]),
                       vk::BufferUsageFlagBits::eTransferSrc |
                           vk::BufferUsageFlagBits::eTransferDst,
                       vk::MemoryPropertyFlagBits::eDeviceLocal, {family});
}
void TrainingOptimizerCaptureSession::recordPre(
    vk::CommandBuffer command, const std::array<vk::DescriptorBufferInfo, 7>& sources,
    const TrainingPushConstants& push, const TrainingForwardCamera& camera,
    uint64_t dispatchedHash) {
    if (recorded_ || push.gaussianCount != header_.gaussianCount || !push.optimizerEnabled ||
        dispatchedHash != header_.shaderHash)
        throw std::runtime_error("Optimizer capture hook mismatch/reuse");
    header_.push = push;
    header_.camera = camera;
    header_.completedIteration = push.trainingIteration + 1;
    header_.completedOptimizerStep = push.optimizerStep;
    visibility(command, vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite);
    for (size_t i = 0; i < pre_.size(); ++i) {
        const uint64_t bytes = uint64_t(push.gaussianCount) * header_.strides[i];
        if (sources[i].range < bytes)
            throw std::runtime_error("Optimizer capture source capacity mismatch");
        if (bytes)
            command.copyBuffer(sources[i].buffer, pre_[i].getBuffer(),
                               vk::BufferCopy{sources[i].offset, 0, bytes});
    }
    // Transfer reads finish before the original shader can overwrite the pre-state.
    visibility(command, vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
    recorded_ = true;
}
void TrainingOptimizerCaptureSession::save(const std::array<vk::DescriptorBufferInfo, 3>& post) {
    if (!recorded_ || !completed_)
        throw std::runtime_error("Optimizer step not completed; refusing post-reference export");
    writeOptimizerCapture(
        path_, header_, [&](size_t block, uint64_t offset, std::span<std::byte> bytes) {
            transfer_.download(block < 6 ? pre_[block].getDescriptorInfo() : post[block - 6],
                               offset, bytes);
        });
    // The binary is authoritative. A small companion makes the capture's actual
    // path and explicit lack of native resource evidence visible without allocating it.
    const auto verified = readOptimizerCaptureHeader(path_, false);
    auto manifestPath = path_;
    manifestPath.replace_extension(".json");
    FILE* raw = nullptr;
#ifdef _WIN32
    _wfopen_s(&raw, manifestPath.c_str(), L"wbx");
#else
    raw = std::fopen(manifestPath.c_str(), "wbx");
#endif
    if (!raw)
        throw std::runtime_error(
            "Optimizer binary saved, but companion manifest exists/cannot be created");
    std::unique_ptr<FILE, decltype(&std::fclose)> file(raw, &std::fclose);
    std::ostringstream text;
    text
        << '{' << optimizerCaptureIdentityJson(verified)
        << ",\"format\":\"V3DGOPT1\",\"version\":1,\"header_bytes\":" << sizeof(verified)
        << ",\"file_bytes\":" << optimizerCaptureBytes(verified.gaussianCount)
        << ",\"gaussians\":" << verified.gaussianCount << ",\"capacity\":" << verified.capacity
        << ",\"completed_iteration\":" << verified.completedIteration
        << ",\"optimizer_step\":" << verified.completedOptimizerStep
        << ",\"frame\":" << verified.frame << ",\"extent\":[" << verified.push.width << ','
        << verified.push.height << ']' << ",\"active_SH_degree\":" << verified.push.activeSHDegree
        << ",\"pixel_mode\":" << verified.pixelMode << ",\"ssim_mode\":" << verified.ssimMode
        << ",\"tile_mode\":" << verified.tileMode
        << ",\"composite_mode\":" << verified.compositeMode
        << ",\"tensor_flags\":" << verified.tensorFlags
        << ",\"ordinary_fused\":true,\"optimizer_enabled\":true,\"ADC\":false,\"opacity_reset\":"
           "false"
        << ",\"validation_enabled_on_step\":"
        << (verified.push.validationEnabled ? "true" : "false")
        << ",\"step_mutated_and_completed\":true,\"diagnostic_step_excluded_from_performance\":true"
        << ",\"native\":null,\"recommendation\":\"INSUFFICIENT_EVIDENCE\",\"blocks\":[";
    for (size_t i = 0; i < verified.blocks.size(); ++i) {
        if (i) text << ',';
        text << "{\"stride\":" << verified.strides[i] << ",\"bytes\":" << verified.blocks[i].bytes
             << ",\"fnv1a64\":\"" << verified.blocks[i].checksum << "\"}";
    }
    text << "]}\n";
    const auto json = text.str();
    if (std::fwrite(json.data(), 1, json.size(), raw) != json.size() || std::fflush(raw) != 0)
        throw std::runtime_error("Optimizer binary saved, but manifest write failed");
}
} // namespace vulkan3DGS
