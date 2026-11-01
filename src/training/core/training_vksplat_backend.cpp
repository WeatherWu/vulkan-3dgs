#include "training/core/training_vksplat_backend.hpp"

#include "utils/file_utils.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace vulkan3DGS {
namespace {
constexpr std::array<uint32_t, 10> nativeBindings{66, 67, 68, 69, 70, 71, 72, 73, 74, 76};
constexpr std::array<const char*, 19> shaderNames{"train_vksplat_project",
                                                  "train_vksplat_optimizer",
                                                  "train_vksplat_state_import",
                                                  "train_vksplat_state_export",
                                                  "train_vksplat_composite",
                                                  "train_vksplat_composite_subtile",
                                                  "train_vksplat_tensor_shared",
                                                  "train_vksplat_tensor_global",
                                                  "train_vksplat_validation",
                                                  "train_vksplat_composite_64",
                                                  "train_vksplat_composite_subtile_64",
                                                  "train_vksplat_project_soa",
                                                  "train_vksplat_composite_soa",
                                                  "train_vksplat_composite_subtile_soa",
                                                  "train_vksplat_tensor_shared_soa",
                                                  "train_vksplat_tensor_global_soa",
                                                  "train_vksplat_composite_64_soa",
                                                  "train_vksplat_composite_subtile_64_soa",
                                                  "train_vksplat_legacy_emit_soa"};
std::vector<uint32_t> bindingsFor(uint32_t stage) {
    if (stage == 11) {
        std::vector<uint32_t> b(nativeBindings.begin(), nativeBindings.begin() + 4);
        b.insert(b.end(), {11, 64, 80, 81, 82, 83, 84, 85, 86});
        return b;
    }
    if (stage >= 12 && stage <= 17) {
        if (stage == 14 || stage == 15) return {4, 5, 12, 13, 14, 80, 81, 82, 83, 84, 85, 86};
        return {4, 5, 6, 14, 80, 81, 82, 83, 84, 85, 86};
    }
    if (stage == 18) return {4, 20, 21, 22, 32, 80, 81, 82, 83, 84, 85, 86};
    if (stage == 4 || stage == 5 || stage == 9 || stage == 10) return {3, 4, 5, 6, 14};
    if (stage == 6 || stage == 7) return {3, 4, 5, 12, 13, 14};
    std::vector<uint32_t> result(nativeBindings.begin(), nativeBindings.end());
    const std::array<std::vector<uint32_t>, 4> extras{
        {{3, 32, 75, 77, 11, 64}, {13, 32, 11, 64}, {0, 2, 16, 78, 79}, {0, 2, 16, 78, 79}}};
    if (stage < 4)
        result.insert(result.end(), extras[stage].begin(), extras[stage].end());
    else
        result.insert(result.end(), {19, 30});
    return result;
}
} // namespace

TrainingVkSplatBackend::TrainingVkSplatBackend(vk::Device device, vk::PhysicalDevice physical,
                                               vk::Queue queue, uint32_t family, bool adcReuse,
                                               bool projectionSoA)
    : device_(device), physical_(physical), queue_(queue), family_(family),
      projectionSoA_(projectionSoA), optimizedRecording_(projectionSoA) {
    vk::PhysicalDeviceSubgroupProperties subgroup;
    vk::PhysicalDeviceProperties2 properties;
    properties.pNext = &subgroup;
    physical_.getProperties2(&properties);
    subgroupSize_ = subgroup.subgroupSize;
    pixelSupported_ = subgroupSize_ == 32u &&
                      properties.properties.limits.maxComputeSharedMemorySize >= 45u * 1024u;
    pool_.create(device_, family_, vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
    command_ = pool_.allocateCommandBuffer();
    fence_ = device_.createFenceUnique({});
    adc_ = std::make_unique<TrainingVkSplatAdc>(*this, adcReuse);
}

TrainingVkSplatBackend::~TrainingVkSplatBackend() = default;

void TrainingVkSplatBackend::ensureCapacity(uint32_t count) {
    if (count <= capacity_) return;
    const uint32_t capacity = (count + 31u) / 32u * 32u;
    const std::array<uint64_t, 10> stride{12, 192, 16, 16, 24, 192, 192, 32, 32, 12};
    std::array<Buffer, 10> replacement;
    auto allocate = [&](Buffer& buffer, uint64_t bytes) {
        if (bytes > physical_.getProperties().limits.maxStorageBufferRange)
            throw std::runtime_error("VkSplat model field exceeds maxStorageBufferRange");
        buffer.create(device_, physical_, queue_, family_, nullptr, bytes,
                      vk::BufferUsageFlagBits::eStorageBuffer |
                          vk::BufferUsageFlagBits::eTransferSrc |
                          vk::BufferUsageFlagBits::eTransferDst,
                      vk::MemoryPropertyFlagBits::eDeviceLocal, {family_});
    };
    for (size_t i = 0; i < replacement.size(); ++i)
        allocate(replacement[i], stride[i] * capacity);
    Buffer rectangles;
    allocate(rectangles, uint64_t(capacity) * 16u);
    Buffer radii;
    allocate(radii, uint64_t(capacity) * 4u);
    adc_->ensureStatistics(capacity);
    officialRadii_ = std::move(radii);
    model_ = std::move(replacement);
    rectangles_ = std::move(rectangles);
    capacity_ = capacity;
}

void TrainingVkSplatBackend::initializePipeline(Stage stage) {
    if (sets_[stage]) return;
    if (optimizedRecording_) cachedBindings_[stage] = bindingsFor(stage);
    ComputePipelineConfig config{};
    uint32_t storage = 0, uniforms = 0;
    for (auto binding : bindingsFor(stage)) {
        const bool uniform = binding == 11 || binding == 64;
        config.descriptorBindings.push_back(uniform ? backwardUniformBinding(binding)
                                                    : backwardStorageBinding(binding));
        uniform ? ++uniforms : ++storage;
    }
    config.pushConstantSize = sizeof(TrainingPushConstants);
    pipelines_[stage].initialize(
        device_, FileUtils::resolvePath(std::string("shaders/") + shaderNames[stage] + ".comp.spv"),
        config);
    sets_[stage] = descriptors_[stage].create(device_, storage, uniforms,
                                              {pipelines_[stage].getDescriptorSetLayout()})[0];
}

void TrainingVkSplatBackend::record(Stage stage, const BackwardPassContext& context) {
    initializePipeline(stage);
    const auto referenceBindings =
        optimizedRecording_ ? std::vector<uint32_t>{} : bindingsFor(stage);
    const auto& bindings = optimizedRecording_ ? cachedBindings_[stage] : referenceBindings;
    std::vector<vk::DescriptorBufferInfo> referenceInfos;
    auto& infos = optimizedRecording_ ? cachedInfos_[stage] : referenceInfos;
    infos.clear();
    const auto& b = context.buffers;
    for (auto binding : bindings) {
        if (binding >= 80 && binding <= 86) {
            infos.push_back(soa_[binding - 80].getDescriptorInfo());
            continue;
        }
        auto native = std::find(nativeBindings.begin(), nativeBindings.end(), binding);
        if (native != nativeBindings.end()) {
            infos.push_back(model_[native - nativeBindings.begin()].getDescriptorInfo());
            continue;
        }
        switch (binding) {
            case 0:
                infos.push_back(b.gaussianParamsInfo());
                break;
            case 2:
                infos.push_back(b.adamStatesInfo());
                break;
            case 3:
                infos.push_back(b.projectedInfo());
                break;
            case 4:
                infos.push_back(b.tileItemsInfo());
                break;
            case 5:
                infos.push_back(b.tileRangesInfo());
                break;
            case 6:
                infos.push_back(b.renderedColorInfo());
                break;
            case 11:
                infos.push_back(b.cameraInfo());
                break;
            case 12:
                infos.push_back(b.pixelGradsInfo());
                break;
            case 13:
                infos.push_back(b.projectedGradsInfo());
                break;
            case 14:
                infos.push_back(b.pixelBlendStatesInfo());
                break;
            case 16:
                infos.push_back(b.densificationStatesInfo());
                break;
            case 19:
                infos.push_back(b.countersInfo());
                break;
            case 30:
                infos.push_back(b.gaussianValidationPartialsInfo());
                break;
            case 20:
                infos.push_back(b.tileKeyLowInfo());
                break;
            case 21:
                infos.push_back(b.tileKeyHighInfo());
                break;
            case 22:
                infos.push_back(b.tileSortIndicesInfo());
                break;
            case 32:
                infos.push_back(b.gaussianTileRangesInfo());
                break;
            case 64:
                infos.push_back(b.cameraIntrinsicsInfo());
                break;
            case 75:
                infos.push_back(rectangles());
                break;
            case 77:
                infos.push_back(officialRadii_.getDescriptorInfo());
                break;
            case 78:
                infos.push_back(adc_->gradients());
                break;
            case 79:
                infos.push_back(adc_->radii());
                break;
            default:
                throw std::logic_error("Unknown VkSplat binding");
        }
    }
    std::vector<vk::WriteDescriptorSet> referenceWrites;
    auto& writes = optimizedRecording_ ? cachedWrites_[stage] : referenceWrites;
    writes.resize(bindings.size());
    for (size_t i = 0; i < bindings.size(); ++i)
        writes[i]
            .setDstSet(sets_[stage])
            .setDstBinding(bindings[i])
            .setDescriptorCount(1)
            .setDescriptorType(bindings[i] == 11 || bindings[i] == 64
                                   ? vk::DescriptorType::eUniformBuffer
                                   : vk::DescriptorType::eStorageBuffer)
            .setPBufferInfo(&infos[i]);
    if (lastBindings_[stage] != infos) {
        device_.updateDescriptorSets(writes, {});
        lastBindings_[stage] = infos;
    }
    const auto layout = pipelines_[stage].getPipelineLayout();
    context.commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute,
                                       pipelines_[stage].getPipeline());
    context.commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, layout, 0,
                                             sets_[stage], {});
    context.commandBuffer.pushConstants(layout, vk::ShaderStageFlagBits::eCompute, 0,
                                        sizeof(TrainingPushConstants), &context.pushConstants);
    const auto& p = context.pushConstants;
    if ((stage >= Composite && stage <= PixelGlobal) || stage == Composite64 ||
        stage == Subtile64 || (stage >= CompositeSoA && stage <= Subtile64SoA))
        context.commandBuffer.dispatch((p.width + 15) / 16, (p.height + 15) / 16, 1);
    else if (p.gaussianCount)
        context.commandBuffer.dispatch(
            (p.gaussianCount + (stage <= Optimize || stage == ProjectSoA ? 31u : 255u)) /
                (stage <= Optimize || stage == ProjectSoA ? 32u : 256u),
            1, 1);
    vk::MemoryBarrier barrier{vk::AccessFlagBits::eShaderWrite,
                              vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite |
                                  vk::AccessFlagBits::eTransferRead};
    context.commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                          vk::PipelineStageFlagBits::eComputeShader |
                                              vk::PipelineStageFlagBits::eTransfer,
                                          {}, barrier, {}, {});
}

void TrainingVkSplatBackend::boundary(Stage stage, TrainingBuffers& buffers, uint32_t count) {
    if (adc_->failed())
        throw std::runtime_error("VkSplat native state requires explicit dataset reload");
    const auto started = std::chrono::steady_clock::now();
    if (stage == Import)
        ensureCapacity(std::max(count, 1u));
    else
        buffers.ensureCanonicalModelStorage();
    TrainingPushConstants push{};
    push.gaussianCount = count;
    command_.reset();
    command_.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    vk::MemoryBarrier barrier{vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
                              vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite};
    command_.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                             vk::PipelineStageFlagBits::eComputeShader, {}, barrier, {}, {});
    record(stage, BackwardPassContext{buffers, command_, push});
    command_.end();
    queue_.submit(vk::SubmitInfo{}.setCommandBuffers(command_), *fence_);
    (void)device_.waitForFences(*fence_, true, UINT64_MAX);
    device_.resetFences(*fence_);
    if (stage == Import) buffers.releaseCanonicalModelStorage();
    boundarySeconds_ +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
}

void TrainingVkSplatBackend::importCanonical(TrainingBuffers& b, uint32_t count) {
    boundary(Import, b, count);
}
void TrainingVkSplatBackend::exportCanonical(TrainingBuffers& b, uint32_t count) {
    boundary(Export, b, count);
}
void TrainingVkSplatBackend::recordProject(const BackwardPassContext& c) {
    if (!c.pushConstants.width || !c.pushConstants.height || c.pushConstants.width > 65535u ||
        c.pushConstants.height > 65535u)
        throw std::runtime_error(
            "VkSplat image dimensions must fit the official packed image-size ABI");
    if (c.buffers.uploadedCamera().model != glm::mat4(1.0f))
        throw std::runtime_error(
            "VkSplat backend expects world-space parameters and an identity model transform");
    if (!c.buffers.hasExactIntrinsics())
        throw std::runtime_error("VkSplat projection requires exact intrinsics");
    if (projectionSoA_) {
        if (std::max(c.pushConstants.gaussianCount, 1u) > soaCapacity_) {
            std::array<Buffer, 7> next;
            const uint32_t cap = (std::max(c.pushConstants.gaussianCount, 1u) + 31u) / 32u * 32u;
            constexpr std::array<uint32_t, 7> stride{8, 4, 16, 12, 8, 4, 4};
            for (uint32_t i = 0; i < 7; ++i) {
                const uint64_t bytes = uint64_t(cap) * stride[i];
                if (bytes > physical_.getProperties().limits.maxStorageBufferRange)
                    throw std::runtime_error("VkSplat SoA output exceeds buffer limits");
                next[i].create(device_, physical_, queue_, family_, nullptr, bytes,
                               vk::BufferUsageFlagBits::eStorageBuffer |
                                   vk::BufferUsageFlagBits::eTransferSrc |
                                   vk::BufferUsageFlagBits::eTransferDst,
                               vk::MemoryPropertyFlagBits::eDeviceLocal, {family_});
            }
            soa_ = std::move(next);
            soaCapacity_ = cap;
        }
        record(ProjectSoA, c);
    } else
        record(Project, c);
}
void TrainingVkSplatBackend::recordComposite(const BackwardPassContext& c, bool subtile) {
    if (subgroupSize_ != 32u && subgroupSize_ != 64u)
        throw std::runtime_error("VkSplat composite requires subgroup size 32 or 64");
    if (projectionSoA_)
        record(subgroupSize_ == 64u ? (subtile ? Subtile64SoA : Composite64SoA)
                                    : (subtile ? SubtileSoA : CompositeSoA),
               c);
    else
        record(subgroupSize_ == 64u ? (subtile ? Subtile64 : Composite64)
                                    : (subtile ? Subtile : Composite),
               c);
}
bool TrainingVkSplatBackend::supportsLifecycle(vk::PhysicalDevice physical) {
    vk::PhysicalDeviceSubgroupProperties subgroup;
    vk::PhysicalDeviceProperties2 properties;
    properties.pNext = &subgroup;
    physical.getProperties2(&properties);
    return subgroup.subgroupSize == 32u &&
           properties.properties.limits.maxComputeSharedMemorySize >= 45u * 1024u;
}
void TrainingVkSplatBackend::recordPixel(const BackwardPassContext& c, bool global) {
    if (!pixelSupported_)
        throw std::runtime_error(
            "VkSplat Tensor requires native subgroup32 and at least 45 KiB shared memory");
    record(projectionSoA_ ? (global ? PixelGlobalSoA : PixelSharedSoA)
                          : (global ? PixelGlobal : PixelShared),
           c);
}
void TrainingVkSplatBackend::recordOptimizer(const BackwardPassContext& c) {
    record(Optimize, c);
    adc_->recordStatistics(c);
}
void TrainingVkSplatBackend::recordValidation(const BackwardPassContext& c) {
    record(Validate, c);
}
std::array<vk::DescriptorBufferInfo, 7> TrainingVkSplatBackend::projectionInfos() const {
    std::array<vk::DescriptorBufferInfo, 7> result;
    for (uint32_t i = 0; i < 7; ++i)
        result[i] = soa_[i].getDescriptorInfo();
    return result;
}
void TrainingVkSplatBackend::recordSoALegacyEmit(const BackwardPassContext& c) {
    record(LegacyEmitSoA, c);
}
void TrainingVkSplatBackend::recordSoAPrefix(const BackwardPassContext& c,
                                             const TrainingPrefixPushConstants& prefix,
                                             uint32_t groups) {
    if (!soaPrefixSet_) {
        ComputePipelineConfig config;
        config.pushConstantSize = sizeof(prefix);
        for (uint32_t b : {32u, 33u, 86u})
            config.descriptorBindings.push_back(backwardStorageBinding(b));
        soaPrefixPipeline_.initialize(device_, "shaders/train_vksplat_prefix_soa.comp.spv", config);
        soaPrefixSet_ = soaPrefixDescriptors_.create(
            device_, 3, 0, {soaPrefixPipeline_.getDescriptorSetLayout()})[0];
    }
    std::array<vk::DescriptorBufferInfo, 3> info{c.buffers.gaussianTileRangesInfo(),
                                                 c.buffers.gaussianTilePrefixScratchInfo(),
                                                 soa_[6].getDescriptorInfo()};
    std::array<vk::WriteDescriptorSet, 3> writes;
    constexpr std::array<uint32_t, 3> bindings{32, 33, 86};
    for (uint32_t i = 0; i < 3; ++i)
        writes[i] = {soaPrefixSet_, bindings[i], 0, 1, vk::DescriptorType::eStorageBuffer,
                     nullptr,       &info[i]};
    device_.updateDescriptorSets(writes, {});
    auto layout = soaPrefixPipeline_.getPipelineLayout();
    c.commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, soaPrefixPipeline_.getPipeline());
    c.commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, layout, 0, soaPrefixSet_,
                                       {});
    c.commandBuffer.pushConstants(layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(prefix),
                                  &prefix);
    if (groups) c.commandBuffer.dispatch(groups, 1, 1);
}
uint64_t TrainingVkSplatBackend::allocatedBytes() const {
    uint64_t bytes = rectangles_.getSize() + officialRadii_.getSize() + adc_->allocatedBytes();
    for (const auto& b : soa_)
        bytes += b.getSize();
    for (const auto& b : model_)
        bytes += b.getSize();
    return bytes;
}

} // namespace vulkan3DGS
