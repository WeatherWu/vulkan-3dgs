#pragma once

#include "training/core/backward/projection_optimizer_pass.hpp"
#include "training_morton_checks.hpp"
#include "utils/file_utils.hpp"
#include <glm/gtc/matrix_transform.hpp>

namespace direct_intrinsics_checks {
using namespace vulkan3DGS;
using morton_checks::read;
using morton_checks::require;
using morton_checks::upload;

inline glm::dvec2 oracle(glm::dvec3 point, const TrainingForwardCamera& camera,
                         const glm::vec4& intrinsics) {
    const auto p = glm::dmat4(camera.view) * glm::dmat4(camera.model) * glm::dvec4(point, 1);
    return {double(intrinsics.x) * p.x / p.z + double(intrinsics.z) - 0.5,
            double(intrinsics.y) * p.y / p.z + double(intrinsics.w) - 0.5};
}
struct Snapshot {
    std::vector<uint32_t> ids;
    std::vector<glm::uvec4> ranges;
    std::vector<glm::vec4> colors;
    std::vector<ProjectedGaussianGrad> gradients;
};
inline Snapshot snapshot(vk::Device device, vk::PhysicalDevice physical, vk::Queue queue,
                         uint32_t family, bool direct, const TrainingForwardCamera& camera,
                         const TrainingCameraIntrinsics& intrinsics,
                         const std::vector<GaussianTrainParam>& params) {
    GaussianTraining training;
    training.initialize(device, physical, queue, family, family);
    training.setDirectIntrinsicsProjection(direct);
    training.setFasterSortEnabled(true);
    training.setPixelTo2DGSMode(TrainingPixelTo2DGSMode::Direct);
    training.setStepObservationEnabled(true);
    TrainingScheduleConfig schedule;
    schedule.totalIterations = 1;
    training.setScheduleConfig(schedule);
    TrainingDensificationConfig density;
    density.enabled = false;
    training.setDensificationConfig(density);
    training.resize(uint32_t(params.size()), {37, 23});
    training.setSsimMode(TrainingSsimMode::BackwardSeparable);
    training.initializeTrainingRenderers(device, physical, queue, family, uint32_t(params.size()),
                                         {37, 23});
    training.buffers().uploadGaussianParams(params.data(), uint32_t(params.size()));
    training.buffers().uploadCamera(camera, intrinsics);
    std::vector<uint32_t> target(37 * 23, 0xff302010);
    training.buffers().uploadTargetColor(reinterpret_cast<const uint8_t*>(target.data()), 37, 23);
    training.trainStep();
    require(training.lastProjectionState().active == direct, "Paired direct/matrix path mismatch");
    require(!training.lastStepObservation().optimizerUpdated, "Paired snapshot changed parameters");
    OptimizerCaptureTransfer transfer(device, physical, queue, family);
    Snapshot out;
    out.ids = read<uint32_t>(transfer, training.buffers().tileItemsInfo());
    out.ids.resize(training.lastStepObservation().tileItems);
    out.ranges = read<glm::uvec4>(transfer, training.buffers().tileRangesInfo());
    out.colors = read<glm::vec4>(transfer, training.buffers().renderedColorInfo());
    out.gradients = read<ProjectedGaussianGrad>(transfer, training.buffers().projectedGradsInfo());
    require(training.validationStats().valid, "Paired direct snapshot validation failed");
    return out;
}
template <class T>
uint64_t differingFloatWords(const std::vector<T>& a, const std::vector<T>& b,
                             double& maxDifference) {
    require(a.size() == b.size(), "Paired float output size mismatch");
    auto x = std::as_bytes(std::span(a)), y = std::as_bytes(std::span(b));
    uint64_t count = 0;
    for (size_t i = 0; i < x.size(); i += 4) {
        float xv, yv;
        std::memcpy(&xv, x.data() + i, 4);
        std::memcpy(&yv, y.data() + i, 4);
        require(std::isfinite(xv) && std::isfinite(yv),
                "Paired direct output contains non-finite values");
        count += std::memcmp(x.data() + i, y.data() + i, 4) != 0;
        maxDifference = std::max(maxDifference, std::abs(double(xv) - yv));
    }
    return count;
}
inline std::string run(vk::Device device, vk::PhysicalDevice physical, vk::Queue queue,
                       uint32_t family) {
    OptimizerCaptureTransfer transfer(device, physical, queue, family);
    TrainingBuffers buffers;
    buffers.initialize(device, physical, queue, family, family);
    constexpr uint32_t count = 7;
    buffers.resize(count, {37, 23});
    const auto usage = vk::BufferUsageFlagBits::eStorageBuffer |
                       vk::BufferUsageFlagBits::eTransferSrc |
                       vk::BufferUsageFlagBits::eTransferDst;
    auto create = [&] {
        Buffer b;
        b.create(device, physical, queue, family, nullptr, count * 16, usage,
                 vk::MemoryPropertyFlagBits::eDeviceLocal, {family});
        return b;
    };
    auto gradients = create(), outputs = create(), coordinates = create();
    ComputePipeline probe, matrix;
    ComputePipelineConfig config;
    for (auto binding : {0u, 3u, 65u, 66u, 67u})
        config.descriptorBindings.push_back(backwardStorageBinding(binding));
    config.descriptorBindings.push_back(backwardUniformBinding(11));
    config.descriptorBindings.push_back(backwardUniformBinding(64));
    config.pushConstantSize = sizeof(TrainingPushConstants);
    probe.initialize(device, "shaders/train_direct_intrinsics_probe.comp.spv", config);
    matrix.initialize(device, "shaders/train_matrix_projection_probe.comp.spv", config);
    BackwardDescriptorPool descriptors;
    const auto sets = descriptors.create(
        device, 10, 4, {probe.getDescriptorSetLayout(), matrix.getDescriptorSetLayout()});
    auto bind = [&](vk::DescriptorSet set, const std::vector<uint32_t>& bindings,
                    const std::vector<vk::DescriptorBufferInfo>& infos) {
        std::vector<vk::WriteDescriptorSet> writes(bindings.size());
        for (size_t i = 0; i < writes.size(); ++i)
            writes[i]
                .setDstSet(set)
                .setDstBinding(bindings[i])
                .setDescriptorCount(1)
                .setDescriptorType(bindings[i] == 11 || bindings[i] == 64
                                       ? vk::DescriptorType::eUniformBuffer
                                       : vk::DescriptorType::eStorageBuffer)
                .setPBufferInfo(&infos[i]);
        device.updateDescriptorSets(writes, {});
    };
    bind(sets[0], {0, 3, 65, 66, 67, 11, 64},
         {buffers.gaussianParamsInfo(), buffers.projectedInfo(), gradients.getDescriptorInfo(),
          outputs.getDescriptorInfo(), coordinates.getDescriptorInfo(), buffers.cameraInfo(),
          buffers.cameraIntrinsicsInfo()});
    bind(sets[1], {0, 3, 65, 66, 67, 11, 64},
         {buffers.gaussianParamsInfo(), buffers.projectedInfo(), gradients.getDescriptorInfo(),
          outputs.getDescriptorInfo(), coordinates.getDescriptorInfo(), buffers.cameraInfo(),
          buffers.cameraIntrinsicsInfo()});
    CommandPool pool;
    pool.create(device, family, vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
    auto command = pool.allocateCommandBuffer();
    auto fence = device.createFenceUnique({});
    TrainingPushConstants push{};
    push.gaussianCount = count;
    push.width = 37;
    push.height = 23;
    push.pixelCount = 37 * 23;
    push.activeSHDegree = 3;
    auto dispatch = [&](ComputePipeline& pipeline, vk::DescriptorSet set) {
        command.reset();
        command.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        vk::MemoryBarrier barrier{
            vk::AccessFlagBits::eMemoryWrite | vk::AccessFlagBits::eMemoryRead,
            vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite};
        command.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                                vk::PipelineStageFlagBits::eAllCommands, {}, barrier, {}, {});
        command.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.getPipeline());
        command.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipeline.getPipelineLayout(), 0,
                                   set, {});
        command.pushConstants(pipeline.getPipelineLayout(), vk::ShaderStageFlagBits::eCompute, 0,
                              sizeof(push), &push);
        command.dispatch(1, 1, 1);
        command.end();
        device.resetFences(*fence);
        vk::SubmitInfo submit;
        submit.setCommandBuffers(command);
        queue.submit(submit, *fence);
        (void)device.waitForFences(*fence, true, UINT64_MAX);
    };
    uint64_t xyDifferences = 0;
    double maxXY = 0, maxGradient = 0, maxGradientVsMatrix = 0;
    TrainingForwardCamera pairCamera;
    TrainingCameraIntrinsics pairIntrinsics;
    std::vector<GaussianTrainParam> pairParams;
    for (uint32_t scene = 0; scene < 4; ++scene) {
        TrainingForwardCamera camera;
        camera.viewport = {0, 0, 37, 23};
        camera.focalTan = {25.25f, 19.75f, 37.0f / 50.5f, 23.0f / 39.5f};
        const glm::vec4 intrinsics{25.25f, 19.75f, scene ? 11.3f : 18.5f, scene ? 7.2f : 11.5f};
        camera.projection = glm::mat4(0);
        camera.projection[0][0] = 2 * intrinsics.x / 37;
        camera.projection[1][1] = -2 * intrinsics.y / 23;
        camera.projection[2][0] = 2 * intrinsics.z / 37 - 1;
        camera.projection[2][1] = 1 - 2 * intrinsics.w / 23;
        camera.projection[2][2] = 1.0f;
        camera.projection[2][3] = 1;
        if (scene == 2)
            camera.view = glm::translate(glm::rotate(glm::mat4(1), 0.23f, glm::vec3(0, 0, 1)),
                                         glm::vec3(0.125f, -0.25f, 0));
        if (scene == 3)
            camera.model = glm::scale(glm::rotate(glm::mat4(1), 0.17f, glm::vec3(0, 1, 0)),
                                      glm::vec3(0.7f, 1.2f, 0.9f));
        std::vector<GaussianTrainParam> params(count);
        for (uint32_t i = 0; i < count; ++i) {
            params[i].positionOpacity = {float(i) * 0.13f - 0.39f, float(i % 3) * 0.1f - 0.1f,
                                         2.0f + float(i) * 0.125f, 0};
            params[i].scale = {-3, -3, -3, 0};
            params[i].rotation = {0, 0, 0, 1};
        }
        if (scene == 0) {
            params[0].positionOpacity.z = 0.199999f;
            params[1].positionOpacity.z = 0.200001f;
            params[2].positionOpacity.x = -1.45f;
        }
        buffers.uploadGaussianParams(params.data(), count);
        pairCamera = camera;
        pairIntrinsics = {intrinsics};
        pairParams = params;
        buffers.uploadCamera(camera, TrainingCameraIntrinsics{intrinsics});
        require(buffers.hasExactIntrinsics(), "Exact direct intrinsics unavailable");
        std::vector<glm::vec4> centerGrad(count, glm::vec4(0.25f, -0.375f, 0, 0));
        upload(transfer, gradients.getDescriptorInfo(), centerGrad);
        dispatch(matrix, sets[1]);
        const auto baseline = read<ProjectedGaussian>(transfer, buffers.projectedInfo());
        const auto matrixGrad = read<glm::vec4>(transfer, outputs.getDescriptorInfo());
        const auto matrixCamera = read<glm::vec4>(transfer, coordinates.getDescriptorInfo());
        dispatch(probe, sets[0]);
        const auto direct = read<ProjectedGaussian>(transfer, buffers.projectedInfo());
        const auto cameraSpace = read<glm::vec4>(transfer, coordinates.getDescriptorInfo());
        const auto actualGrad = read<glm::vec4>(transfer, outputs.getDescriptorInfo());
        morton_checks::exact(cameraSpace, matrixCamera,
                             "Direct projection changed camera transform bits");
        for (uint32_t i = 0; i < count; ++i) {
            require(std::bit_cast<uint32_t>(baseline[i].centerRadius.z) ==
                        std::bit_cast<uint32_t>(direct[i].centerRadius.z),
                    "Direct projection changed depth bits");
            if (!direct[i].centerRadius.w) continue;
            const auto point = glm::dvec3(cameraSpace[i]);
            const glm::dvec2 expected{double(intrinsics.x) * point.x / point.z + intrinsics.z - 0.5,
                                      double(intrinsics.y) * point.y / point.z + intrinsics.w -
                                          0.5};
            for (int axis = 0; axis < 2; ++axis) {
                const double bound = 8 * double(std::numeric_limits<float>::epsilon()) *
                                     (std::abs(double(intrinsics[axis]) * point[axis] / point.z) +
                                      std::abs(double(intrinsics[axis + 2])) + 0.5);
                require(std::isfinite(direct[i].centerRadius[axis]) &&
                            std::abs(double(direct[i].centerRadius[axis]) - expected[axis]) <=
                                bound,
                        "Direct coordinate FP64 oracle bound failed");
                xyDifferences += std::bit_cast<uint32_t>(baseline[i].centerRadius[axis]) !=
                                 std::bit_cast<uint32_t>(direct[i].centerRadius[axis]);
                maxXY = std::max(maxXY, std::abs(double(direct[i].centerRadius[axis]) -
                                                 baseline[i].centerRadius[axis]));
            }
            const auto p = glm::dmat4(camera.view) * glm::dmat4(camera.model) *
                           glm::dvec4(glm::dvec3(params[i].positionOpacity), 1);
            const glm::dvec4 cameraGrad{
                0.25 * intrinsics.x / p.z, -0.375 * intrinsics.y / p.z,
                -(0.25 * intrinsics.x * p.x - 0.375 * intrinsics.y * p.y) / (p.z * p.z), 0};
            const auto expectedGradient = glm::transpose(glm::dmat4(camera.model)) *
                                          glm::transpose(glm::dmat4(camera.view)) * cameraGrad;
            for (int axis = 0; axis < 3; ++axis) {
                maxGradientVsMatrix =
                    std::max(maxGradientVsMatrix,
                             std::abs(double(actualGrad[i][axis]) - matrixGrad[i][axis]));
                const double tolerance = 1e-5 + 1e-3 * std::abs(expectedGradient[axis]);
                require(std::isfinite(actualGrad[i][axis]) &&
                            std::abs(double(actualGrad[i][axis]) - expectedGradient[axis]) <=
                                tolerance,
                        "Direct gradient analytic oracle failed");
                glm::dvec3 plus(params[i].positionOpacity), minus = plus;
                plus[axis] += 1e-5;
                minus[axis] -= 1e-5;
                const auto delta =
                    (oracle(plus, camera, intrinsics) - oracle(minus, camera, intrinsics)) / (2e-5);
                const double finiteDifference = 0.25 * delta.x - 0.375 * delta.y;
                require(std::abs(finiteDifference - expectedGradient[axis]) <= tolerance,
                        "Direct gradient finite difference failed");
                maxGradient = std::max(
                    maxGradient, std::abs(double(actualGrad[i][axis]) - expectedGradient[axis]));
            }
        }
        buffers.uploadCamera(camera);
        require(!buffers.hasExactIntrinsics(),
                "Camera replacement retained stale direct intrinsics");
    }
    const auto steps = morton_checks::lifecycle(device, physical, queue, family, true);
    const auto a =
        snapshot(device, physical, queue, family, false, pairCamera, pairIntrinsics, pairParams);
    const auto b =
        snapshot(device, physical, queue, family, true, pairCamera, pairIntrinsics, pairParams);
    uint64_t ids = 0, ranges = 0;
    for (size_t i = 0; i < std::min(a.ids.size(), b.ids.size()); ++i)
        ids += a.ids[i] != b.ids[i];
    require(a.ranges.size() == b.ranges.size(), "Paired tile range size mismatch");
    for (size_t i = 0; i < a.ranges.size(); ++i)
        ranges += std::memcmp(&a.ranges[i], &b.ranges[i], 16) != 0;
    double maxColorDifference = 0, maxProjectedGradientDifference = 0;
    const auto colorDifferences = differingFloatWords(a.colors, b.colors, maxColorDifference);
    const auto gradientDifferences =
        differingFloatWords(a.gradients, b.gradients, maxProjectedGradientDifference);
    std::ostringstream out;
    out << "{\"cases\":4,\"depth_bitwise\":true,\"fp64_oracle\":true,\"finite_difference\":true,"
           "\"xy_differing_components\":"
        << xyDifferences << ",\"max_xy_difference\":" << maxXY
        << ",\"max_gradient_error\":" << maxGradient
        << ",\"max_gradient_difference_vs_matrix\":" << maxGradientVsMatrix
        << ",\"coverage\":{\"matrix_items\":" << a.ids.size()
        << ",\"direct_items\":" << b.ids.size() << ",\"id_common_prefix_differences\":" << ids
        << ",\"range_differences\":" << ranges << "},\"color_differing_words\":" << colorDifferences
        << ",\"projected_gradient_differing_words\":" << gradientDifferences
        << ",\"max_color_difference\":" << maxColorDifference
        << ",\"max_projected_gradient_difference\":" << maxProjectedGradientDifference
        << ",\"expected_missing_intrinsics_fallback_checks\":1,\"positive_direct_owner_steps\":6,"
           "\"stopped_reconfiguration_checked\":true,"
           "\"intrinsics_fp32\":[25.25,19.75,11.3000002,7.19999981],\"direct_probe_shader_"
           "fnv1a64\":\""
        << optimizerShaderHash(
               FileUtils::resolvePath("shaders/train_direct_intrinsics_probe.comp.spv"))
        << "\",\"direct_forward_shader_fnv1a64\":\""
        << optimizerShaderHash(
               FileUtils::resolvePath("shaders/train_forward_project_direct_intrinsics.comp.spv"))
        << "\",\"direct_backward_shader_fnv1a64\":\""
        << optimizerShaderHash(FileUtils::resolvePath(
               "shaders/train_backward_2dgs_to_3dgs_direct_intrinsics.comp.spv"))
        << "\",\"direct_fused_shader_fnv1a64\":\""
        << optimizerShaderHash(FileUtils::resolvePath(
               "shaders/train_backward_2dgs_to_3dgs_optimizer_direct_intrinsics.comp.spv"))
        << "\",\"synthetic_owner_steps\":" << steps << ",\"timing_admission\":false}";
    return out.str();
}
} // namespace direct_intrinsics_checks
