#pragma once

#include "training_optimizer_reference.hpp"
#include "training_vksplat_checks.hpp"
#include <glm/gtc/quaternion.hpp>

namespace vksplat_backend_checks {
using namespace vulkan3DGS;
using morton_checks::require;
using vksplat_checks::close;

inline double projectionObjective(const GaussianTrainParam& g, const TrainingForwardCamera& camera,
                                  glm::vec4 intrinsics, const ProjectedGaussianGrad& upstream) {
    const glm::dvec3 pos = glm::dmat4(camera.view) * glm::dvec4(glm::dvec3(g.positionOpacity), 1);
    const auto q =
        glm::normalize(glm::dquat(g.rotation.w, g.rotation.x, g.rotation.y, g.rotation.z));
    const glm::dmat3 rotation = glm::mat3_cast(q);
    glm::dmat3 scale(0);
    for (int i = 0; i < 3; ++i)
        scale[i][i] = std::exp(2.0 * g.scale[i]);
    const glm::dmat3 view(camera.view);
    const auto covariance =
        view * rotation * scale * glm::transpose(rotation) * glm::transpose(view);
    glm::dmat3 j(0);
    j[0][0] = intrinsics.x / pos.z;
    j[1][1] = intrinsics.y / pos.z;
    j[2][0] = -intrinsics.x * pos.x / (pos.z * pos.z);
    j[2][1] = -intrinsics.y * pos.y / (pos.z * pos.z);
    auto c = j * covariance * glm::transpose(j);
    c[0][0] += 0.3;
    c[1][1] += 0.3;
    const double det = c[0][0] * c[1][1] - c[0][1] * c[1][0];
    return upstream.centerRadius.x * (intrinsics.x * pos.x / pos.z + intrinsics.z - 0.5) +
           upstream.centerRadius.y * (intrinsics.y * pos.y / pos.z + intrinsics.w - 0.5) +
           upstream.conicOpacity.x * c[1][1] / det - 2.0 * upstream.conicOpacity.y * c[0][1] / det +
           upstream.conicOpacity.z * c[0][0] / det +
           upstream.conicOpacity.w / (1.0 + std::exp(-double(g.positionOpacity.w)));
}

inline uint32_t states(vk::Device device, vk::PhysicalDevice physical, vk::Queue queue,
                       uint32_t family) {
    OptimizerCaptureTransfer transfer(device, physical, queue, family);
    CommandPool pool;
    pool.create(device, family, vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
    auto command = pool.allocateCommandBuffer();
    auto fence = device.createFenceUnique({});
    auto submit = [&] {
        command.end();
        queue.submit(vk::SubmitInfo{}.setCommandBuffers(command), *fence);
        (void)device.waitForFences(*fence, true, UINT64_MAX);
        device.resetFences(*fence);
    };
    uint32_t cases = 0;
    for (uint32_t count : {0u, 1u, 31u, 32u, 33u, 63u, 64u, 65u}) {
        TrainingBuffers buffers;
        buffers.initialize(device, physical, queue, family, family);
        buffers.resize(std::max(count, 1u), {33, 31});
        std::vector<GaussianTrainParam> parameters(std::max(count, 1u));
        std::vector<AdamState> moments(parameters.size());
        for (size_t i = 0; i < parameters.size(); ++i) {
            auto& g = parameters[i];
            g.positionOpacity = {0.21f + float(i) * 0.0003f, -0.12f, 2.3f, -0.4f};
            if (count > 1 && i == count - 1) g.positionOpacity.z = -0.05f;
            g.scale = {-1.4f, -1.9f, -2.1f, 0.75f};
            g.rotation = glm::normalize(glm::vec4(0.13f, -0.21f, 0.07f, 1.0f));
            for (uint32_t b = 0; b < 12; ++b)
                g.sh[b] = glm::vec4(float(b + 1) * 0.001f + float(i) * 0.0001f);
            auto m = std::bit_cast<std::array<float, 120>>(moments[i]);
            for (size_t k = 0; k < m.size(); ++k)
                m[k] = float(k + 1) * 0.00001f + float(i) * 0.000001f;
            moments[i] = std::bit_cast<AdamState>(m);
        }
        morton_checks::upload(transfer, buffers.gaussianParamsInfo(), parameters);
        morton_checks::upload(transfer, buffers.adamStatesInfo(), moments);
        TrainingVkSplatBackend core(device, physical, queue, family);
        core.importCanonical(buffers, count);
        require(!buffers.gaussianParamsInfo().buffer && !buffers.adamStatesInfo().buffer,
                "Native model retained a complete canonical mirror");
        core.exportCanonical(buffers, count);
        const auto actual =
            morton_checks::read<GaussianTrainParam>(transfer, buffers.gaussianParamsInfo());
        const auto actualMoments =
            morton_checks::read<AdamState>(transfer, buffers.adamStatesInfo());
        for (uint32_t i = 0; i < count; ++i) {
            require(std::memcmp(&actualMoments[i], &moments[i], sizeof(AdamState)) == 0,
                    "Native moment packing was not bitwise reversible");
            auto a = std::bit_cast<std::array<float, 60>>(actual[i]);
            auto b = std::bit_cast<std::array<float, 60>>(parameters[i]);
            for (size_t k = 0; k < a.size(); ++k)
                close(a[k], b[k], "Native parameter conversion mismatch");
        }
        core.importCanonical(buffers, count);
        TrainingForwardCamera camera{};
        camera.view = glm::mat4(1);
        camera.model = glm::mat4(1);
        camera.view[3] = {-0.03f, 0.02f, 0.1f, 1.0f};
        camera.viewport = {0, 0, 33, 31};
        const glm::vec4 intrinsics{17, 19, 15.7f, 14.8f};
        camera.focalTan = {17, 19, 33.0f / 34, 31.0f / 38};
        buffers.uploadCameraWithIntrinsics(camera, {intrinsics});
        TrainingPushConstants push{};
        push.gaussianCount = count;
        push.width = 33;
        push.height = 31;
        push.pixelCount = 33 * 31;
        push.activeSHDegree = 3;
        push.optimizerStep = 17;
        push.optimizerOneMinusBeta1 = 1.0f - push.optimizerBeta1;
        push.optimizerOneMinusBeta2 = 1.0f - push.optimizerBeta2;
        push.optimizerInvFirstMomentCorrection =
            1.0f / (1.0f - std::pow(push.optimizerBeta1, 17.0f));
        push.optimizerInvSecondMomentCorrection =
            1.0f / (1.0f - std::pow(push.optimizerBeta2, 17.0f));
        command.reset();
        command.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        core.recordProject({buffers, command, push});
        submit();
        auto projected = morton_checks::read<ProjectedGaussian>(transfer, buffers.projectedInfo());
        for (uint32_t i = 0; i < count; ++i) {
            const auto pos = camera.view * glm::vec4(glm::vec3(parameters[i].positionOpacity), 1);
            close(projected[i].centerRadius.x, intrinsics.x * pos.x / pos.z + intrinsics.z - 0.5,
                  "Official pinhole X differs from independent formula");
            close(projected[i].centerRadius.y, intrinsics.y * pos.y / pos.z + intrinsics.w - 0.5,
                  "Official pinhole Y differs from independent formula");
        }
        std::vector<ProjectedGaussianGrad> gradient(parameters.size());
        for (auto& g : gradient) {
            g.centerRadius = {0.012f, -0.017f, 0, 0};
            g.conicOpacity = {0.031f, 0.027f, -0.019f, 0.013f};
        }
        morton_checks::upload(transfer, buffers.projectedGradsInfo(), gradient);
        command.reset();
        command.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        core.recordOptimizer({buffers, command, push});
        submit();
        core.exportCanonical(buffers, count);
        const auto updated = morton_checks::read<AdamState>(transfer, buffers.adamStatesInfo());
        if (count) {
            auto raw = std::bit_cast<std::array<float, 60>>(parameters[0]);
            const auto oldMoment = std::bit_cast<std::array<float, 60>>(moments[0].firstMoment);
            const auto newMoment = std::bit_cast<std::array<float, 60>>(updated[0].firstMoment);
            for (size_t k = 0; k < raw.size(); ++k) {
                if (k == 7) continue;
                double derivative = 0;
                if (k < 12) {
                    const float before = raw[k];
                    raw[k] = before + 0.0002f;
                    const float high = raw[k];
                    const double plus = projectionObjective(std::bit_cast<GaussianTrainParam>(raw),
                                                            camera, intrinsics, gradient[0]);
                    raw[k] = before - 0.0002f;
                    const double minus = projectionObjective(std::bit_cast<GaussianTrainParam>(raw),
                                                             camera, intrinsics, gradient[0]);
                    derivative = (plus - minus) / (double(high) - raw[k]);
                    raw[k] = before;
                }
                const double recovered =
                    (newMoment[k] - double(push.optimizerBeta1) * oldMoment[k]) /
                    push.optimizerOneMinusBeta1;
                close(
                    recovered, derivative,
                    "Official projection/activation/quaternion gradient failed finite differences");
            }
        }
        if (count == 1) {
            auto invalid = updated;
            invalid[0].firstMoment.sh[11].w = std::numeric_limits<float>::quiet_NaN();
            morton_checks::upload(transfer, buffers.adamStatesInfo(), invalid);
            core.importCanonical(buffers, count);
            command.reset();
            command.begin(
                vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
            core.recordValidation({buffers, command, push});
            submit();
            const auto partial = morton_checks::read<TrainingGaussianValidationPartial>(
                transfer, buffers.gaussianValidationPartialsInfo());
            require(partial[0].nonFiniteGaussianCount == 1 && partial[0].nonFiniteSHCount == 0,
                    "Native validation did not identify non-finite Adam state separately from "
                    "parameter fields");
        }
        ++cases;
    }
    return cases;
}

inline uint32_t composite(vk::Device device, vk::PhysicalDevice physical, vk::Queue queue,
                          uint32_t family) {
    OptimizerCaptureTransfer transfer(device, physical, queue, family);
    CommandPool pool;
    pool.create(device, family, vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
    const auto command = pool.allocateCommandBuffer();
    auto fence = device.createFenceUnique({});
    auto submit = [&] {
        command.end();
        queue.submit(vk::SubmitInfo{}.setCommandBuffers(command), *fence);
        (void)device.waitForFences(*fence, true, UINT64_MAX);
        device.resetFences(*fence);
    };
    uint32_t cases = 0;
    for (auto extent : {TrainingExtent{1, 1}, TrainingExtent{17, 19}, TrainingExtent{33, 31}}) {
        TrainingBuffers buffers;
        buffers.initialize(device, physical, queue, family, family);
        buffers.resize(3, extent);
        std::vector<GaussianTrainParam> seed(3);
        for (uint32_t i = 0; i < 3; ++i) {
            seed[i].positionOpacity = {0.3f - float(i) * 0.2f, -0.12f, 2.0f + float(i) * 0.1f,
                                       -0.3f};
            seed[i].scale = {-0.8f, -1.2f, -1.5f, 0};
            seed[i].rotation = {0, 0, 0, 1};
            seed[i].sh[0] = {0.1f, -0.1f, 0.2f, 0};
        }
        buffers.uploadGaussianParams(seed.data(), 3);
        TrainingVkSplatBackend core(device, physical, queue, family);
        core.importCanonical(buffers, 3);
        TrainingForwardCamera camera{};
        camera.view = camera.model = glm::mat4(1);
        camera.viewport = {0, 0, float(extent.width), float(extent.height)};
        const float fx = float(extent.width) * 0.7f, fy = float(extent.height) * 0.7f;
        camera.focalTan = {fx, fy, 0.7f, 0.7f};
        buffers.uploadCameraWithIntrinsics(
            camera, {{fx, fy, float(extent.width) * 0.5f, float(extent.height) * 0.5f}});
        GaussianForwardRenderer forward;
        forward.initialize(device, physical, queue, family, 3, extent);
        forward.setFasterSortEnabled(true);
        TrainingPushConstants push{};
        push.gaussianCount = 3;
        push.width = extent.width;
        push.height = extent.height;
        push.pixelCount = extent.width * extent.height;
        std::vector<glm::vec4> reference;
        std::vector<PixelBlendState> referenceBlends;
        for (bool mask : {false, true, false}) {
            forward.setVkSplatBackend(&core, mask);
            command.reset();
            command.begin(
                vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
            forward.setTrainingBuffers(buffers, command, push);
            forward.prepareTileItems();
            buffers.recordTileItemCountReadback(command);
            submit();
            push.tileItemCount = buffers.requiredTileItemCount();
            require(push.tileItemCount <= buffers.tileItemCapacity(),
                    "Native fixture exceeded item capacity");
            command.reset();
            command.begin(
                vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
            forward.setTrainingBuffers(buffers, command, push);
            forward.renderPreparedTiles(push.tileItemCount);
            submit();
            const auto colors =
                morton_checks::read<glm::vec4>(transfer, buffers.renderedColorInfo());
            const auto blends =
                morton_checks::read<PixelBlendState>(transfer, buffers.pixelBlendStatesInfo());
            if (reference.empty()) {
                reference = colors;
                referenceBlends = blends;
            } else {
                for (size_t i = 0; i < colors.size(); ++i)
                    for (int c = 0; c < 4; ++c)
                        close(colors[i][c], reference[i][c],
                              "Subtile changed output against same-state unmasked reference");
                morton_checks::exact(blends, referenceBlends,
                                     "Subtile changed forward/backward pixel state");
            }
            const auto projected =
                morton_checks::read<ProjectedGaussian>(transfer, buffers.projectedInfo());
            const auto ids = morton_checks::read<uint32_t>(transfer, buffers.tileItemsInfo());
            const auto ranges = morton_checks::read<glm::uvec4>(transfer, buffers.tileRangesInfo());
            for (uint32_t y = 0; y < extent.height; ++y)
                for (uint32_t x = 0; x < extent.width; ++x) {
                    const auto range = ranges[(y / 16) * ((extent.width + 15) / 16) + x / 16];
                    glm::dvec3 color(0);
                    double t = 1;
                    float previous = -1;
                    for (uint32_t j = 0; j < range.y; ++j) {
                        const auto& g = projected[ids[range.x + j]];
                        require(g.centerRadius.z >= previous,
                                "Native emit/sort did not preserve exact depth order");
                        previous = g.centerRadius.z;
                        const double dx = double(x) - g.centerRadius.x,
                                     dy = double(y) - g.centerRadius.y;
                        const double power =
                            -0.5 * (g.conicOpacity.x * dx * dx + 2 * g.conicOpacity.y * dx * dy +
                                    g.conicOpacity.z * dy * dy);
                        const double a =
                            std::min(0.999, double(g.conicOpacity.w) * std::exp(power));
                        if (a < 1.0 / 255) continue;
                        const double next = t * (1 - a);
                        if (next < 1e-4) break;
                        color += glm::dvec3(g.color) * a * t;
                        t = next;
                    }
                    const size_t i = size_t(y) * extent.width + x;
                    for (int c = 0; c < 3; ++c)
                        close(colors[i][c], color[c],
                              "Native composite differs from scalar FP64 oracle");
                    close(colors[i].w, 1 - t, "Native composite opacity mismatch");
                }
            ++cases;
        }
        forward.cleanup();
    }
    return cases;
}

inline uint32_t stateEvents(vk::Device device, vk::PhysicalDevice physical, vk::Queue queue,
                            uint32_t family) {
    OptimizerCaptureTransfer transfer(device, physical, queue, family);
    TrainingBuffers buffers;
    buffers.initialize(device, physical, queue, family, family);
    buffers.resize(4, {17, 19});
    std::vector<GaussianTrainParam> seed(4);
    std::vector<GaussianDensificationState> statistics(4);
    for (uint32_t i = 0; i < 4; ++i) {
        seed[i].positionOpacity = {float(i) * 0.1f, 0, 2, i == 2 ? -30.0f : 0.0f};
        seed[i].scale = glm::vec4(i == 1 ? -2.8f : -8.0f);
        seed[i].rotation = {0, 0, 0, 1};
        statistics[i].screenGradSum = 0.01f;
        statistics[i].screenGradCount = 1;
    }
    buffers.uploadGaussianParams(seed.data(), 4);
    std::vector<AdamState> initialMoments(4);
    for (uint32_t i = 0; i < 4; ++i) {
        std::array<float, 120> values;
        for (size_t j = 0; j < values.size(); ++j)
            values[j] = float(i + 1) * 0.001f + float(j) * 0.000001f;
        initialMoments[i] = std::bit_cast<AdamState>(values);
    }
    morton_checks::upload(transfer, buffers.adamStatesInfo(), initialMoments);
    morton_checks::upload(transfer, buffers.densificationStatesInfo(), statistics);
    TrainingVkSplatBackend core(device, physical, queue, family);
    core.importCanonical(buffers, 4);
    core.exportCanonical(buffers, 4);
    buffers.ensureDensificationCapacity(32);
    GaussianDensificationRenderer adc;
    adc.initialize(device, physical, queue, family);
    CommandPool pool;
    pool.create(device, family);
    auto command = pool.allocateCommandBuffer();
    auto fence = device.createFenceUnique({});
    command.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    TrainingDensificationPushConstants push{};
    push.gaussianCount = 4;
    push.maxGaussianCount = 32;
    push.resetOpacity = 1;
    push.randomSeed = 17;
    adc.setTrainingBuffers(buffers, command, push);
    adc.densifyAndPrune();
    buffers.recordDensificationStatsReadback(command);
    command.end();
    queue.submit(vk::SubmitInfo{}.setCommandBuffers(command), *fence);
    (void)device.waitForFences(*fence, true, UINT64_MAX);
    const auto result = buffers.densificationStats();
    require(result.outputCount > 0 && result.cloneSources > 0 && result.splitSources > 0 &&
                result.prunedSources > 0,
            "Native ADC boundary did not cover clone/split/prune");
    adc.cleanup();
    buffers.adoptDensifiedGaussians(result.outputCount);
    core.importCanonical(buffers, result.outputCount);
    core.exportCanonical(buffers, result.outputCount);
    const auto before =
        morton_checks::read<GaussianTrainParam>(transfer, buffers.gaussianParamsInfo());
    const auto moments = morton_checks::read<AdamState>(transfer, buffers.adamStatesInfo());
    glm::vec3 minimum(std::numeric_limits<float>::infinity()),
        maximum(-std::numeric_limits<float>::infinity());
    for (uint32_t i = 0; i < result.outputCount; ++i) {
        minimum = glm::min(minimum, glm::vec3(before[i].positionOpacity));
        maximum = glm::max(maximum, glm::vec3(before[i].positionOpacity));
        for (float value : std::bit_cast<std::array<float, 120>>(moments[i]))
            require(std::isfinite(value), "Native ADC moments are non-finite");
    }
    const auto delta = maximum - minimum;
    const float side = std::max({delta.x, delta.y, delta.z});
    std::vector<uint32_t> order(result.outputCount);
    std::iota(order.begin(), order.end(), 0u);
    std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) {
        return morton_checks::encode(glm::vec3(before[a].positionOpacity), minimum, side) <
               morton_checks::encode(glm::vec3(before[b].positionOpacity), minimum, side);
    });
    TrainingMortonReorder reorder(device, physical, queue, family);
    (void)reorder.execute(buffers, result.outputCount);
    core.importCanonical(buffers, result.outputCount);
    core.exportCanonical(buffers, result.outputCount);
    const auto after = morton_checks::read<AdamState>(transfer, buffers.adamStatesInfo());
    for (uint32_t i = 0; i < result.outputCount; ++i)
        require(std::memcmp(&after[i], &moments[order[i]], sizeof(AdamState)) == 0,
                "Native Morton boundary changed moment values/permutation");
    return 1;
}

inline uint32_t lifecycle(vk::Device device, vk::PhysicalDevice physical, vk::Queue queue,
                          uint32_t family) {
    GaussianTraining training;
    training.initialize(device, physical, queue, family, family);
    training.setVkSplatCompute(true);
    training.setVkSplatLoss(true);
    training.setFasterSortEnabled(true);
    training.setMortonOrderingEnabled(true);
    training.setPixelTo2DGSMode(TrainingPixelTo2DGSMode::VkSplatTensor);
    training.setStepObservationEnabled(true);
    training.resize(4, {17, 19});
    std::array<GaussianTrainParam, 4> seed{};
    for (uint32_t i = 0; i < 4; ++i) {
        seed[i].positionOpacity = {0.3f - float(i) * 0.2f, 0, 2, i == 3 ? -30.0f : 0.0f};
        seed[i].scale = glm::vec4(i == 0 ? -8.0f : -4.0f);
        seed[i].rotation = {0, 0, 0, 1};
    }
    training.buffers().uploadGaussianParams(seed.data(), 4);
    training.initializeTrainingRenderers(device, physical, queue, family, 4, {17, 19});
    TrainingForwardCamera camera{};
    camera.view = camera.model = glm::mat4(1);
    camera.viewport = {0, 0, 17, 19};
    camera.focalTan = {8, 9, 17.0f / 16, 19.0f / 18};
    training.buffers().uploadCameraWithIntrinsics(camera, {{8, 9, 8.3f, 9.1f}});
    std::vector<uint32_t> target(17 * 19, 0xff201040u);
    training.buffers().uploadTargetColor(reinterpret_cast<const uint8_t*>(target.data()), 17, 19);
    TrainingOptimizerConfig optimizer;
    optimizer.shDegreeInterval = 1;
    training.setOptimizerConfig(optimizer);
    TrainingDensificationConfig adc;
    adc.densifyFromIteration = 0;
    adc.densifyUntilIteration = 7;
    adc.densificationInterval = 2;
    adc.opacityResetInterval = 4;
    adc.maxGaussianCount = 128;
    adc.densifyGradThreshold = 0;
    training.setDensificationConfig(adc);
    training.setValidationInterval(1);
    uint32_t adoptions = 0, resets = 0;
    for (uint32_t i = 1; i <= 8; ++i) {
        if (i == 3) training.setVkSplatSubtile(true);
        if (i == 4)
            training.initializeTrainingRenderers(device, physical, queue, family,
                                                 training.gaussianCount(), {17, 19});
        if (i == 5) training.setVkSplatSubtile(false);
        training.trainStep();
        const auto& observation = training.lastStepObservation();
        require(observation.fasterSortActive, "Native renderer used non-exact sorting fallback");
        require(training.lastVkSplatSubtile() == (i == 3 || i == 4),
                "Native subtile actual state mismatch");
        require(training.validationStats().invalidLossCount == 0 &&
                    training.validationStats().nonFiniteGaussianCount == 0,
                "Native lifecycle produced non-finite loss/model");
        adoptions += observation.densification;
        resets += observation.opacityReset;
        require(!training.buffers().gaussianParamsInfo().buffer,
                "Native lifecycle retained canonical model");
    }
    require(adoptions == 3 && resets == 1 && training.mortonStats().reorderCount == 1,
            "Native ADC/reset/Morton lifecycle coverage incomplete");
    require(training.inspectTrainingStateFinite().passed(),
            "Native final finite inspection failed");
    return 8;
}
} // namespace vksplat_backend_checks
