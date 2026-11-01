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

inline glm::dvec3 officialNormal(uint32_t seed, uint32_t index) {
    auto hash = [](uint32_t x) {
        x ^= x >> 16;
        x *= 0x7feb352d;
        x ^= x >> 15;
        x *= 0x846ca68b;
        x ^= x >> 16;
        return x;
    };
    const double r0 = double(hash(seed ^ ((index + 1u) * 0x9e3779b9u))) / 4294967296.0;
    const double r1 = double(hash(index ^ ((seed + 1u) * 0x9e3779b9u))) / 4294967296.0;
    const double r2 = double(hash(seed ^ ((index + 2u) * 0x9e3779b9u))) / 4294967296.0;
    const double r3 = double(hash(index ^ ((seed + 2u) * 0x9e3779b9u))) / 4294967296.0;
    constexpr double pi = 3.14159265358979323846;
    return {std::sqrt(-2 * std::log(1 - r0)) * std::cos(2 * pi * r1),
            std::sqrt(-2 * std::log(1 - r0)) * std::sin(2 * pi * r1),
            std::sqrt(-2 * std::log(1 - r2)) * std::cos(2 * pi * r3)};
}

inline uint32_t adcBoundaries(vk::Device device, vk::PhysicalDevice physical, vk::Queue queue,
                              uint32_t family, bool reuse = false) {
    OptimizerCaptureTransfer transfer(device, physical, queue, family);
    uint32_t cases = 0;
    for (uint32_t n : {0u, 1u, 31u, 32u, 33u, 255u, 256u, 257u, 1023u, 1024u, 1025u, 4097u}) {
        TrainingBuffers b;
        b.initialize(device, physical, queue, family, family);
        b.resize(n, {17, 19});
        std::vector<GaussianTrainParam> model(std::max(n, 1u));
        std::vector<AdamState> moments(std::max(n, 1u));
        std::vector<GaussianDensificationState> stats(std::max(n, 1u));
        for (uint32_t i = 0; i < n; ++i) {
            model[i].positionOpacity = {float(i), 0, 2, 0};
            model[i].scale = glm::vec4(-8);
            model[i].rotation = {0, 0, 0, 1};
            for (uint32_t j = 0; j < 12; ++j)
                model[i].sh[j] = glm::vec4(float(i * 12 + j) * 0.00001f);
            moments[i].firstMoment.sh[11] = glm::vec4(float(i + 1));
            moments[i].secondMoment.sh[11] = glm::vec4(float(i + 2));
            stats[i].screenGradCount = 1;
            stats[i].screenGradSum =
                (i % 3 == 0) ? 0.0004f : 0.0002f; // strict >, including threshold ties
        }
        if (n) b.uploadGaussianParams(model.data(), n);
        morton_checks::upload(transfer, b.adamStatesInfo(), moments);
        morton_checks::upload(transfer, b.densificationStatesInfo(), stats);
        TrainingVkSplatBackend core(device, physical, queue, family, reuse);
        core.importCanonical(b, n);
        if (reuse) b.enableNativeWorkspaceReuse();
        TrainingDensificationConfig c;
        c.maxGaussianCount = 10000;
        core.adc().prepare(c, 1, 100, false);
        const auto event = core.adc().execute(n, true, false);
        uint32_t expected = (n + 2) / 3;
        require(event.stats.outputCount == n + expected && event.stats.cloneSources == expected &&
                    event.stats.splitSources == 0,
                "Native ADC stable prefix/boundary counts failed");
        std::default_random_engine referenceRng(42);
        const auto firstSeed = std::uniform_int_distribution<uint32_t>()(referenceRng);
        const auto secondSeed = std::uniform_int_distribution<uint32_t>()(referenceRng);
        require(event.seed == firstSeed, "ADC RNG must start at the official seed42 sequence");
        if (n) {
            b.adoptNativeGaussianCount(event.stats.outputCount);
            core.exportCanonical(b, event.stats.outputCount);
            auto actual = morton_checks::read<GaussianTrainParam>(transfer, b.gaussianParamsInfo());
            auto m = morton_checks::read<AdamState>(transfer, b.adamStatesInfo());
            for (uint32_t i = 0; i < n + expected; ++i) {
                uint32_t source = i < n ? i : (i - n) * 3;
                require(std::memcmp(actual[i].sh, model[source].sh, sizeof(model[source].sh)) == 0,
                        "Stable clone SH indexing/physical column compaction failed");
                if (i < n)
                    require(std::memcmp(&m[i], &moments[i], sizeof(AdamState)) == 0,
                            "Parent moments changed");
                else
                    for (float value : std::bit_cast<std::array<float, 120>>(m[i]))
                        require(value == 0, "Child moments not zero");
            }
            core.importCanonical(b, event.stats.outputCount);
            core.adc().prepare(c, 1, 101, true); // benchmarks must not advance the independent RNG
            core.adc().prepare(c, 1, 101, false);
            const auto second = core.adc().execute(event.stats.outputCount, true, false);
            require(second.seed == secondSeed &&
                        second.stats.outputCount == event.stats.outputCount,
                    "Statistics clear/benchmark RNG/second ADC count failed");
        }
        ++cases;
    }
    for (bool empty : {false, true}) {
        TrainingBuffers b;
        b.initialize(device, physical, queue, family, family);
        b.resize(1, {17, 19});
        GaussianTrainParam g{};
        g.positionOpacity = {0, 0, 2, empty ? -30.0f : 0.0f};
        g.scale = glm::vec4(-8);
        g.rotation = {0, 0, 0, 1};
        b.uploadGaussianParams(&g, 1);
        std::vector<GaussianDensificationState> stats(1);
        stats[0].screenGradCount = 1;
        stats[0].screenGradSum = 1;
        morton_checks::upload(transfer, b.densificationStatesInfo(), stats);
        TrainingVkSplatBackend core(device, physical, queue, family, reuse);
        core.importCanonical(b, 1);
        TrainingDensificationConfig c;
        c.maxGaussianCount = empty ? 10 : 1;
        core.adc().prepare(c, 1, 100, false);
        std::cerr << "VkSplat ADC expected rejection: "
                  << (empty ? "empty keep mask" : "growth budget overflow") << "\n";
        bool rejected = false;
        try {
            (void)core.adc().execute(1, true, false);
        } catch (const std::runtime_error& e) {
            rejected =
                std::string(e.what()).find(empty ? "keep mask is empty"
                                                 : "exceeds Max Gaussians") != std::string::npos;
        }
        require(rejected, "Native ADC must explicitly reject cap overflow/empty keep mask");
        if (!empty) {
            core.exportCanonical(b, 1);
            auto actual = morton_checks::read<GaussianTrainParam>(transfer, b.gaussianParamsInfo());
            require(actual[0].positionOpacity == g.positionOpacity, "Cap rejection mutated model");
        } else
            require(core.adc().failed(), "Empty native model must block resume");
        ++cases;
    }
    {
        TrainingBuffers b;
        b.initialize(device, physical, queue, family, family);
        b.resize(4, {17, 19});
        std::vector<GaussianTrainParam> model(4);
        for (auto& g : model) {
            g.positionOpacity = {0, 0, 2, 0};
            g.scale = glm::vec4(-8);
            g.rotation = {0, 0, 0, 1};
        }
        model[1].positionOpacity.w = -30;
        model[2].scale = glm::vec4(std::log(0.2f));
        model[3].scale.x = std::numeric_limits<float>::quiet_NaN();
        b.uploadGaussianParams(model.data(), 4);
        TrainingVkSplatBackend core(device, physical, queue, family, reuse);
        core.importCanonical(b, 4);
        TrainingDensificationConfig c;
        c.maxGaussianCount = 8;
        core.adc().prepare(c, 1, 4000, false);
        auto event = core.adc().execute(4, true, false);
        require(event.stats.outputCount == 1 && event.stats.prunedSources == 3 &&
                    event.stats.pruneOpacityHits == 1 && event.stats.pruneWorldHits == 1,
                "Official opacity/world/nonfinite keep-mask predicates differ");
        ++cases;
    }
    {
        TrainingBuffers b;
        b.initialize(device, physical, queue, family, family);
        b.resize(2, {17, 19});
        std::array<GaussianTrainParam, 2> model{};
        for (auto& g : model) {
            g.positionOpacity = {0, 0, 2, 0};
            g.scale = glm::vec4(-4);
            g.rotation = {0, 0, 0, 1};
        }
        model[1].positionOpacity.x = 10000;
        b.uploadGaussianParams(model.data(), 2);
        TrainingForwardCamera camera{};
        camera.view = camera.model = glm::mat4(1);
        camera.viewport = {0, 0, 17, 19};
        camera.focalTan = {8, 9, 17.0f / 16, 19.0f / 18};
        b.uploadCameraWithIntrinsics(camera, {{8, 9, 8.3f, 9.1f}});
        std::vector<ProjectedGaussianGrad> g(2);
        for (auto& value : g)
            value.centerRadius = {0.02f, -0.01f, 0, 0};
        morton_checks::upload(transfer, b.projectedGradsInfo(), g);
        TrainingVkSplatBackend core(device, physical, queue, family, reuse);
        core.importCanonical(b, 2);
        TrainingDensificationConfig c;
        core.adc().prepare(c, 1, 1, false);
        CommandPool pool;
        pool.create(device, family, vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
        auto command = pool.allocateCommandBuffer();
        auto fence = device.createFenceUnique({});
        auto run = [&](bool benchmark) {
            core.adc().prepare(c, 1, 2, benchmark);
            command.reset();
            command.begin(
                vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
            TrainingPushConstants push{};
            push.gaussianCount = 2;
            push.width = 17;
            push.height = 19;
            core.recordProject({b, command, push});
            core.adc().recordStatistics({b, command, push});
            command.end();
            queue.submit(vk::SubmitInfo{}.setCommandBuffers(command), *fence);
            (void)device.waitForFences(*fence, true, UINT64_MAX);
            device.resetFences(*fence);
        };
        run(false);
        auto stats = morton_checks::read<glm::vec2>(transfer, core.adc().gradients());
        close(stats[0].x, 0.5 * std::hypot(0.02 * 17, 0.01 * 19),
              "Official normalized positive-radius gradient mismatch");
        require(stats[0].y == 1 && stats[1] == glm::vec2(0),
                "Official radius visibility/count predicate mismatch");
        run(true);
        auto benchmark = morton_checks::read<glm::vec2>(transfer, core.adc().gradients());
        require(benchmark == stats, "Benchmark must not accumulate native ADC statistics");
        ++cases;
    }
    return cases;
}

inline uint32_t stateEvents(vk::Device device, vk::PhysicalDevice physical, vk::Queue queue,
                            uint32_t family, bool reuse = false) {
    OptimizerCaptureTransfer transfer(device, physical, queue, family);
    TrainingBuffers buffers;
    buffers.initialize(device, physical, queue, family, family);
    buffers.resize(4, {17, 19});
    std::vector<GaussianTrainParam> seed(4);
    std::vector<GaussianDensificationState> statistics(4);
    for (uint32_t i = 0; i < 4; ++i) {
        seed[i].positionOpacity = {float(i) * 0.1f, 0, 2, i == 2 ? -30.0f : 0.0f};
        seed[i].scale = glm::vec4(i == 1 ? -2.8f : -8.0f);
        seed[i].rotation = i == 1 ? glm::vec4(0.15f, -0.3f, 0.1f, 2.0f) : glm::vec4(0, 0, 0, 1);
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
    TrainingVkSplatBackend core(device, physical, queue, family, reuse);
    core.importCanonical(buffers, 4);
    if (reuse) buffers.enableNativeWorkspaceReuse();
    TrainingDensificationConfig config;
    config.densifyGradThreshold = 0.0002f;
    config.maxGaussianCount = 32;
    core.adc().prepare(config, 1, 100, false);
    const auto event = core.adc().execute(4, true, true);
    const auto result = event.stats;
    require(result.cloneSources == 3 && result.splitSources == 1 && result.outputCount == 6 &&
                result.prunedSources == 2 && event.reset,
            "Native official ADC counts mismatch");
    require(!buffers.gaussianParamsInfo().buffer && buffers.densificationCapacity() == 0,
            "Native ADC must not materialize canonical or legacy densification storage");
    buffers.adoptNativeGaussianCount(result.outputCount);
    core.exportCanonical(buffers, result.outputCount);
    const auto before =
        morton_checks::read<GaussianTrainParam>(transfer, buffers.gaussianParamsInfo());
    const auto moments = morton_checks::read<AdamState>(transfer, buffers.adamStatesInfo());
    for (auto [dst, randomId] : {std::pair{1u, 1u}, std::pair{5u, 7u}}) {
        auto q = glm::normalize(glm::dquat(seed[1].rotation.w, seed[1].rotation.x,
                                           seed[1].rotation.y, seed[1].rotation.z));
        auto displacement =
            glm::mat3_cast(q) * (officialNormal(event.seed, randomId) * std::exp(-2.8));
        for (int axis = 0; axis < 4; ++axis)
            close(before[dst].rotation[axis], glm::dvec4(q.x, q.y, q.z, q.w)[axis],
                  "Official split quaternion normalization mismatch");
        for (int axis = 0; axis < 3; ++axis)
            close(before[dst].positionOpacity[axis],
                  double(seed[1].positionOpacity[axis]) + displacement[axis],
                  "Official randn3 split displacement mismatch");
    }
    const std::array<uint32_t, 6> source{0, 1, 3, 0, 3, 1};
    for (uint32_t i = 0; i < 6; ++i) {
        const float opacity = 1.0f / (1.0f + std::exp(-before[i].positionOpacity.w));
        close(opacity, 0.01, "Native reset opacity mismatch");
        require(moments[i].firstMoment.positionOpacity.w == 0 &&
                    moments[i].secondMoment.positionOpacity.w == 0,
                "Native reset did not clear opacity moments");
        if (i == 0 || i == 2) {
            auto expected = initialMoments[source[i]];
            expected.firstMoment.positionOpacity.w = expected.secondMoment.positionOpacity.w = 0;
            require(std::memcmp(&expected, &moments[i], sizeof(AdamState)) == 0,
                    "Duplicate parent moments must survive reset except opacity");
        } else
            for (float value : std::bit_cast<std::array<float, 120>>(moments[i]))
                require(value == 0, "Split/child moments must all be zero");
        if (i == 1 || i == 5)
            close(std::exp(before[i].scale.x), std::exp(-2.8f) / 1.6,
                  "Official split scale mismatch");
    }

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
    std::vector<glm::vec2> running(event.capacity);
    std::vector<float> radii(event.capacity);
    for (uint32_t i = 0; i < result.outputCount; ++i) {
        running[i] = {float(i + 1), float(i + 2)};
        radii[i] = float(i) * 0.1f;
    }
    morton_checks::upload(transfer, core.adc().gradients(), running);
    morton_checks::upload(transfer, core.adc().radii(), radii);
    core.exportCanonical(buffers, result.outputCount);
    TrainingMortonReorder reorder(device, physical, queue, family);
    (void)reorder.execute(buffers, result.outputCount);
    core.importCanonical(buffers, result.outputCount);
    core.exportCanonical(buffers, result.outputCount);
    const auto after = morton_checks::read<AdamState>(transfer, buffers.adamStatesInfo());
    for (uint32_t i = 0; i < result.outputCount; ++i)
        require(std::memcmp(&after[i], &moments[order[i]], sizeof(AdamState)) == 0,
                "Native Morton boundary changed moment values/permutation");
    auto moved = morton_checks::read<glm::vec2>(transfer, core.adc().gradients());
    auto movedRadius = morton_checks::read<float>(transfer, core.adc().radii());
    for (uint32_t i = 0; i < result.outputCount; ++i) {
        require(moved[i] == running[order[i]] && movedRadius[i] == radii[order[i]],
                "Native running statistics must follow Morton permutation exactly");
    }
    // A subsequent empty interval cannot inherit the injected diagnostic gradients.
    std::fill(running.begin(), running.end(), glm::vec2(0));
    morton_checks::upload(transfer, core.adc().gradients(), running);
    core.importCanonical(buffers, result.outputCount);
    std::fill(running.begin(), running.end(), glm::vec2(0));
    morton_checks::upload(transfer, core.adc().gradients(), running);
    core.adc().prepare(config, 1, 101, false);
    const auto second = core.adc().execute(result.outputCount, true, false);
    require(second.stats.outputCount == result.outputCount && second.stats.cloneSources == 0 &&
                second.stats.splitSources == 0,
            "ADC statistics must be cleared and survive Morton boundary");
    require(second.seed != event.seed, "ADC RNG must advance per real step");
    return 2;
}

inline uint32_t projectionStorage(vk::Device device, vk::PhysicalDevice physical, vk::Queue queue,
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
    for (uint32_t count : {0u, 1u, 31u, 32u, 33u, 65u, 257u}) {
        constexpr TrainingExtent extent{33, 31};
        std::vector<ProjectedGaussian> referenceProjected;
        std::vector<glm::vec4> referenceColors;
        std::vector<PixelBlendState> referenceBlends;
        std::vector<uint32_t> referenceIDs;
        std::vector<glm::uvec4> referenceRanges;
        std::vector<ProjectedGaussianGrad> referenceGradients;
        uint32_t referenceCount = 0;
        for (bool soa : {false, true}) {
            TrainingBuffers b;
            b.initialize(device, physical, queue, family, family);
            b.resize(std::max(count, 1u), extent);
            std::vector<GaussianTrainParam> g(std::max(count, 1u));
            for (uint32_t i = 0; i < g.size(); ++i) {
                g[i].positionOpacity = {float(i % 7) * 0.04f - 0.1f, -0.12f,
                                        2.0f + float(i % 5) * 0.01f, -0.3f};
                if (i % 11 == 10) g[i].positionOpacity.z = -1;
                g[i].scale = {-2.0f, -2.2f, -2.5f, 0};
                g[i].rotation = {0, 0, 0, 1};
                for (uint32_t k = 0; k < 12; ++k)
                    g[i].sh[k] = glm::vec4(0.001f * float(k + i % 3));
            }
            if (count) b.uploadGaussianParams(g.data(), count);
            TrainingVkSplatBackend core(device, physical, queue, family, false, soa);
            core.importCanonical(b, count);
            TrainingForwardCamera camera{};
            camera.view = camera.model = glm::mat4(1);
            camera.viewport = {0, 0, 33, 31};
            camera.focalTan = {17, 19, 33.0f / 34, 31.0f / 38};
            b.uploadCameraWithIntrinsics(camera, {{17, 19, 15.7f, 14.8f}});
            GaussianForwardRenderer forward;
            struct ForwardCleanup {
                GaussianForwardRenderer& renderer;
                ~ForwardCleanup() {
                    renderer.cleanup();
                }
            } cleanup{forward};
            forward.initialize(device, physical, queue, family, count, extent);
            forward.setFasterSortEnabled(true);
            forward.setVkSplatBackend(&core, false);
            TrainingPushConstants push{};
            push.gaussianCount = count;
            push.width = 33;
            push.height = 31;
            push.pixelCount = 33 * 31;
            push.activeSHDegree = 3;
            command.reset();
            command.begin(
                vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
            forward.setTrainingBuffers(b, command, push);
            forward.prepareTileItems();
            b.recordTileItemCountReadback(command);
            submit();
            push.tileItemCount = b.requiredTileItemCount();
            if (push.tileItemCount > b.tileItemCapacity())
                b.resizeTileItems(push.tileItemCount + 256);
            command.reset();
            command.begin(
                vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
            forward.setTrainingBuffers(b, command, push);
            forward.renderPreparedTiles(push.tileItemCount);
            submit();
            auto ranges = morton_checks::read<glm::uvec4>(transfer, b.tileRangesInfo());
            auto ids = morton_checks::read<uint32_t>(transfer, b.tileItemsInfo());
            ids.resize(push.tileItemCount);
            auto colors = morton_checks::read<glm::vec4>(transfer, b.renderedColorInfo());
            auto blends = morton_checks::read<PixelBlendState>(transfer, b.pixelBlendStatesInfo());
            std::vector<ProjectedGaussian> projected(count);
            if (!soa) {
                auto values = morton_checks::read<ProjectedGaussian>(transfer, b.projectedInfo());
                std::copy_n(values.begin(), count, projected.begin());
            } else {
                auto info = core.projectionInfos();
                auto xy = morton_checks::read<glm::vec2>(transfer, info[0]);
                auto depth = morton_checks::read<float>(transfer, info[1]);
                auto conic = morton_checks::read<glm::vec4>(transfer, info[2]);
                auto rgb = morton_checks::read<float>(transfer, info[3]);
                auto radius = morton_checks::read<int32_t>(transfer, info[5]);
                auto counts = morton_checks::read<uint32_t>(transfer, info[6]);
                for (uint32_t i = 0; i < count; ++i)
                    if (counts[i]) {
                        projected[i].centerRadius = {xy[i].x, xy[i].y, depth[i], float(radius[i])};
                        projected[i].conicOpacity = conic[i];
                        projected[i].color = {rgb[3 * i], rgb[3 * i + 1], rgb[3 * i + 2], 1};
                    }
            }
            if (!soa) {
                referenceProjected = projected;
                referenceIDs = ids;
                referenceRanges = ranges;
                referenceColors = colors;
                referenceBlends = blends;
                referenceCount = push.tileItemCount;
            } else {
                require(push.tileItemCount == referenceCount, "SoA overlap count differs");
                morton_checks::exact(projected, referenceProjected,
                                     "SoA valid projection/depth bits differ");
                morton_checks::exact(ids, referenceIDs, "SoA exact sort IDs differ");
                morton_checks::exact(ranges, referenceRanges, "SoA exact tile ranges differ");
                morton_checks::exact(colors, referenceColors,
                                     "SoA reference composite bits differ");
                morton_checks::exact(blends, referenceBlends, "SoA backward pixel states differ");
            }
            command.reset();
            command.begin(
                vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
            core.recordComposite({b, command, push}, true);
            submit();
            auto masked = morton_checks::read<glm::vec4>(transfer, b.renderedColorInfo());
            auto maskedBlends =
                morton_checks::read<PixelBlendState>(transfer, b.pixelBlendStatesInfo());
            for (size_t i = 0; i < colors.size(); ++i)
                for (uint32_t c = 0; c < 4; ++c)
                    close(masked[i][c], colors[i][c], "SoA/reference subtile changed colors");
            morton_checks::exact(maskedBlends, blends,
                                 "SoA/reference subtile changed backward state");
            if (soa) {
                auto info = core.projectionInfos();
                command.reset();
                command.begin(
                    vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
                for (auto descriptor : info)
                    command.fillBuffer(descriptor.buffer, 0, descriptor.range, 0x3f123456u);
                vk::MemoryBarrier fillVisible{vk::AccessFlagBits::eTransferWrite,
                                              vk::AccessFlagBits::eShaderRead |
                                                  vk::AccessFlagBits::eShaderWrite};
                command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                        vk::PipelineStageFlagBits::eComputeShader, {}, fillVisible,
                                        {}, {});
                forward.setTrainingBuffers(b, command, push);
                forward.prepareTileItems();
                b.recordTileItemCountReadback(command);
                submit();
                require(b.requiredTileItemCount() == push.tileItemCount,
                        "Repeated SoA overlap count changed");
                command.reset();
                command.begin(
                    vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
                forward.setTrainingBuffers(b, command, push);
                forward.renderPreparedTiles(push.tileItemCount);
                submit();
                constexpr std::array<uint32_t, 7> words{2, 1, 4, 3, 2, 1, 1};
                for (uint32_t field = 0; field < 7; ++field) {
                    const auto values = morton_checks::read<uint32_t>(transfer, info[field]);
                    for (size_t i = size_t(count) * words[field]; i < values.size(); ++i)
                        require(values[i] == 0x3f123456u,
                                "SoA projection overwrote capacity tail canary");
                }
                auto repeatedIDs = morton_checks::read<uint32_t>(transfer, b.tileItemsInfo());
                repeatedIDs.resize(push.tileItemCount);
                morton_checks::exact(repeatedIDs, referenceIDs,
                                     "Repeated SoA sort changed exact IDs");
            }
            if (TrainingVkSplatBackend::supportsLifecycle(physical) && count) {
                std::vector<PixelGrad> pixels(33 * 31);
                for (auto& v : pixels)
                    v.color = {0.01f, -0.02f, 0.03f, 0};
                morton_checks::upload(transfer, b.pixelGradsInfo(), pixels);
                command.reset();
                command.begin(
                    vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
                command.fillBuffer(b.projectedGradsInfo().buffer, 0, VK_WHOLE_SIZE, 0);
                vk::MemoryBarrier ready{vk::AccessFlagBits::eTransferWrite,
                                        vk::AccessFlagBits::eShaderRead |
                                            vk::AccessFlagBits::eShaderWrite};
                command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                        vk::PipelineStageFlagBits::eComputeShader, {}, ready, {},
                                        {});
                core.recordPixel({b, command, push}, true);
                submit();
                auto gradient =
                    morton_checks::read<ProjectedGaussianGrad>(transfer, b.projectedGradsInfo());
                gradient.resize(count);
                if (!soa)
                    referenceGradients = gradient;
                else
                    for (uint32_t i = 0; i < count; ++i) {
                        auto actual = std::bit_cast<std::array<float, 12>>(gradient[i]);
                        auto expected = std::bit_cast<std::array<float, 12>>(referenceGradients[i]);
                        for (uint32_t k = 0; k < 12; ++k)
                            close(actual[k], expected[k], "SoA Tensor gradient differs");
                    }
            }
        }
        ++cases;
    }
    return cases;
}

inline uint32_t lifecycle(vk::Device device, vk::PhysicalDevice physical, vk::Queue queue,
                          uint32_t family, bool reuse = false, bool soa = false,
                          bool onCheck = false) {
    GaussianTraining training;
    training.initialize(device, physical, queue, family, family);
    training.setVkSplatCompute(true);
    training.setVkSplatOptimizations(reuse, soa);
    training.setLossOnCheck(onCheck);
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
    training.setValidationInterval(onCheck ? 3u : 1u);
    uint32_t adoptions = 0, resets = 0;
    uint32_t previousCount = training.gaussianCount();
    bool grew = false;
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
        require(observation.gaussiansBefore == previousCount,
                "Native next-step count differs from adopted count");
        if (observation.densification) {
            const auto& stats = training.densificationStats();
            require(stats.outputCount > 0 && training.gaussianCount() == stats.outputCount,
                    "Native lifecycle did not adopt actual nonempty ADC output");
            require(stats.outputCount == previousCount + stats.cloneSources + stats.splitSources -
                                             stats.prunedSources,
                    "Native lifecycle count conservation failed");
            grew |= stats.outputCount > previousCount;
        }
        previousCount = training.gaussianCount();
        adoptions += observation.densification;
        if (observation.opacityReset)
            require(training.inspectNativeOpacityReset(),
                    "Native lifecycle reset must change opacity and clear both opacity moments");
        resets += observation.opacityReset;
        require(!training.buffers().gaussianParamsInfo().buffer,
                "Native lifecycle retained canonical model");
    }
    require(grew && adoptions == 3 && resets == 1 && training.mortonStats().reorderCount == 1,
            "Native ADC/reset/Morton lifecycle coverage incomplete");
    require(training.inspectTrainingStateFinite().passed(),
            "Native final finite inspection failed");
    require(training.lastLossStep() == 8, "Explicit finite inspection did not refresh scalar Loss");
    adc.densifyUntilIteration = 12;
    adc.densifyGradThreshold = 1e6f;
    adc.minOpacity = 0.9f;
    training.setDensificationConfig(adc);
    training.trainStep();
    bool emptyRejected = false;
    try {
        training.trainStep();
    } catch (const VkSplatAdcEmptyResult& e) {
        emptyRejected = e.result.stats.outputCount == 0 && training.gaussianCount() == 0 &&
                        training.densificationStats().outputCount == 0 &&
                        !training.hasTrainableModel();
    }
    require(emptyRejected,
            "Empty native ADC must propagate zero count and stop the training model");
    return 9;
}
} // namespace vksplat_backend_checks
