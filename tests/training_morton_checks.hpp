#pragma once

#include "training/core/gaussian_training.hpp"
#include "training/core/training_morton_reorder.hpp"
#include "training/core/training_optimizer_capture_gpu.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <sstream>

namespace morton_checks {
using namespace vulkan3DGS;
inline void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
inline uint64_t encode(glm::vec3 point, glm::vec3 minimum, float side) {
    const float inverse = side > 0 ? 1.0f / side : 0;
    uint64_t result = 0;
    for (int axis = 0; axis < 3; ++axis) {
        const float normalized = std::clamp((point[axis] - minimum[axis]) * inverse, 0.0f, 1.0f);
        const uint32_t coordinate = static_cast<uint32_t>(normalized * 2097151.0f);
        for (uint32_t bit = 0; bit < 21; ++bit)
            result |= uint64_t((coordinate >> bit) & 1u) << (3u * bit + uint32_t(axis));
    }
    return result;
}
inline void policy() {
    TrainingMortonStats stats;
    uint32_t events = 0;
    for (uint32_t n = 0; n <= 30000; ++n) {
        require(!trainingMortonDue(false, false, n, stats), "Disabled Morton event");
        require(!trainingMortonDue(true, true, n, stats), "Fixed Benchmark mutated model order");
        if (trainingMortonDue(true, false, n, stats)) {
            require(n == events * 5000 && events < 4, "Morton event schedule mismatch");
            ++stats.reorderCount;
            stats.lastCompletedSteps = n;
            require(!trainingMortonDue(true, false, n, stats), "Repeated event after pause/resume");
            ++events;
        }
    }
    require(events == 4, "Morton schedule must contain all four upstream events");
    require(encode({1, 0, 0}, {}, 1) == 0x1249249249249249ull, "Morton X bits");
    require(encode({0, 1, 0}, {}, 1) == 0x2492492492492492ull, "Morton Y bits");
    require(encode({0, 0, 1}, {}, 1) == 0x4924924924924924ull, "Morton Z bits");
    require(encode({1, 1, 1}, {}, 1) == 0x7fffffffffffffffull, "Full 63-bit Morton range");
    require(encode({3, 4, 5}, {3, 4, 5}, 0) == 0, "Degenerate Morton cube");
    require(encode({1, 0, 0}, {}, 2) == encode({0.5f, 0, 0}, {}, 1), "Longest-side normalization");
}
template <class T>
std::vector<T> read(OptimizerCaptureTransfer& transfer, vk::DescriptorBufferInfo info) {
    std::vector<T> data(size_t(info.range / sizeof(T)));
    auto bytes = std::as_writable_bytes(std::span(data));
    for (size_t offset = 0; offset < bytes.size();) {
        auto part = bytes.subspan(
            offset, std::min(bytes.size() - offset, size_t(optimizerCaptureChunkBytes)));
        transfer.download(info, offset, part);
        offset += part.size();
    }
    return data;
}
template <class T>
void upload(OptimizerCaptureTransfer& transfer, vk::DescriptorBufferInfo info,
            const std::vector<T>& data) {
    auto bytes = std::as_bytes(std::span(data));
    for (size_t offset = 0; offset < bytes.size();) {
        auto part = bytes.subspan(
            offset, std::min(bytes.size() - offset, size_t(optimizerCaptureChunkBytes)));
        transfer.upload(info, offset, part);
        offset += part.size();
    }
}
template <class T>
void exact(const std::vector<T>& actual, const std::vector<T>& expected, const char* message) {
    require(actual.size() == expected.size() &&
                std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(T)) == 0,
            message);
}
inline uint32_t lifecycle(vk::Device device, vk::PhysicalDevice physical, vk::Queue queue,
                          uint32_t family) {
    GaussianTraining training;
    training.setMortonOrderingEnabled(true);
    training.setFasterSortEnabled(true);
    training.setPixelTo2DGSMode(TrainingPixelTo2DGSMode::Direct);
    training.setStepObservationEnabled(true);
    training.initialize(device, physical, queue, family, family);
    training.resize(4, {16, 16});
    training.initializeTrainingRenderers(device, physical, queue, family, 4, {16, 16});
    TrainingForwardCamera camera{};
    camera.view = camera.model = camera.projection = glm::mat4(1);
    camera.viewport = {16, 16, 1.0f / 16, 1.0f / 16};
    camera.focalTan = {8, 8, 1, 1};
    training.buffers().uploadCamera(camera);
    std::array<uint32_t, 256> target{};
    training.buffers().uploadTargetColor(reinterpret_cast<const uint8_t*>(target.data()), 16, 16);
    std::array<GaussianTrainParam, 4> seed{};
    for (uint32_t i = 0; i < seed.size(); ++i) {
        seed[i].positionOpacity = {0.3f - float(i) * 0.2f, 0, 2, i == 3 ? -30.0f : 0.0f};
        seed[i].scale = glm::vec4(i == 0 ? -8.0f : -4.0f);
        seed[i].rotation = {0, 0, 0, 1};
    }
    training.buffers().uploadGaussianParams(seed.data(), 4);
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
    require(training.mortonStats().reorderCount == 0,
            "Renderer initialization reported Morton execution");
    uint32_t steps = 0, adcSteps = 0, resets = 0, clones = 0, splits = 0, prunes = 0;
    for (uint32_t i = 1; i <= 8; ++i) {
        if (i == 2) training.setMortonOrderingEnabled(false);
        if (i == 3) training.setMortonOrderingEnabled(true);
        if (i == 4) {
            training.initializeTrainingRenderers(device, physical, queue, family,
                                                 training.gaussianCount(), {16, 16});
            require(training.mortonOrderingEnabled() && training.mortonStats().reorderCount == 1,
                    "Renderer rebuild lost Morton preference/history");
        }
        training.trainStep();
        ++steps;
        const auto& observation = training.lastStepObservation();
        require(observation.mortonReordered == (i == 1) &&
                    observation.mortonOrderingRequested == (i != 2) &&
                    observation.morton.reorderCount == 1,
                "Completed-step Morton metadata mismatch");
        if (i == 1) {
            const auto& profile = training.profilingStats();
            const auto& reorderCpu = profile.cpu[size_t(TrainingCpuProfileStage::MortonReorder)];
            require(reorderCpu.sampleCount == 1 &&
                        profile.cpu[size_t(TrainingCpuProfileStage::Total)].lastMs >=
                            reorderCpu.lastMs,
                    "Z-order cost missing from total step CPU timing");
        }
        require(training.trainingIteration() == i && training.mortonStats().reorderCount == 1,
                "Synthetic pause/resume changed event count or skipped a step");
        require(training.validationStats().valid &&
                    training.validationStats().nonFiniteGaussianCount == 0,
                "Morton downstream validation failed");
        require(observation.activeSHDegree == std::min(i, 3u), "Morton changed SH schedule");
        adcSteps += observation.densification;
        if (observation.densification) {
            const auto& actual = training.densificationStats();
            clones += actual.cloneSources;
            splits += actual.splitSources;
            prunes += actual.prunedSources;
        }
        resets += observation.opacityReset;
    }
    require(adcSteps == 3 && resets == 1, "Morton lifecycle missed ADC/reset events");
    require(clones && splits && prunes, "Morton lifecycle did not exercise clone/split/prune");
    const auto finite = training.inspectTrainingStateFinite();
    require(finite.passed(), "Morton synthetic downstream produced non-finite state");
    const auto beforeExport = training.buffers().downloadGaussianParams(training.gaussianCount());
    const auto path =
        std::filesystem::temp_directory_path() /
        ("morton-export-" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".ply");
    require(training.exportToPLY(path), "Morton model export failed");
    std::filesystem::remove(path);
    exact(training.buffers().downloadGaussianParams(training.gaussianCount()), beforeExport,
          "Morton export modified live model order");
    training.cleanup();
    require(training.mortonOrderingEnabled() && !training.mortonStats().reorderCount,
            "Model cleanup lost preference or retained old Morton history");
    training.initialize(device, physical, queue, family, family);
    training.resize(1, {16, 16});
    require(trainingMortonDue(training.mortonOrderingEnabled(), false, 0, training.mortonStats()),
            "New model did not reset initial Morton event");
    return steps;
}

inline std::string run(vk::Device device, vk::PhysicalDevice physical, vk::Queue queue,
                       uint32_t family) {
    policy();
    OptimizerCaptureTransfer transfer(device, physical, queue, family);
    TrainingBuffers buffers;
    buffers.initialize(device, physical, queue, family, family);
    TrainingMortonReorder reorder(device, physical, queue, family);
    std::ostringstream report;
    report << "{\"schema\":\"training_morton_checks_v1\",\"cases\":[";
    uint32_t caseIndex = 0;
    for (uint32_t count : {0u, 1u, 255u, 256u, 257u, 822u, 32769u, 2u}) {
        buffers.resize(std::max(count, 1u), {16, 16});
        for (uint32_t shape = 0; shape < 3; ++shape) {
            if (shape == 2 && count) {
                buffers.ensureDensificationCapacity(std::max(count * 2, 1u));
                buffers.adoptDensifiedGaussians(count);
            }
            const auto paramsInfo = buffers.gaussianParamsInfo();
            const auto adamInfo = buffers.adamStatesInfo();
            const auto denseInfo = buffers.densificationStatesInfo();
            std::vector<GaussianTrainParam> params(
                size_t(paramsInfo.range / sizeof(GaussianTrainParam)));
            std::vector<AdamState> adam(size_t(adamInfo.range / sizeof(AdamState)));
            std::vector<GaussianDensificationState> dense(
                size_t(denseInfo.range / sizeof(GaussianDensificationState)));
            for (size_t i = 0; i < params.size(); ++i) {
                std::array<float, 60> values;
                for (size_t j = 0; j < values.size(); ++j)
                    values[j] = float(i * 64 + j + 1);
                std::memcpy(&params[i], values.data(), sizeof(values));
                if (shape == 1)
                    params[i].positionOpacity = {3, -4, 5, float(i + 1)};
                else
                    params[i].positionOpacity = {float((i * 13) % 17) - 8, float((i * 7) % 9) - 4,
                                                 float((i * 3) % 5) - 2, float(i + 1)};
                std::array<float, 120> moments;
                for (size_t j = 0; j < moments.size(); ++j)
                    moments[j] = float(i * 128 + j + 1) / 16;
                std::memcpy(&adam[i], moments.data(), sizeof(moments));
            }
            for (size_t i = 0; i < dense.size(); ++i)
                dense[i] = {float(i) + 0.25f, uint32_t(i + 7), float(i) + 0.5f, uint32_t(i + 23)};
            upload(transfer, paramsInfo, params);
            upload(transfer, adamInfo, adam);
            upload(transfer, denseInfo, dense);
            std::vector<uint32_t> permutation(count);
            std::iota(permutation.begin(), permutation.end(), 0);
            glm::vec3 minimum{}, maximum{};
            if (count) {
                minimum = maximum = glm::vec3(params[0].positionOpacity);
                for (uint32_t i = 1; i < count; ++i) {
                    minimum = glm::min(minimum, glm::vec3(params[i].positionOpacity));
                    maximum = glm::max(maximum, glm::vec3(params[i].positionOpacity));
                }
            }
            const auto extent = maximum - minimum;
            const float side = std::max({extent.x, extent.y, extent.z});
            std::stable_sort(permutation.begin(), permutation.end(), [&](auto a, auto b) {
                return encode(glm::vec3(params[a].positionOpacity), minimum, side) <
                       encode(glm::vec3(params[b].positionOpacity), minimum, side);
            });
            auto expectedParams = params;
            auto expectedAdam = adam;
            auto expectedDense = dense;
            for (uint32_t i = 0; i < count; ++i) {
                expectedParams[i] = params[permutation[i]];
                expectedAdam[i] = adam[permutation[i]];
                expectedDense[i] = dense[permutation[i]];
            }
            const auto result = reorder.execute(buffers, count);
            auto actualParams = read<GaussianTrainParam>(transfer, buffers.gaussianParamsInfo());
            auto actualAdam = read<AdamState>(transfer, buffers.adamStatesInfo());
            auto actualDense =
                read<GaussianDensificationState>(transfer, buffers.densificationStatesInfo());
            exact(actualParams, expectedParams, "Morton params/tail/padding permutation mismatch");
            exact(actualAdam, expectedAdam, "Morton nonzero moments/tail permutation mismatch");
            exact(actualDense, expectedDense, "Morton ADC statistics/tail permutation mismatch");
            for (uint32_t i = 0; i < count; ++i) {
                expectedParams[permutation[i]] = actualParams[i];
                expectedAdam[permutation[i]] = actualAdam[i];
                expectedDense[permutation[i]] = actualDense[i];
            }
            exact(expectedParams, params, "Morton inverse params mismatch");
            exact(expectedAdam, adam, "Morton inverse moments mismatch");
            exact(expectedDense, dense, "Morton inverse statistics mismatch");
            if (caseIndex++) report << ',';
            report << "{\"count\":" << count << ",\"shape\":" << shape
                   << ",\"bitwise_state\":true,\"inverse\":true,\"capacity_tail\":true,\"workspace_"
                      "bytes\":"
                   << result.workspaceBytes << '}';
        }
    }
    buffers.resize(2, {16, 16});
    for (float value : {std::numeric_limits<float>::quiet_NaN(), INFINITY, -INFINITY}) {
        std::vector<GaussianTrainParam> params(buffers.gaussianCapacity());
        params[0].positionOpacity = {value, 0, 1, 0};
        upload(transfer, buffers.gaussianParamsInfo(), params);
        const auto beforeInfo = buffers.gaussianParamsInfo();
        const auto beforeAdam = read<AdamState>(transfer, buffers.adamStatesInfo());
        const auto beforeDense =
            read<GaussianDensificationState>(transfer, buffers.densificationStatesInfo());
        bool refused = false;
        try {
            (void)reorder.execute(buffers, 2);
        } catch (const std::runtime_error&) {
            refused = true;
        }
        require(refused && beforeInfo.buffer == buffers.gaussianParamsInfo().buffer,
                "Non-finite Morton input adopted");
        exact(read<GaussianTrainParam>(transfer, buffers.gaussianParamsInfo()), params,
              "Rejected Morton input modified parameters");
        exact(read<AdamState>(transfer, buffers.adamStatesInfo()), beforeAdam,
              "Rejected Morton input modified moments");
        exact(read<GaussianDensificationState>(transfer, buffers.densificationStatesInfo()),
              beforeDense, "Rejected Morton input modified ADC");
    }
    const auto syntheticSteps = lifecycle(device, physical, queue, family);
    report << "],\"nonfinite_refusals\":3,\"schedule_events\":4,\"synthetic_owner_steps\":"
           << syntheticSteps << ",\"real_training_executed\":false}";
    return report.str();
}
} // namespace morton_checks
