#include "training/core/backward/projection_optimizer_pass.hpp"
#include "training_optimizer_gpu_session.hpp"
#include "training_optimizer_r1_evaluation.hpp"
#include "training_optimizer_reference.hpp"
#include "utils/logger.hpp"

#include <atomic>
#include <charconv>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>

namespace {
using optimizer_gpu::check;
std::atomic<uint32_t> validationErrors{0};
using optimizer_candidate::Method;
struct CandidateProgress {
    uint32_t isolatedCases = 0, sequenceSteps = 0;
};
VKAPI_ATTR vk::Bool32 VKAPI_CALL onMessage(vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
                                           vk::DebugUtilsMessageTypeFlagsEXT,
                                           const vk::DebugUtilsMessengerCallbackDataEXT* data,
                                           void*) {
    if (severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eError) {
        ++validationErrors;
        std::cerr << "S4 Vulkan validation: " << data->pMessage << '\n';
    }
    return VK_FALSE;
}
std::string quoted(std::string text) {
    std::string out = "\"";
    for (auto c : text) {
        if (c == '\\' || c == '"') out += '\\';
        out += static_cast<unsigned char>(c) < 32 ? '?' : c;
    }
    return out + '"';
}
void reportFile(const std::filesystem::path& path, const std::string& report) {
    FILE* raw = nullptr;
#ifdef _WIN32
    _wfopen_s(&raw, path.c_str(), L"wbx");
#else
    raw = std::fopen(path.c_str(), "wbx");
#endif
    check(raw != nullptr, "S4 report already exists or cannot be created");
    std::unique_ptr<FILE, decltype(&std::fclose)> file(raw, &std::fclose);
    check(std::fwrite(report.data(), 1, report.size(), raw) == report.size() &&
              std::fflush(raw) == 0,
          "S4 report write failed");
}
struct Fixture {
    OptimizerCaptureHeader h{};
    std::array<std::vector<std::byte>, 9> blocks;
    std::vector<GaussianTrainParam> params, expectedParams;
    std::vector<AdamState> states, expectedStates;
    std::vector<GaussianDensificationState> dense, expectedDense;
    OptimizerCaptureSource source() const {
        return [this](size_t b, uint64_t offset, std::span<std::byte> dest) {
            check(b < 9 && offset + dest.size() <= blocks[b].size(), "S4 synthetic bounds");
            std::memcpy(dest.data(), blocks[b].data() + offset, dest.size());
        };
    }
};
template <class T> void pack(std::vector<std::byte>& dest, const std::vector<T>& src) {
    dest.resize(src.size() * sizeof(T));
    if (!dest.empty()) std::memcpy(dest.data(), src.data(), dest.size());
}
Fixture fixture(uint32_t count, uint32_t degree, uint32_t pattern) {
    Fixture f;
    f.h.gaussianCount = f.h.capacity = f.h.push.gaussianCount = count;
    f.h.synthetic = 1;
    f.h.push.width = f.h.push.height = 16;
    f.h.push.pixelCount = 256;
    f.h.push.activeSHDegree = degree;
    f.h.completedIteration = f.h.completedOptimizerStep = 1;
    f.h.camera.viewport = {0, 0, 16, 16};
    f.h.camera.focalTan = {8, 8, 1, 1};
    f.h.camera.cameraPosition = {0, 0, 0, 1};
    f.h.push.optimizerInvFirstMomentCorrection = 10;
    f.h.push.optimizerInvSecondMomentCorrection = 1000;
    if (pattern == 1) f.h.push.optimizerGradClip = 0.0001f;
    if (pattern == 2) {
        f.h.push.positionLearningRate = f.h.push.opacityLearningRate = 0;
        f.h.push.scaleLearningRate = f.h.push.rotationLearningRate = 0;
        f.h.push.featureLearningRate = f.h.push.featureRestLearningRate = 0;
    }
    if (pattern == 3) {
        f.h.push.optimizerBeta1 = f.h.push.optimizerBeta2 = 0;
        f.h.push.optimizerOneMinusBeta1 = f.h.push.optimizerOneMinusBeta2 = 1;
        f.h.push.optimizerInvFirstMomentCorrection = f.h.push.optimizerInvSecondMomentCorrection =
            1;
        f.h.push.optimizerEpsilon = 1e-4f;
    }
    if (pattern == 4) {
        f.h.push.optimizerBeta1 = f.h.push.optimizerBeta2 = 0.999999f;
        f.h.push.optimizerOneMinusBeta1 = f.h.push.optimizerOneMinusBeta2 = 1 - 0.999999f;
        f.h.push.optimizerInvFirstMomentCorrection = f.h.push.optimizerInvSecondMomentCorrection =
            1000000;
    }
    if (pattern == 6) f.h.camera.cameraPosition = {0, 0, 2, 1};
    f.params.resize(count);
    f.states.resize(count);
    f.dense.resize(count);
    std::vector<ProjectedGaussian> projected(count);
    std::vector<ProjectedGaussianGrad> projectedGrads(count);
    std::vector<glm::uvec2> ranges(count);
    for (uint32_t i = 0; i < count; ++i) {
        auto& g = f.params[i];
        auto& a = f.states[i];
        g.positionOpacity = {0.1f, -0.2f, 2, 0};
        g.scale = {-2, -2, -2, 0.25f};
        g.rotation = {0, 0, 0, 1};
        for (uint32_t block = 0; block < 12; ++block)
            g.sh[block] = glm::vec4(0.01f);
        for (size_t word = 0; word < sizeof(GaussianGrad) / sizeof(float); ++word) {
            const float first = (word % 2 ? -1.f : 1.f) * 0.0002f;
            const float second = pattern == 3 ? 0.f : 0.0004f;
            std::memcpy(reinterpret_cast<std::byte*>(&a.firstMoment) + word * 4, &first, 4);
            std::memcpy(reinterpret_cast<std::byte*>(&a.secondMoment) + word * 4, &second, 4);
        }
        f.dense[i] = {0.125f, 7u, 4.f, 0x12345678u};
        const bool support = i % 3 != 0;
        projected[i].centerRadius = {8, 8, 2, support ? 4.f : 0.f};
        projected[i].conicOpacity = {1, 0, 1, 0.5f};
        projected[i].color = {0.5f, 0.5f, 0.5f, 1};
        ranges[i] = {0, i % 2};
        // Opacity-only known derivative: no shared CPU projection implementation.
        projectedGrads[i].conicOpacity.w = pattern == 3 ? (i % 2 ? -1e-18f : 1e-18f)
                                           : i % 4 == 0 ? 0.f
                                                        : (i % 2 ? -0.03f : 0.02f);
        if (pattern >= 5) {
            // Full projection/SH derivatives, clamp masks and nontrivial quaternion.
            // Compare to ORIGINAL GPU output, not an incomplete CPU projection oracle.
            g.rotation = {0.1f, -0.2f, 0.05f, 1.f};
            if (i % 4 == 0) g.sh[0].x = -2.f;
            projectedGrads[i].centerRadius = {0.02f, -0.03f, 0.001f, 0.f};
            projectedGrads[i].conicOpacity = {-0.0002f, 0.0001f, 0.0003f, -0.02f};
            projectedGrads[i].color = {0.02f, -0.01f, 0.03f, 0.f};
        }
        if (pattern == 6) {
            constexpr std::array<float, 5> distances{0.f, 1e-7f, 0.999e-6f, 1.001e-6f, 1e-5f};
            g.positionOpacity = {distances[i % distances.size()], 0, 2, 0};
        }
        if (pattern == 7) {
            for (auto& pack : g.sh)
                pack = glm::vec4(0);
            const float boundary = -0.5f / 0.28209479177387814f;
            const std::array<float, 3> values{std::nextafter(boundary, -INFINITY), boundary,
                                              std::nextafter(boundary, INFINITY)};
            // SH coefficients use planar RGB packs: coefficient zero occupies packs 0/4/8.
            for (uint32_t channel = 0; channel < 3; ++channel)
                g.sh[channel * 4].x = values[(i + channel) % values.size()];
        }
        if (pattern == 8) {
            g.positionOpacity = {float(int(i % 11) - 5) * 0.13f, float(int(i % 7) - 3) * 0.17f,
                                 1.5f + float(i % 5) * 0.2f, 0};
            for (uint32_t pack = 0; pack < 12; ++pack)
                for (uint32_t lane = 0; lane < 4; ++lane)
                    g.sh[pack][lane] = float(int((i + pack * 7 + lane * 3) % 31) - 15) * 0.13f;
        }
    }
    f.expectedParams = f.params;
    f.expectedStates = f.states;
    f.expectedDense = f.dense;
    for (uint32_t i = 0; i < count; ++i) {
        GaussianGrad grad{};
        if (projected[i].centerRadius.w > 0) {
            grad.positionOpacity.w = projectedGrads[i].conicOpacity.w * 0.25f;
            if (ranges[i].y) ++f.expectedDense[i].screenGradCount;
        }
        optimizer_reference::adam(f.expectedParams[i], f.expectedStates[i], grad, f.h.push);
    }
    pack(f.blocks[0], f.params);
    pack(f.blocks[1], f.states);
    pack(f.blocks[2], projected);
    pack(f.blocks[3], projectedGrads);
    pack(f.blocks[4], f.dense);
    pack(f.blocks[5], ranges);
    pack(f.blocks[6], f.expectedParams);
    pack(f.blocks[7], f.expectedStates);
    pack(f.blocks[8], f.expectedDense);
    for (size_t b = 0; b < 9; ++b) {
        f.h.blocks[b].bytes = f.blocks[b].size();
        f.h.blocks[b].checksum = optimizerHash(f.blocks[b]);
    }
    return f;
}
void cpuCheck(optimizer_gpu::Session& session, const Fixture& f) {
    std::vector<std::byte> actual(optimizerCaptureChunkBytes);
    for (size_t block = 6; block < 9; ++block) {
        const uint64_t length = f.blocks[block].size();
        for (uint64_t offset = 0; offset < length;) {
            const size_t n = size_t(std::min<uint64_t>(actual.size(), length - offset));
            session.readOutput(block, offset, std::span(actual).first(n));
            for (size_t i = 0; i < n; i += 4) {
                float gpu, cpu;
                std::memcpy(&gpu, actual.data() + i, 4);
                std::memcpy(&cpu, f.blocks[block].data() + offset + i, 4);
                if (block == 8) {
                    check(std::memcmp(actual.data() + i, f.blocks[block].data() + offset + i, 4) ==
                              0,
                          "S4 CPU densification/padding mismatch");
                } else if (block == 6) {
                    float old;
                    std::memcpy(&old, f.blocks[0].data() + offset + i, 4);
                    const double delta = double(cpu) - old;
                    const float next = std::nextafter(cpu, INFINITY);
                    const double allowance =
                        4 * std::abs(double(next) - cpu) + std::abs(delta) * 2e-6;
                    check(std::isfinite(gpu) && std::abs((double(gpu) - old) - delta) <= allowance,
                          "S4 CPU parameter UPDATE delta mismatch");
                } else {
                    check(std::isfinite(gpu) &&
                              std::abs(double(gpu) - cpu) <= 5e-6 * std::abs(double(cpu)) + 1e-35,
                          "S4 CPU first/second moment mismatch");
                }
            }
            offset += n;
        }
    }
}
void captureHook(vk::Device device, vk::PhysicalDevice physical, vk::Queue queue, uint32_t family,
                 const std::filesystem::path& shader,
                 const std::filesystem::path& syntheticOutput = {}) {
    const auto dir = std::filesystem::temp_directory_path() /
                     ("v3dgs-optimizer-hook-" +
                      std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(dir);
    try {
        for (uint32_t count : {1u, 257u}) {
            std::cerr << "S4 hook G=" << count << ": creating fixture/buffers\n";
            auto fixtureData = fixture(count, 2, 1);
            fillOptimizerCaptureIdentity(fixtureData.h, physical, shader);
            TrainingBuffers buffers;
            buffers.initialize(device, physical, queue, family, family);
            buffers.resize(count, {16, 16});
            buffers.uploadCamera(fixtureData.h.camera);
            std::cerr << "S4 hook: buffers initialized; uploading exact seeds\n";
            const std::array<vk::DescriptorBufferInfo, 6> sources{
                buffers.gaussianParamsInfo(),      buffers.adamStatesInfo(),
                buffers.projectedInfo(),           buffers.projectedGradsInfo(),
                buffers.densificationStatesInfo(), buffers.gaussianTileRangesInfo()};
            OptimizerCaptureTransfer transfer(device, physical, queue, family);
            for (size_t b = 0; b < sources.size(); ++b)
                transfer.upload(sources[b], 0, fixtureData.blocks[b]);
            const auto path = dir / (std::to_string(count) + ".optimizer.bin");
            TrainingOptimizerCaptureSession capture(device, physical, queue, family, fixtureData.h,
                                                    path);
            bool rejected = false;
            try {
                capture.save({sources[0], sources[1], sources[4]});
            } catch (const std::exception&) {
                rejected = true;
            }
            check(rejected && !std::filesystem::exists(path),
                  "S4 unsubmitted capture exported a post-reference");
            ProjectionOptimizerPass pass;
            capture.markPrepareSubmitted();
            check(capture.submitted() && !capture.optimizerSubmitted() && !capture.completed(),
                  "S4 prepare mutation was not distinguished from an optimizer update");
            std::cerr << "S4 hook: pre clones allocated; creating original pipelines\n";
            pass.initialize(device, shader.parent_path());
            std::cerr << "S4 hook: recording original dispatch\n";
            CommandPool pool;
            pool.create(device, family);
            auto command = pool.allocateCommandBuffer();
            command.begin(
                vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
            BackwardPassContext context{buffers, command, fixtureData.h.push};
            context.optimizerCapture = &capture;
            pass.recordProjectionAndOptimize(context);
            command.end();
            auto fence = device.createFenceUnique({});
            vk::SubmitInfo submit;
            submit.setCommandBuffers(command);
            capture.markSubmitted();
            queue.submit(submit, *fence);
            (void)device.waitForFences(*fence, true, UINT64_MAX);
            capture.markCompleted();
            std::cerr << "S4 hook: dispatch complete; exporting\n";
            capture.save({sources[0], sources[1], sources[4]});
            const auto header = readOptimizerCaptureHeader(path);
            check(std::memcmp(&header.camera, &fixtureData.h.camera, sizeof(header.camera)) == 0 &&
                      std::memcmp(&header.push, &fixtureData.h.push, sizeof(header.push)) == 0,
                  "S4 hook camera/push are not the dispatched bytes");
            for (size_t b = 0; b < 6; ++b)
                check(header.blocks[b].checksum == fixtureData.h.blocks[b].checksum,
                      "S4 hook captured post-state instead of pre-state");
            auto fileSource = [&](size_t block, uint64_t offset, std::span<std::byte> out) {
                readOptimizerCaptureChunk(path, header, block, offset, out);
            };
            if (count == 257 && !syntheticOutput.empty()) {
                writeOptimizerCapture(syntheticOutput, header, fileSource);
                std::cerr << "S4 SYNTHETIC-ONLY fixture saved: " << syntheticOutput << '\n';
            }
            optimizer_gpu::Session replay(device, physical, queue, family, header, fileSource,
                                          shader);
            replay.dispatch();
            const auto difference = replay.compare(fileSource);
            check(!difference.words && !difference.nonFinite,
                  "S4 captured ORIGINAL dispatch did not bitwise replay");
            replay.dispatch();
            check(replay.compare(fileSource).words == 0, "S4 captured original A/A failed");
            replay.verifyReadonly();
            cpuCheck(replay, fixtureData);
            check(!validationErrors,
                  "S4 hook/copy/original/replay synchronization validation failed");
        }
    } catch (...) {
        std::filesystem::remove_all(dir);
        throw;
    }
    std::filesystem::remove_all(dir);
    std::cout << "S4 real dispatch pre-copy/post-reference hook and bitwise file replay passed\n";
}
uint32_t candidateSequences(vk::Device device, vk::PhysicalDevice physical, vk::Queue queue,
                            uint32_t family, const std::filesystem::path& shader,
                            const std::filesystem::path& candidate, CandidateProgress& progress) {
    uint32_t steps = 0;
    for (uint32_t count : {17u, 257u}) {
        std::array<std::vector<std::byte>, 3> previous;
        for (uint32_t degree : {0u, 1u, 2u, 3u, 1u, 0u}) {
            auto f = fixture(count, degree, 5);
            constexpr std::array<size_t, 3> inputs{0, 1, 4};
            if (!previous[0].empty())
                for (size_t i = 0; i < inputs.size(); ++i) {
                    f.blocks[inputs[i]] = previous[i];
                    f.h.blocks[inputs[i]].checksum = optimizerHash(f.blocks[inputs[i]]);
                }
            optimizer_gpu::Session session(device, physical, queue, family, f.h, f.source(), shader,
                                           candidate);
            session.dispatch();
            // Original GPU oracle for THIS deterministic optimizer sequence step.
            // Projected inputs are intentionally fixed; this is not a training sequence.
            for (size_t b = 6; b < 9; ++b)
                session.readOutput(b, 0, f.blocks[b]); // small synthetic counts only
            session.dispatch(true, optimizer_gpu::Variant::Candidate);
            const std::string sequenceLabel =
                "Optimizer candidate state-sequence G=" + std::to_string(count) +
                " SH=" + std::to_string(degree) + " sequenceStep=" + std::to_string(steps);
            optimizer_gpu::requireMatch(session.compare(f.source()), sequenceLabel.c_str());
            session.dispatch();
            optimizer_gpu::requireMatch(session.compare(f.source()),
                                        "Optimizer candidate state-sequence baseline recovery");
            session.verifyReadonly();
            check(!validationErrors,
                  "Optimizer candidate state-sequence Vulkan validation failure");
            for (size_t i = 0; i < 3; ++i)
                previous[i] = std::move(f.blocks[i + 6]);
            ++steps;
            ++progress.sequenceSteps;
        }
    }
    std::cout << "Optimizer candidate 12 optimizer-only sequence steps passed (SH0/1/2/3/1/0); NOT "
                 "full training\n";
    return steps;
}
uint32_t synthetic(vk::Device device, vk::PhysicalDevice physical, vk::Queue queue, uint32_t family,
                   const std::filesystem::path& shader, const std::filesystem::path& candidate,
                   CandidateProgress& progress, Method method) {
    uint32_t cases = 0;
    const uint32_t patterns = candidate.empty() ? 5u : method == Method::SHColorReuse ? 9u : 6u;
    for (uint32_t count : {0u, 1u, 255u, 256u, 257u, 1025u, 8193u}) {
        for (uint32_t degree = 0; degree < 4; ++degree) {
            for (uint32_t pattern = 0; pattern < patterns; ++pattern) {
                auto f = fixture(count, degree, pattern);
                optimizer_gpu::Session session(device, physical, queue, family, f.h, f.source(),
                                               shader, candidate);
                session.dispatch();
                if (pattern < 5) cpuCheck(session, f);
                std::array<std::vector<std::byte>, 3> first;
                for (size_t b = 6; b < 9; ++b) {
                    first[b - 6].resize(f.blocks[b].size());
                    for (uint64_t offset = 0; offset < first[b - 6].size();) {
                        auto span =
                            std::span(first[b - 6])
                                .subspan(size_t(offset),
                                         size_t(std::min<uint64_t>(optimizerCaptureChunkBytes,
                                                                   first[b - 6].size() - offset)));
                        session.readOutput(b, offset, span);
                        offset += span.size();
                    }
                }
                auto reference = [&](size_t b, uint64_t off, std::span<std::byte> out) {
                    std::memcpy(out.data(), first[b - 6].data() + off, out.size());
                };
                session.dispatch();
                auto difference = session.compare(reference);
                check(!difference.words && !difference.nonFinite,
                      "S4 original shader A/A not bitwise repeatable");
                if (!candidate.empty()) {
                    const std::string label = "Optimizer synthetic G=" + std::to_string(count) +
                                              " SH=" + std::to_string(degree) +
                                              " pattern=" + std::to_string(pattern);
                    session.dispatch(true, optimizer_gpu::Variant::Candidate);
                    optimizer_gpu::requireMatch(session.compare(reference), label.c_str());
                    session.dispatch(true, optimizer_gpu::Variant::Candidate);
                    optimizer_gpu::requireMatch(session.compare(reference),
                                                (label + " candidate repeat").c_str());
                    if (pattern < 5) cpuCheck(session, f);
                }
                if (count && pattern == 0) {
                    check(difference.parameterUpdates > 0,
                          "S4 noop/skip-invisible negative control not detected");
                    session.dispatch(false);
                    check(session.compare(reference).words > 0,
                          "S4 params-only restore negative control not detected");
                    session.dispatch();
                    check(session.compare(reference).words == 0,
                          "S4 recovery after negative control failed");
                }
                session.verifyReadonly();
                check(validationErrors == 0, "S4 synthetic Vulkan validation failure");
                ++cases;
                if (!candidate.empty()) ++progress.isolatedCases;
            }
        }
        std::cout << "S4 original fused synthetic G=" << count << " degree0-3, " << patterns
                  << " configurations passed";
        if (!candidate.empty())
            std::cout << " incl. bitwise " << optimizer_candidate::describe(method).name;
        std::cout << '\n';
    }
    if (!candidate.empty())
        cases += candidateSequences(device, physical, queue, family, shader, candidate, progress);
    return cases;
}
} // namespace

int main(int argc, char** argv) {
    std::filesystem::path input, report, syntheticOutput;
    bool requireValidation = false, runSynthetic = false;
    Method method = Method::None;
    bool profileVariantSpecified = false;
    uint32_t profileSeconds = 0;
    uint64_t candidateHash = 0;
    CandidateProgress progress;
    std::string identityJson;
    auto profileVariant = optimizer_gpu::Variant::Candidate;
    std::ostringstream details;
    bool validation = false, syncValidation = false;
    try {
        std::cerr << "S4: parsing independent optimizer diagnostic request\n";
        Logger::get_instance().set_level(LogLevel::WARN_VULKAN_3DGS);
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--synthetic")
                runSynthetic = true;
            else if (arg == "--require-validation")
                requireValidation = true;
            else if (arg == "--candidate-r1" || arg == "--candidate-sh-color-reuse")
                method = optimizer_candidate::select(method, arg);
            else if (arg == "--profile-r1-seconds" && i + 1 < argc) {
                const std::string value = argv[++i];
                const auto parsed =
                    std::from_chars(value.data(), value.data() + value.size(), profileSeconds);
                check(parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size() &&
                          profileSeconds >= 1 && profileSeconds <= 30,
                      "--profile-r1-seconds requires an integer in [1,30]");
            } else if (arg == "--profile-variant" && i + 1 < argc) {
                const std::string value = argv[++i];
                check(value == "baseline" || value == "stream-adam", "Unknown R1 profile variant");
                profileVariant = value == "baseline" ? optimizer_gpu::Variant::Baseline
                                                     : optimizer_gpu::Variant::Candidate;
                profileVariantSpecified = true;
            } else if ((arg == "--replay" || arg == "--report" ||
                        arg == "--write-synthetic-fixture") &&
                       i + 1 < argc) {
                if (arg == "--replay")
                    input = argv[++i];
                else if (arg == "--write-synthetic-fixture")
                    syntheticOutput = argv[++i];
                else
                    report = argv[++i];
            } else
                throw std::runtime_error(
                    "Usage: training_optimizer_gpu_tests (--synthetic | --replay "
                    "INPUT.optimizer.bin) [--report NEW.json] [--require-validation] "
                    "[--candidate-sh-color-reuse] "
                    "[--candidate-r1 "
                    "[--profile-r1-seconds 1..30 [--profile-variant baseline|stream-adam]]]");
        }
        check(runSynthetic != !input.empty(), "Choose exactly one of --synthetic and --replay");
        const auto candidate = optimizer_candidate::describe(method);
        const bool hasCandidate = method != Method::None;
        check(!profileSeconds || (method == Method::StreamAdam && !input.empty()),
              "R1 profiling requires --candidate-r1 --replay");
        check(!profileVariantSpecified || profileSeconds,
              "--profile-variant requires a profile window");
        check(syntheticOutput.empty() || runSynthetic,
              "--write-synthetic-fixture requires --synthetic");
        check(syntheticOutput.empty() || !std::filesystem::exists(syntheticOutput),
              "Synthetic fixture output already exists");
        if (report.empty() && !input.empty()) {
            report = input;
            report.replace_extension(candidate.reportExtension);
        }
        check(report.empty() || !std::filesystem::exists(report),
              "Report exists; choose a NEW --report filename");
        if (!input.empty()) input = std::filesystem::absolute(input);
        if (!report.empty()) report = std::filesystem::absolute(report);
        if (!syntheticOutput.empty()) syntheticOutput = std::filesystem::absolute(syntheticOutput);
        std::optional<OptimizerCaptureHeader> capture;
        if (!input.empty()) capture = readOptimizerCaptureHeader(input);
        const auto shader = std::filesystem::path(OPTIMIZER_RUNTIME_DIR) /
                            "shaders/train_backward_2dgs_to_3dgs_optimizer.comp.spv";
        const auto candidateShader =
            hasCandidate ? shader.parent_path() / candidate.shader : std::filesystem::path{};
        candidateHash = hasCandidate ? optimizerShaderHash(candidateShader) : 0;
        VULKAN_HPP_DEFAULT_DISPATCHER.init();
        std::cerr << "S4: Vulkan loader initialized\n";
        for (const auto& layer : vk::enumerateInstanceLayerProperties())
            validation |= std::strcmp(layer.layerName.data(), "VK_LAYER_KHRONOS_validation") == 0;
        const char* layer = "VK_LAYER_KHRONOS_validation";
        std::vector<const char*> extensions;
        vk::ValidationFeatureEnableEXT sync =
            vk::ValidationFeatureEnableEXT::eSynchronizationValidation;
        vk::ValidationFeaturesEXT features;
        vk::ApplicationInfo app{"optimizer-s4", 1, "vulkan-3dgs", 1, VK_API_VERSION_1_2};
        vk::InstanceCreateInfo create;
        create.setPApplicationInfo(&app);
        if (validation) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
            for (const auto& ext : vk::enumerateInstanceExtensionProperties(std::string(layer)))
                syncValidation |= std::strcmp(ext.extensionName.data(),
                                              VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME) == 0;
            if (syncValidation) {
                extensions.push_back(VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME);
                features.setEnabledValidationFeatures(sync);
                create.setPNext(&features);
            }
            create.setPEnabledLayerNames(layer).setPEnabledExtensionNames(extensions);
        }
        check(!requireValidation || (validation && syncValidation),
              "Required core/synchronization validation unavailable");
        auto instance = vk::createInstanceUnique(create);
        VULKAN_HPP_DEFAULT_DISPATCHER.init(*instance);
        std::cerr << "S4: Vulkan instance initialized\n";
        vk::UniqueDebugUtilsMessengerEXT messenger;
        if (validation) {
            vk::DebugUtilsMessengerCreateInfoEXT info;
            info.setMessageSeverity(vk::DebugUtilsMessageSeverityFlagBitsEXT::eError)
                .setMessageType(vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
                                vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance);
            info.pfnUserCallback = onMessage;
            messenger = instance->createDebugUtilsMessengerEXTUnique(info);
        }
        auto devices = instance->enumeratePhysicalDevices();
        check(!devices.empty(), "S4 GPU unavailable");
        auto physical = devices.front();
        bool found = !capture.has_value();
        if (capture) {
            for (auto candidate : devices) {
                OptimizerCaptureHeader identity;
                fillOptimizerCaptureIdentity(identity, candidate, shader);
                if (optimizerCaptureIdentityMatches(*capture, identity)) {
                    physical = candidate;
                    found = true;
                    break;
                }
            }
        }
        check(found, "S4 device/driver/compiler/shader mismatch: target-device bitwise replay "
                     "refused; use --synthetic locally");
        OptimizerCaptureHeader identity;
        fillOptimizerCaptureIdentity(identity, physical, shader);
        std::cerr << "S4: original shader identity recorded on " << identity.deviceName.data()
                  << '\n';
        identityJson = optimizerCaptureIdentityJson(identity);
        details << identityJson;
        const auto families = physical.getQueueFamilyProperties();
        uint32_t family = 0;
        for (; family < families.size(); ++family)
            if (families[family].queueFlags & vk::QueueFlagBits::eCompute) break;
        check(family < families.size(), "S4 no compute queue");
        const float priority = 1;
        vk::DeviceQueueCreateInfo queueInfo{{}, family, 1, &priority};
        vk::DeviceCreateInfo deviceInfo;
        deviceInfo.setQueueCreateInfos(queueInfo);
        auto device = physical.createDeviceUnique(deviceInfo);
        VULKAN_HPP_DEFAULT_DISPATCHER.init(*device);
        std::cerr << "S4: private compute device initialized\n";
        const auto queue = device->getQueue(family, 0);
        if (runSynthetic) {
            captureHook(*device, physical, queue, family, shader, syntheticOutput);
            const auto cases = synthetic(*device, physical, queue, family, shader, candidateShader,
                                         progress, method);
            details << ",\"synthetic_cases\":" << cases << ",\"target_capture_verified\":false";
            if (hasCandidate)
                details << ",\"" << candidate.key << "\":{\"candidate\":\"" << candidate.name
                        << "\",\"bitwise_synthetic\":true,"
                           "\"candidate_shader_fnv1a64\":\""
                        << candidateHash
                        << "\",\"isolated_cases_completed\":" << progress.isolatedCases
                        << ",\"sequence_steps_completed\":" << progress.sequenceSteps
                        << ",\"integrated_into_training\":false,\"short_gain_percent\":null,"
                           "\"verdict\":\"SYNTHETIC_PASSED_TARGET_REPLAY_REQUIRED\"}";
        } else {
            if (hasCandidate) {
                const auto sequenceSteps = candidateSequences(*device, physical, queue, family,
                                                              shader, candidateShader, progress);
                details << ",\"" << candidate.key << "_selftest_sequence_steps\":" << sequenceSteps;
            }
            auto source = [&](size_t block, uint64_t offset, std::span<std::byte> out) {
                readOptimizerCaptureChunk(input, *capture, block, offset, out);
            };
            optimizer_gpu::Session session(*device, physical, queue, family, *capture, source,
                                           shader, candidateShader);
            session.dispatch();
            auto first = session.compare(source);
            optimizer_gpu::requireMatch(
                first, "S4 original GPU replay mismatch against captured post-reference");
            session.dispatch();
            auto repeat = session.compare(source);
            optimizer_gpu::requireMatch(repeat, "S4 original A/A reference mismatch");
            std::vector<optimizer_gpu::Sample> samples;
            if (hasCandidate) {
                details << optimizer_r1::evaluateReplay(session, *capture, source, candidateHash,
                                                        validationErrors, profileSeconds,
                                                        profileVariant, candidate);
            } else {
                session.dispatch();
                session.dispatch();
                for (uint32_t i = 0; i < 8; ++i)
                    samples.push_back(session.dispatch());
                optimizer_gpu::requireMatch(session.compare(source),
                                            "S4 timed invocation state contamination");
            }
            session.verifyReadonly();
            details << ",\"synthetic\":" << (capture->synthetic ? "true" : "false")
                    << ",\"target_capture_verified\":" << (capture->synthetic ? "false" : "true")
                    << ",\"bitwise_post_reference\":true,\"bitwise_"
                       "AA\":true"
                    << ",\"all_mutable_state_restored_each_dispatch\":true"
                    << ",\"gaussians\":" << capture->gaussianCount
                    << ",\"capacity\":" << capture->capacity << ",\"frame\":" << capture->frame
                    << ",\"completed_iteration\":" << capture->completedIteration
                    << ",\"optimizer_step\":" << capture->completedOptimizerStep
                    << ",\"active_SH_degree\":" << capture->push.activeSHDegree << ",\"extent\":["
                    << capture->push.width << ',' << capture->push.height << ']'
                    << ",\"pixel_mode\":" << capture->pixelMode
                    << ",\"ssim_mode\":" << capture->ssimMode
                    << ",\"validation_requested_on_step\":"
                    << (capture->push.validationEnabled ? "true" : "false")
                    << ",\"ordinary_fused\":true,\"ADC\":false,\"opacity_reset\":false"
                    << ",\"parameter_updated_components\":" << first.parameterUpdates
                    << ",\"max_parameter_update\":" << first.maxParameterUpdate
                    << ",\"restore_bytes_per_dispatch\":" << uint64_t(capture->gaussianCount) * 736
                    << ",\"restore_cache_effect\":\"full seed-to-work copy before EACH invocation; "
                       "cache-perturbed isolated baseline, not ordinary-training wall time\""
                    << ",\"samples\":[";
            for (size_t i = 0; i < samples.size(); ++i) {
                if (i) details << ',';
                details << "{\"kernel_ms\":";
                if (session.timestampsAvailable())
                    details << samples[i].kernelMs;
                else
                    details << "null";
                details << ",\"restore_ms\":";
                if (session.timestampsAvailable())
                    details << samples[i].restoreMs;
                else
                    details << "null";
                details << ",\"cpu_submit_wait_ms\":" << samples[i].cpuSubmitWaitMs
                        << ",\"restore_cpu_submit_wait_ms\":" << samples[i].restoreCpuMs << '}';
            }
            details << ']';
            if (!hasCandidate) details << ",\"warmup\":2,\"measured\":8";
        }
        device->waitIdle();
        check(!validationErrors, "S4 core/synchronization validation errors");
        const std::string result =
            "{" + details.str() +
            (hasCandidate ? std::string(",\"status\":\"") +
                                (method == Method::StreamAdam
                                     ? "R1_CHECKS_COMPLETED_SEE_R1_VERDICT"
                                     : "CANDIDATE_CHECKS_COMPLETED_SEE_CANDIDATE_VERDICT") +
                                "\",\"recommendation\":"
                                "\"DO_NOT_INTEGRATE_WITHOUT_TARGET_TIMING_AND_NATIVE_REVIEW\","
                                "\"speedup\":null,"
                          : ",\"status\":\"BASELINE_CHECKS_PASSED_NATIVE_EVIDENCE_REQUIRED\","
                            "\"recommendation\":"
                            "\"INSUFFICIENT_EVIDENCE\",\"speedup\":null,") +
            "\"native\":{\"registers\":null,\"shared_bytes\":null,\"resident_warps\":null,\"spill_"
            "bytes\":null,\"local_load_store\":null,\"L1\":null,\"L2\":null,\"DRAM\":null,\"SM\":"
            "null},"
            "\"validation_enabled\":" +
            (validation ? "true" : "false") +
            ",\"synchronization_validation_enabled\":" + (syncValidation ? "true" : "false") +
            ",\"validation_errors\":" + std::to_string(validationErrors.load()) + "}\n";
        if (!report.empty()) reportFile(report, result);
        std::cout << result;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "S4 optimizer diagnostics failed: " << error.what() << '\n';
        if (!report.empty() && !std::filesystem::exists(report)) {
            try {
                std::ostringstream failure;
                failure << '{';
                if (!identityJson.empty()) failure << identityJson << ',';
                failure << "\"status\":\"STOP\",\"recommendation\":\"DO_NOT_INTEGRATE\","
                           "\"speedup\":null"
                        << ",\"validation_enabled\":" << (validation ? "true" : "false")
                        << ",\"synchronization_validation_enabled\":"
                        << (syncValidation ? "true" : "false")
                        << ",\"validation_errors\":" << validationErrors.load();
                if (method != Method::None) {
                    const auto candidate = optimizer_candidate::describe(method);
                    failure << ",\"" << candidate.key << "\":{\"candidate\":\"" << candidate.name
                            << "\",\"candidate_shader_"
                               "fnv1a64\":\""
                            << candidateHash
                            << "\",\"isolated_cases_completed\":" << progress.isolatedCases
                            << ",\"sequence_steps_completed\":" << progress.sequenceSteps
                            << ",\"integrated_into_training\":false,\"verdict\":\"STOP\",\"short_"
                               "gain_percent\":null}";
                }
                failure << ",\"error\":" << quoted(error.what()) << "}\n";
                reportFile(report, failure.str());
            } catch (...) {}
        }
        return 1;
    }
}
