#include "training/core/training_optimizer_capture.hpp"
#include "training_optimizer_r1_metrics.hpp"
#include "training_optimizer_reference.hpp"

#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace vulkan3DGS;
namespace {
void check(bool okay, const char* message) {
    if (!okay) throw std::runtime_error(message);
}
template <class F> void reject(F&& f, const char* message) {
    bool rejected = false;
    try {
        f();
    } catch (const std::exception&) {
        rejected = true;
    }
    check(rejected, message);
}
OptimizerCaptureHeader header(uint32_t count) {
    OptimizerCaptureHeader h{};
    h.gaussianCount = h.capacity = h.push.gaussianCount = count;
    h.push.width = h.push.height = h.push.pixelCount = 1;
    h.completedIteration = h.completedOptimizerStep = 1;
    h.shaderHash = 123;
    h.synthetic = 1;
    return h;
}
void seal(OptimizerCaptureHeader& h) {
    h.headerChecksum = 0;
    h.headerChecksum = optimizerHash(std::as_bytes(std::span{&h, 1}));
}
void protocol() {
    const auto dir = std::filesystem::temp_directory_path() /
                     ("v3dgs-optimizer-test-" +
                      std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(dir);
    const auto path = dir / "fixture.optimizer.bin";
    auto source = [](size_t block, uint64_t offset, std::span<std::byte> data) {
        for (size_t i = 0; i < data.size(); ++i)
            data[i] = std::byte((offset + i) % 4 == 0 ? ((offset + i) / 4 + block) % 251 : 0);
    };
    try {
        for (uint32_t count : {0u, 1u, 257u, 5000u}) {
            const auto file = dir / (std::to_string(count) + ".bin");
            writeOptimizerCapture(file, header(count), source);
            const auto h = readOptimizerCaptureHeader(file);
            check(h.gaussianCount == count, "Capture roundtrip count");
            reject([&] { writeOptimizerCapture(file, header(count), source); },
                   "Capture overwrite accepted");
            check(std::filesystem::file_size(file) == optimizerCaptureBytes(count),
                  "Capture overwrite changed size");
        }
        writeOptimizerCapture(path, header(1), source);
        auto h = readOptimizerCaptureHeader(path);
        auto malformed = h;
        malformed.version = 2;
        seal(malformed);
        reject([&] { validateOptimizerCaptureHeader(malformed, optimizerCaptureBytes(1)); },
               "Unknown version accepted");
        malformed = h;
        malformed.strides[1] = 240;
        seal(malformed);
        reject([&] { validateOptimizerCaptureHeader(malformed, optimizerCaptureBytes(1)); },
               "Wrong stride accepted");
        malformed = h;
        malformed.gaussianCount = UINT32_MAX;
        seal(malformed);
        reject(
            [&] { validateOptimizerCaptureHeader(malformed, optimizerCaptureBytes(UINT32_MAX)); },
            "Huge capture accepted");
        malformed = h;
        malformed.push.optimizerStep++;
        seal(malformed);
        reject([&] { validateOptimizerCaptureHeader(malformed, optimizerCaptureBytes(1)); },
               "Mixed pre/post step accepted");
        malformed = h;
        malformed.blocks[0].bytes = UINT64_MAX;
        seal(malformed);
        reject([&] { validateOptimizerCaptureHeader(malformed, optimizerCaptureBytes(1)); },
               "Overflow block accepted");
        reject([&] { validateOptimizerCaptureHeader(h, optimizerCaptureBytes(1), 64); },
               "File budget not checked");
        reject([&] { validateOptimizerCaptureHeader(h, optimizerCaptureBytes(1) + 1); },
               "Trailing byte accepted");
        reject([&] { validateOptimizerCaptureHeader(h, optimizerCaptureBytes(1) - 1); },
               "Truncation accepted");
        malformed = h;
        malformed.shaderHash++;
        check(!optimizerCaptureIdentityMatches(h, malformed), "Shader mismatch accepted");
        malformed = h;
        malformed.driverVersion++;
        check(!optimizerCaptureIdentityMatches(h, malformed), "Driver mismatch accepted");
        malformed = h;
        malformed.deviceUUID[0]++;
        check(!optimizerCaptureIdentityMatches(h, malformed), "Device mismatch accepted");
        malformed = h;
        malformed.compiler[0]++;
        check(!optimizerCaptureIdentityMatches(h, malformed), "Compiler mismatch accepted");
        std::fstream corrupt(path, std::ios::binary | std::ios::in | std::ios::out);
        corrupt.seekp(sizeof(h) + 1);
        corrupt.put('\xff');
        corrupt.close();
        reject([&] { (void)readOptimizerCaptureHeader(path); }, "Payload corruption accepted");
        // Failure leaves an invalid partial file, never a seemingly successful export.
        reject(
            [&] {
                writeOptimizerCapture(dir / "partial.bin", header(1),
                                      [](size_t, uint64_t, std::span<std::byte>) {
                                          throw std::runtime_error("injected IO failure");
                                      });
            },
            "Source failure swallowed");
        reject([&] { (void)readOptimizerCaptureHeader(dir / "partial.bin"); },
               "Partial export accepted");
        std::filesystem::resize_file(path, 8);
        reject([&] { (void)readOptimizerCaptureHeader(path); }, "Short header accepted");
    } catch (...) {
        std::filesystem::remove_all(dir);
        throw;
    }
    std::filesystem::remove_all(dir);
}
void oracle() {
    TrainingPushConstants p;
    GaussianTrainParam g{};
    AdamState a{};
    GaussianGrad gradient{};
    for (uint32_t degree = 0; degree <= 3; ++degree) {
        p.activeSHDegree = degree;
        g = {};
        a = {};
        a.firstMoment.positionOpacity = glm::vec4(0.25f);
        a.secondMoment.positionOpacity = glm::vec4(0.5f);
        for (auto& pack : a.firstMoment.sh)
            pack = glm::vec4(0.125f);
        for (auto& pack : a.secondMoment.sh)
            pack = glm::vec4(0.25f);
        const auto before = a;
        optimizer_reference::adam(g, a, gradient, p);
        check(g.positionOpacity.x != 0 &&
                  a.firstMoment.positionOpacity.x < before.firstMoment.positionOpacity.x,
              "Dense Adam no-support/zero-gradient oracle broken");
        const uint32_t active = (degree + 1) * (degree + 1);
        for (uint32_t coefficient = active; coefficient < 16; ++coefficient)
            check(a.firstMoment.sh[coefficient / 4][coefficient % 4] == 0.125f &&
                      a.secondMoment.sh[coefficient / 4][coefficient % 4] == 0.25f &&
                      g.sh[coefficient / 4][coefficient % 4] == 0,
                  "Inactive SH moments/params must freeze");
    }
    float x = 1, m = 0, v = 0;
    p.optimizerBeta1 = p.optimizerBeta2 = 0;
    p.optimizerOneMinusBeta1 = p.optimizerOneMinusBeta2 = 1;
    p.optimizerGradClip = 0.01f;
    optimizer_reference::component(x, m, v, -2, 0, true, p);
    check(x == 1 && m == -0.01f && std::abs(v - 0.0001f) < 1e-10f, "Clip/LR0 oracle broken");
    const float old = x;
    optimizer_reference::component(x, m, v, 3, 1, false, p);
    check(x == old && m == -0.01f, "Inactive differs from zero gradient");
}
void r1Contracts() {
    // Original AoS ABI stays unchanged; every vec4 field retains its byte offset.
    static_assert(sizeof(AdamState) == 30 * sizeof(glm::vec4));
    static_assert(offsetof(AdamState, secondMoment) == 15 * sizeof(glm::vec4));
    std::array<AdamState, 3> states{};
    for (size_t gaussian = 0; gaussian < states.size(); ++gaussian)
        for (size_t moment = 0; moment < 2; ++moment)
            for (size_t pack = 0; pack < 15; ++pack) {
                const glm::vec4 value(float(100 * gaussian + 20 * moment + pack));
                std::memcpy(reinterpret_cast<std::byte*>(&states[gaussian]) +
                                (moment * 15 + pack) * 16,
                            &value, 16);
            }
    for (size_t gaussian = 0; gaussian < states.size(); ++gaussian)
        for (size_t pack = 0; pack < 15; ++pack) {
            glm::vec4 m, v;
            const auto* raw = reinterpret_cast<const std::byte*>(states.data());
            std::memcpy(&m, raw + (gaussian * 30 + pack) * 16, 16);
            std::memcpy(&v, raw + (gaussian * 30 + pack + 15) * 16, 16);
            check(m.x == float(100 * gaussian + pack) && v.x == float(100 * gaussian + 20 + pack),
                  "S4-1R AoS float4 view offset contract");
        }
    std::array<double, 8> a{}, b{}, repeat{};
    a.fill(2.8);
    b.fill(2.5);
    repeat.fill(2.8);
    check(optimizer_r1::evaluate(a, b, repeat).verdict == "SHORT_GAIN_REQUIRES_NATIVE_REVIEW",
          "R1 valid short-screen gate");
    b.fill(2.79);
    check(optimizer_r1::evaluate(a, b, repeat).verdict == "STOP_NO_SHORT_GAIN",
          "R1 tiny gain incorrectly admitted");
    b.fill(2.5);
    repeat.fill(2.9);
    check(optimizer_r1::evaluate(a, b, repeat).verdict == "INCONCLUSIVE_BASELINE_DRIFT",
          "R1 baseline drift not rejected");
    repeat.fill(2.8);
    a[0] = 34.815;
    check(optimizer_r1::evaluate(a, b, repeat).verdict == "INCONCLUSIVE_SAMPLE_OUTLIER",
          "R1 first sample silently discarded");
    a.fill(2.8);
    b[0] = 20;
    check(optimizer_r1::evaluate(a, b, repeat).verdict == "INCONCLUSIVE_SAMPLE_OUTLIER",
          "R1 candidate outlier not rejected");
    b.fill(0);
    reject([&] { (void)optimizer_r1::evaluate(a, b, repeat); }, "R1 zero timestamp accepted");
    b.fill(std::numeric_limits<double>::quiet_NaN());
    reject([&] { (void)optimizer_r1::evaluate(a, b, repeat); }, "R1 NaN timestamp accepted");
}
} // namespace
int main() {
    try {
        protocol();
        oracle();
        r1Contracts();
        std::cout << "Optimizer capture protocol and CPU Adam checks passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
