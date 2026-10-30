#pragma once

#include "training/core/backward/backward_loss_pass.hpp"
#include "training/core/training_resident_images.hpp"
#include "training_morton_checks.hpp"

#include <fstream>

namespace vksplat_checks {
using namespace vulkan3DGS;
using morton_checks::require;

inline double objective(const std::vector<glm::vec4>& image, const std::vector<uint32_t>& target,
                        uint32_t w, uint32_t h, double lambda = 0.2) {
    constexpr double weights[] = {0.001028380123898387, 0.0075987582094967365, 0.036000773310661316,
                                  0.10936068743467331,  0.21300552785396576,   0.26601171493530273};
    double loss = 0;
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x) {
            const uint32_t i = y * w + x;
            const double alpha = double(target[i] >> 24) / 255.0;
            for (int c = 0; c < 3; ++c)
                loss += (1.0 - lambda) *
                        std::abs(std::max(double(image[i][c]), 0.0) * alpha -
                                 double((target[i] >> (8 * c)) & 255u) / 255.0 * alpha) /
                        (3.0 * w * h);
            if (x < 5 || y < 5 || x >= w - 5 || y >= h - 5) continue;
            for (int c = 0; c < 3; ++c) {
                double a = 0, b = 0, aa = 0, bb = 0, ab = 0;
                for (int dy = -5; dy <= 5; ++dy)
                    for (int dx = -5; dx <= 5; ++dx) {
                        const auto j = size_t(int(y) + dy) * w + size_t(int(x) + dx);
                        const double mask = double(target[j] >> 24) / 255.0;
                        const double v = std::max(double(image[j][c]), 0.0) * mask;
                        const double t = double((target[j] >> (8 * c)) & 255u) / 255.0 * mask;
                        const double k = weights[5 - std::abs(dx)] * weights[5 - std::abs(dy)];
                        a += k * v;
                        b += k * t;
                        aa += k * v * v;
                        bb += k * t * t;
                        ab += k * v * t;
                    }
                const double ssim = (2 * a * b + 0.0001) * (2 * (ab - a * b) + 0.0009) /
                                    ((a * a + b * b + 0.0001) * (aa - a * a + bb - b * b + 0.0009));
                loss += lambda * (1 - ssim) / (3.0 * (w - 10) * (h - 10));
            }
        }
    return loss;
}

inline void close(double actual, double expected, const char* message) {
    require(std::isfinite(actual) && std::isfinite(expected) &&
                std::abs(actual - expected) <=
                    1e-5 + 1e-3 * std::max(std::abs(actual), std::abs(expected)),
            message);
}

inline std::string run(vk::Device device, vk::PhysicalDevice physical, vk::Queue queue,
                       uint32_t family) {
    OptimizerCaptureTransfer transfer(device, physical, queue, family);
    CommandPool pool;
    pool.create(device, family, vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
    const auto command = pool.allocateCommandBuffer();
    auto fence = device.createFenceUnique({});
    uint32_t cases = 0, fallbackCases = 0;
    for (auto size : {glm::uvec2(11, 11), glm::uvec2(17, 19), glm::uvec2(32, 17), glm::uvec2(2, 2),
                      glm::uvec2(10, 17)}) {
        for (uint32_t alpha : {255u, 128u, 0u, 64u}) {
            for (float lambda : {0.0f, 0.2f, 1.0f}) {
                const auto w = size.x, h = size.y, n = w * h;
                TrainingBuffers buffers;
                buffers.initialize(device, physical, queue, family, family);
                buffers.resize(1, {w, h});
                std::vector<glm::vec4> rendered(n);
                std::vector<uint32_t> target(n);
                std::vector<PixelBlendState> blends(n);
                for (uint32_t i = 0; i < n; ++i) {
                    rendered[i] = {0.23f + float(i % 9) * 0.013f, 0.42f, 0.31f, 1.0f};
                    target[i] = ((alpha == 64 ? (i % 3 == 0   ? 0u
                                                 : i % 3 == 1 ? 128u
                                                              : 255u)
                                              : alpha)
                                 << 24) |
                                0x008469ab;
                }
                morton_checks::upload(transfer, buffers.renderedColorInfo(), rendered);
                morton_checks::upload(transfer, buffers.targetColorInfo(), target);
                morton_checks::upload(transfer, buffers.pixelBlendStatesInfo(), blends);
                BackwardLossPass pass;
                pass.initialize(device);
                pass.configureVkSplat(physical, queue, family);
                pass.setVkSplat(true);
                TrainingPushConstants push{};
                push.width = w;
                push.height = h;
                push.pixelCount = n;
                push.gaussianCount = 1;
                push.lossDssimWeight = lambda;
                push.validationEnabled = 1;
                command.reset();
                command.begin(
                    vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
                const BackwardPassContext context{buffers, command, push};
                pass.recordLoss(context);
                pass.recordLossToPixel(context);
                command.end();
                queue.submit(vk::SubmitInfo{}.setCommandBuffers(command), *fence);
                (void)device.waitForFences(*fence, true, UINT64_MAX);
                device.resetFences(*fence);
                if (w <= 10 || h <= 10) {
                    require(!pass.vkSplatActive(), "Small-image SSIM did not report fallback");
                    ++fallbackCases;
                    continue;
                }
                require(pass.vkSplatActive(), "VkSplat Loss unexpectedly fell back");
                const auto loss = morton_checks::read<float>(transfer, buffers.lossInfo());
                close(std::accumulate(loss.begin(), loss.end(), 0.0),
                      objective(rendered, target, w, h, lambda),
                      "VkSplat objective differs from independent FP64 convolution");
                const auto gradients =
                    morton_checks::read<PixelGrad>(transfer, buffers.pixelGradsInfo());
                for (uint32_t i : {0u, n / 2u, n - 1u})
                    for (int c = 0; c < 3; ++c) {
                        const float before = rendered[i][c];
                        rendered[i][c] = before + 0.0001f;
                        const double plus = objective(rendered, target, w, h, lambda);
                        const float high = rendered[i][c];
                        rendered[i][c] = before - 0.0001f;
                        const double minus = objective(rendered, target, w, h, lambda);
                        const double delta = double(high) - rendered[i][c];
                        rendered[i][c] = before;
                        const uint32_t pixelAlpha = target[i] >> 24;
                        const double derivative =
                            pixelAlpha ? (plus - minus) / delta * (255.0 / pixelAlpha) : 0.0;
                        close(gradients[i].color[c], derivative,
                              "VkSplat source-gradient contract mismatch");
                    }
                ++cases;
            }
        }
    }
    const auto directory =
        std::filesystem::temp_directory_path() /
        ("vulkan3dgs-resident-" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(directory);
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() {
            std::error_code e;
            for (int i = 0; i < 3; ++i)
                std::filesystem::remove(path / (std::to_string(i) + ".ppm"), e);
            std::filesystem::remove(path, e);
        }
    } cleanup{directory};
    TrainingDataset dataset;
    for (uint32_t i = 0; i < 3; ++i) {
        TrainingCameraFrame frame{};
        frame.width = 2;
        frame.height = 2;
        frame.fx = frame.fy = 2;
        frame.cx = frame.cy = 1;
        frame.imagePath = directory / (std::to_string(i) + ".ppm");
        std::ofstream image(frame.imagePath, std::ios::binary);
        image << "P6\n2 2\n255\n";
        for (int p = 0; p < 12; ++p)
            image.put(static_cast<char>(i * 19 + uint32_t(p)));
        image.close();
        if (i == 2)
            dataset.validationFrames.push_back(frame);
        else
            dataset.frames.push_back(frame);
    }
    auto resident =
        TrainingResidentImages::prepare(device, physical, queue, family, family, dataset);
    for (size_t i = 0; i < 2; ++i) {
        const auto pixels = morton_checks::read<uint32_t>(transfer, resident->trainingImage(i));
        require(pixels.size() == 4 &&
                    pixels[0] == (0xff000000u | uint32_t(i * 19) | (uint32_t(i * 19 + 1) << 8) |
                                  (uint32_t(i * 19 + 2) << 16)),
                "GPU Resident frame binding mismatch");
    }
    require(resident->stats().residentImages == 3 && resident->stats().uploads == 3 &&
                resident->stats().hits == 2,
            "GPU Resident validation images/counters mismatch");
    auto invalid = dataset;
    invalid.validationFrames[0].imagePath = directory / "missing.ppm";
    bool rejected = false;
    try {
        (void)TrainingResidentImages::prepare(device, physical, queue, family, family, invalid);
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected && resident->stats().residentImages == 3,
            "Failed preload changed current residency");
    for (uint32_t i = 0; i < 8; ++i)
        dataset.sparsePoints.push_back(
            {{float(i % 2) * 0.1f, float((i / 2) % 2) * 0.1f, 2.0f + float(i / 4) * 0.1f},
             {0.3f, 0.4f, 0.2f},
             0,
             0});
    GaussianTraining training;
    training.initialize(device, physical, queue, family, family);
    training.loadDataset(dataset, std::move(resident));
    training.initializeModelFromDataset();
    training.setFasterSortEnabled(true);
    training.setPixelTo2DGSMode(TrainingPixelTo2DGSMode::Direct);
    training.initializeTrainingRenderers(device, physical, queue, family, training.gaussianCount(),
                                         {2, 2});
    TrainingScheduleConfig schedule;
    schedule.totalIterations = 2;
    training.setScheduleConfig(schedule);
    training.trainStep();
    training.trainStep();
    require(training.imageCacheStats().hostHits == 0 &&
                training.imageCacheStats().hostMisses == 0 &&
                training.deviceImageCacheStats().hits == 4 &&
                training.deviceImageCacheStats().uploads == 3,
            "GPU Resident training requested host pixels or uploaded images again");
    return "{\"source_commit\":\"b3ad2b048918496616e0e9346e35c606d813ab55\","
           "\"loss_cases\":" +
           std::to_string(cases) + ",\"loss_fallback_cases\":" + std::to_string(fallbackCases) +
           ",\"soft_alpha_source_gradient_checked\":true,\"resident_images\":3,"
           "\"resident_synthetic_training_steps\":2,\"real_training\":false}";
}
} // namespace vksplat_checks
