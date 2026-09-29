#include "training/ui/training_diagnostics_panel.hpp"

#include <imgui.h>

#include <algorithm>

namespace vulkan3DGS {

void TrainingDiagnosticsPanel::draw(const TrainingUiSnapshot& snapshot,
                                    uint32_t frameCount) {
    const auto& densification = snapshot.densification;
    if (snapshot.densificationIteration > 0) {
        ImGui::Text("Densify/prune iter %u, output %u",
                    snapshot.densificationIteration,
                    densification.outputCount);
        ImGui::Text("Densify kept %u, cloned %u, split %u, pruned %u",
                    densification.keptSources,
                    densification.cloneSources,
                    densification.splitSources,
                    densification.prunedSources);
        ImGui::Text("Prune hits opacity %u, screen %u, world %u",
                    densification.pruneOpacityHits,
                    densification.pruneScreenHits,
                    densification.pruneWorldHits);
    } else {
        ImGui::Text("Densify/prune not run yet");
    }

    const auto& validation = snapshot.validation;
    ImGui::Text("Loss %.6g", validation.meanLoss);
    ImGui::Text("Render alpha %.6g", validation.meanRenderedAlpha);
    ImGui::Text("Tile items %u", validation.tileItemCount);
    ImGui::Text("Backward candidates avg %.2f, contributors avg %.2f, max candidates %u",
                validation.meanProcessedCandidatesPerPixel,
                validation.meanContributorsPerPixel,
                validation.maxProcessedCandidatesPerPixel);

    const auto& candidateProfile = snapshot.candidateProfile;
    if (candidateProfile.sampleCount > 0u) {
        const float emptyCandidatePercent = candidateProfile.meanProcessedCandidatesPerPixel > 0.0f
            ? 100.0f * (1.0f - candidateProfile.meanContributorsPerPixel /
                                  candidateProfile.meanProcessedCandidatesPerPixel)
            : 0.0f;
        ImGui::Text("Validation history n=%u: candidates %.2f, contributors %.2f, empty %.2f%%, max %u",
                    candidateProfile.sampleCount,
                    candidateProfile.meanProcessedCandidatesPerPixel,
                    candidateProfile.meanContributorsPerPixel,
                    emptyCandidatePercent,
                    candidateProfile.maxProcessedCandidatesPerPixel);
        ImGui::Text("Processed pixels 0/1-32/33-64/65-128: %.1f%% / %.1f%% / %.1f%% / %.1f%%",
                    100.0f * candidateProfile.meanPixelFractionByProcessedBucket[0],
                    100.0f * candidateProfile.meanPixelFractionByProcessedBucket[1],
                    100.0f * candidateProfile.meanPixelFractionByProcessedBucket[2],
                    100.0f * candidateProfile.meanPixelFractionByProcessedBucket[3]);
        ImGui::Text("Processed pixels 129-256/257-512/513-1024/>1024: %.1f%% / %.1f%% / %.1f%% / %.1f%%",
                    100.0f * candidateProfile.meanPixelFractionByProcessedBucket[4],
                    100.0f * candidateProfile.meanPixelFractionByProcessedBucket[5],
                    100.0f * candidateProfile.meanPixelFractionByProcessedBucket[6],
                    100.0f * candidateProfile.meanPixelFractionByProcessedBucket[7]);
    }

    if (!validation.valid && snapshot.trainingIteration > 0u) {
        ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.25f, 1.0f),
                           "Validation issue: invalid loss %u, invalid pixels %u, non-finite gaussians %u",
                           validation.invalidLossCount,
                           validation.invalidRenderedPixelCount,
                           validation.nonFiniteGaussianCount);
        if (validation.nonFiniteGaussianCount > 0u) {
            ImGui::Text("First non-finite %u; pos %u, opacity %u, raw/active scale %u/%u, rotation %u, SH %u",
                        validation.firstNonFiniteGaussianIndex,
                        validation.nonFinitePositionCount,
                        validation.nonFiniteOpacityCount,
                        validation.nonFiniteRawScaleCount,
                        validation.nonFiniteActivatedScaleCount,
                        validation.nonFiniteRotationCount,
                        validation.nonFiniteSHCount);
        }
    }

    if (ImGui::CollapsingHeader("Training Profiling", ImGuiTreeNodeFlags_DefaultOpen)) {
        static constexpr const char* cpuStageNames[] = {
            "Frame upload", "Image request/wait", "Target RGBA8 upload",
            "Prepare submit/wait", "Tile count readback", "Tile buffer resize",
            "Main command record", "Main submit/wait", "Validation",
            "Densify adopt", "Total step",
        };
        static constexpr const char* gpuStageNames[] = {
            "Gaussian projection", "Tile coverage count", "Tile prefix", "Tile emit",
            "Tile sort/ranges", "Composite", "Loss", "Backward clear", "Loss to pixel",
            "Pixel to 2DGS", "Tile-local backward", "2DGS to 3DGS",
            "Fused projection/optimizer", "Optimizer", "Validation", "Densify/prune",
        };
        static_assert(sizeof(cpuStageNames) / sizeof(cpuStageNames[0]) ==
                      kTrainingCpuProfileStageCount);
        static_assert(sizeof(gpuStageNames) / sizeof(gpuStageNames[0]) ==
                      kTrainingGpuProfileStageCount);
        const auto& profiling = snapshot.profiling;
        ImGui::Text("CPU last / average (ms)");
        for (size_t i = 0; i < kTrainingCpuProfileStageCount; ++i) {
            const auto& timing = profiling.cpu[i];
            ImGui::Text("%s %.2f / %.2f (n=%u)",
                        cpuStageNames[i], timing.lastMs, timing.averageMs, timing.sampleCount);
        }
        if (profiling.gpuTimestampsAvailable) {
            ImGui::Text("GPU last / average (ms)");
            for (size_t i = 0; i < kTrainingGpuProfileStageCount; ++i) {
                const auto& timing = profiling.gpu[i];
                ImGui::Text("%s %.2f / %.2f (n=%u)",
                            gpuStageNames[i], timing.lastMs, timing.averageMs, timing.sampleCount);
            }
        } else {
            ImGui::TextDisabled("GPU timestamps unavailable on the compute queue");
        }
    }

    if (ImGui::CollapsingHeader("Image Cache", ImGuiTreeNodeFlags_DefaultOpen)) {
        const ImageStreamerStats cacheStats = snapshot.imageCache;
        constexpr double bytesPerMiB = 1024.0 * 1024.0;
        ImGui::Text("Host %.1f / %.1f MiB, %llu images",
                    static_cast<double>(cacheStats.hostCachedBytes) / bytesPerMiB,
                    static_cast<double>(cacheStats.hostBudgetBytes) / bytesPerMiB,
                    static_cast<unsigned long long>(cacheStats.hostCachedImages));
        ImGui::Text("Host hit %llu, miss %llu, wait %llu, evict %llu",
                    static_cast<unsigned long long>(cacheStats.hostHits),
                    static_cast<unsigned long long>(cacheStats.hostMisses),
                    static_cast<unsigned long long>(cacheStats.hostWaits),
                    static_cast<unsigned long long>(cacheStats.hostEvictions));
        ImGui::Text("Prefetch %llu, KTX chunk hit %llu, miss %llu, write %llu",
                    static_cast<unsigned long long>(cacheStats.prefetchRequests),
                    static_cast<unsigned long long>(cacheStats.disk.hits),
                    static_cast<unsigned long long>(cacheStats.disk.misses),
                    static_cast<unsigned long long>(cacheStats.disk.writes));
        ImGui::Text("KTX active %.1f MiB / %llu chunks, history %.1f MiB / %llu chunks",
                    static_cast<double>(cacheStats.disk.activeBytes) / bytesPerMiB,
                    static_cast<unsigned long long>(cacheStats.disk.activeFiles),
                    static_cast<double>(cacheStats.disk.historicalBytes) / bytesPerMiB,
                    static_cast<unsigned long long>(cacheStats.disk.historicalFiles));
        ImGui::Text("KTX total %.1f MiB / %llu chunks, recover %llu, evict %llu",
                    static_cast<double>(cacheStats.disk.cachedBytes) / bytesPerMiB,
                    static_cast<unsigned long long>(cacheStats.disk.cachedFiles),
                    static_cast<unsigned long long>(cacheStats.disk.recoveries),
                    static_cast<unsigned long long>(cacheStats.disk.evictions));

        const DeviceImageCacheStats deviceCacheStats = snapshot.deviceImageCache;
        static constexpr const char* deviceCacheModes[] = {"Streaming", "Partial", "Full"};
        const uint32_t deviceMode = std::min(static_cast<uint32_t>(deviceCacheStats.mode), 2u);
        ImGui::Text("GPU %s %.1f / %.1f MiB, %u resident / %u slots / %u total",
                    deviceCacheModes[deviceMode],
                    static_cast<double>(deviceCacheStats.allocatedBytes) / bytesPerMiB,
                    static_cast<double>(deviceCacheStats.budgetBytes) / bytesPerMiB,
                    deviceCacheStats.residentImages,
                    deviceCacheStats.slotCount,
                    deviceCacheStats.totalImages);
        ImGui::Text("GPU hit %llu, miss %llu, upload %llu, evict %llu",
                    static_cast<unsigned long long>(deviceCacheStats.hits),
                    static_cast<unsigned long long>(deviceCacheStats.misses),
                    static_cast<unsigned long long>(deviceCacheStats.uploads),
                    static_cast<unsigned long long>(deviceCacheStats.evictions));
        ImGui::Text("Upload %s, staging %.1f MiB, pending %u, ring waits %llu",
                    deviceCacheStats.asynchronousUploads ? "async" : "sync",
                    static_cast<double>(deviceCacheStats.stagingBytes) / bytesPerMiB,
                    deviceCacheStats.pendingUploads,
                    static_cast<unsigned long long>(deviceCacheStats.uploadWaits));
        if (deviceCacheStats.heapBudgetBytes > 0) {
            ImGui::Text("VRAM heap %.1f / %.1f MiB%s, cache resizes %llu",
                        static_cast<double>(deviceCacheStats.heapUsageBytes) / bytesPerMiB,
                        static_cast<double>(deviceCacheStats.heapBudgetBytes) / bytesPerMiB,
                        deviceCacheStats.memoryBudgetAvailable ? " (budget)" : " (size only)",
                        static_cast<unsigned long long>(deviceCacheStats.budgetResizes));
        }
    }

    ImGui::Text("Frame %llu / %u",
                static_cast<unsigned long long>(snapshot.currentFrameIndex),
                frameCount);
}

} // namespace vulkan3DGS
