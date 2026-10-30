#include "training/ui/training_command_panel.hpp"

#include "training/ui/training_diagnostics_panel.hpp"

#include <imgui.h>

#include <algorithm>

namespace vulkan3DGS {

TrainingCommandPanelActions TrainingCommandPanel::draw(TrainingSettings& settings,
                                                       const TrainingCommandPanelState& state,
                                                       const TrainingUiSnapshot& snapshot) {
    TrainingCommandPanelActions actions{};

    if (state.trainingRunning) {
        ImGui::BeginDisabled();
    }
    if (ImGui::InputText("Dataset Folder", settings.paths.dataset.data(),
                         settings.paths.dataset.size())) {
        actions.datasetPathChanged = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Browse##DatasetFolder")) {
        actions.browseDataset = true;
    }
    const int previousDownscale = settings.initialization.downscale;
    ImGui::InputInt("Downscale", &settings.initialization.downscale);
    settings.initialization.downscale = std::clamp(settings.initialization.downscale, 1, 16);
    actions.downscaleChanged = settings.initialization.downscale != previousDownscale;

    TrainingSettingsPanel::drawInitialization(settings, state.trainingRunning);
    if (ImGui::Button("Validate")) {
        actions.validateDataset = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Load Dataset")) {
        actions.loadDataset = true;
    }
    if (state.trainingRunning) {
        ImGui::EndDisabled();
    }

    if (state.datasetValid) {
        ImGui::Text("Frames %u", state.frameCount);
        ImGui::Text("Resolution %ux%u", state.imageWidth, state.imageHeight);
    }

    ImGui::Separator();
    actions.settings = TrainingSettingsPanel::drawTrainingAndDensification(settings, snapshot,
                                                                           state.trainingRunning);

    const bool hasDatasetPath = !textBufferString(settings.paths.dataset).empty();
    const bool deviceSettingsPending =
        state.resourceSettingsPending ||
        (settings.run.physicalTileKey16 && settings.run.computeBackend == 0) !=
            state.storage16Requested;
    if (deviceSettingsPending)
        ImGui::TextWrapped(
            "Training resource settings pending: click Load Dataset to apply it to a new model.");
    ImGui::TextDisabled("Device 16-bit request: %s; enabled: %s",
                        state.storage16Requested ? "on" : "off",
                        state.storage16Enabled ? "on" : "off");
    if (!hasDatasetPath || (deviceSettingsPending && !state.trainingRunning)) {
        ImGui::BeginDisabled();
    }
    const bool benchmarkActive = snapshot.fixedBenchmark.active;
    const char* primaryLabel = state.stopRequested ? "Stopping Training..."
                               : benchmarkActive
                                   ? "Stop Benchmark"
                                   : (state.trainingRunning ? "Pause Training" : "Start Training");
    if (state.stopRequested) {
        ImGui::BeginDisabled();
    }
    if (ImGui::Button(primaryLabel)) {
        actions.toggleTraining = true;
    }
    if (state.stopRequested) {
        ImGui::EndDisabled();
    }
    if (!state.trainingRunning) {
        ImGui::SameLine();
        if (ImGui::Button("Start Fixed Benchmark")) {
            actions.startFixedBenchmark = true;
        }
        if (ImGui::Button("Run Checkpoint State Diagnostic")) {
            actions.startCheckpointDiagnostic = true;
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Benchmarks checkpoint state generation; gradients use the original Tensor path.");
        if (ImGui::Button("Run Checkpoint Warp Benchmark")) actions.startCheckpointWarp = true;
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Tests checkpointed warp gradients on this frozen model/frame.");
        if (ImGui::Button("Run S3 EarlyCull Benchmark")) actions.startTensorEarlyCull = true;
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Experimental backward-only exp culling. Requires Tensor Subgroup/Global Baseline; forward and normal training are unchanged.");
    }
    if (!hasDatasetPath || (deviceSettingsPending && !state.trainingRunning)) {
        ImGui::EndDisabled();
    }

    if (!state.trainingRunning) {
        ImGui::BeginDisabled(!snapshot.rendererInitialized || !snapshot.hasTrainableModel ||
                             snapshot.trainingComplete || state.workerActive ||
                             deviceSettingsPending);
        if (ImGui::Button("Capture One Optimizer Step and Pause")) actions.captureOptimizerStep = true;
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
            ImGui::SetTooltip("ACTUALLY TRAINS ONE STEP using the paused model's existing configuration, then pauses and exports pre/post state (~1.97 GB at 1.25M Gaussians). Next step must be ordinary fused, not final/ADC/reset. No automatic extra steps or layout changes.");
        ImGui::EndDisabled();
    }

    ImGui::Text("Dataset valid %s, loaded %s", state.datasetValid ? "yes" : "no",
                state.datasetLoaded ? "yes" : "no");
    ImGui::Text("Training initialized %s, running %s", state.trainingInitialized ? "yes" : "no",
                state.trainingRunning ? "yes" : "no");
    if (!hasDatasetPath) {
        ImGui::TextColored(ImVec4(1.0f, 0.65f, 0.25f, 1.0f),
                           "Choose a dataset folder before starting training.");
    } else if (!state.datasetLoaded) {
        ImGui::TextWrapped("Start Training will validate and load the dataset first.");
    }

    const bool pureWorkerRunning = state.workerActive && state.pureTrainingActive;
    if (!pureWorkerRunning) {
        if (snapshot.totalIterations > 0) {
            ImGui::Text("Iterations %u / %u", snapshot.trainingIteration, snapshot.totalIterations);
        } else {
            ImGui::Text("Iterations %u", snapshot.trainingIteration);
        }
        const uint64_t displayedSteps = state.workerActive
                                            ? static_cast<uint64_t>(snapshot.trainingIteration)
                                            : state.completedSteps;
        ImGui::Text("Steps %llu", static_cast<unsigned long long>(displayedSteps));
    }
    ImGui::Text("Total training time %.*s%s", static_cast<int>(state.elapsedTime.size()),
                state.elapsedTime.data(), state.timerRunning ? " (running)" : "");
    ImGui::Text("Model setup %.3f s", snapshot.modelPreparationSeconds);
    if (snapshot.gpuResidentImages) {
        ImGui::Text("Image preload %.3f s", snapshot.imagePreparationSeconds);
        ImGui::Text("Preload + model setup + training %.3f s",
                    snapshot.imagePreparationSeconds + snapshot.modelPreparationSeconds +
                        state.elapsedSeconds);
    }
    if (!pureWorkerRunning) {
        ImGui::Text("Gaussians %u", snapshot.gaussianCount);
    } else {
        ImGui::TextDisabled(
            "Pure mode: detailed training data will be published after completion.");
    }

    if (!state.trainingRunning) {
        ImGui::InputScalar("Benchmark Frame", ImGuiDataType_U32, &settings.run.benchmarkFrame);
        settings.run.benchmarkFrame =
            state.frameCount > 0u ? std::min(settings.run.benchmarkFrame, state.frameCount - 1u)
                                  : 0u;
        ImGui::InputScalar("Benchmark Warmup", ImGuiDataType_U32,
                           &settings.run.benchmarkWarmupSteps);
        ImGui::InputScalar("Benchmark Measured", ImGuiDataType_U32,
                           &settings.run.benchmarkMeasuredSteps);
        settings.run.benchmarkMeasuredSteps = std::max(settings.run.benchmarkMeasuredSteps, 1u);
    }
    const auto& benchmark = snapshot.fixedBenchmark;
    if (benchmark.checkpointDiagnosticsRequested) {
        ImGui::Text("Checkpoint %s: %u built, %u skipped, %u GPU fallbacks; flags=%u",
                    benchmark.checkpointWarpRequested ? "P2 warp" : "P1 state",
                    benchmark.checkpointDiagnosticSteps, benchmark.checkpointSkippedSteps,
                    benchmark.checkpointFallbackCount, benchmark.checkpointErrorFlags);
    }
    if (benchmark.active || benchmark.complete || benchmark.completedWarmupSteps > 0u ||
        benchmark.completedMeasuredSteps > 0u) {
        ImGui::Text("Fixed benchmark frame %u, warmup %u/%u, measured %u/%u, %s",
                    benchmark.frameIndex, benchmark.completedWarmupSteps, benchmark.warmupSteps,
                    benchmark.completedMeasuredSteps, benchmark.measuredSteps,
                    benchmark.active ? "running" : (benchmark.complete ? "complete" : "stopped"));
        if (benchmark.tileMode == TrainingTileMode::AdaptiveSpatial &&
            benchmark.pixelMode == TrainingPixelTo2DGSMode::VkSplatTensor) {
            ImGui::Text("Benchmark child Tensor batch %u", benchmark.adaptiveTensorChildBatch);
        }
        ImGui::Text("Benchmark SSIM: %s", trainingSsimModeName(benchmark.ssimMode));
        ImGui::Text("Benchmark backward pixel cull: %s", benchmark.pixelCullMethod == 1u
                    ? "EarlyCull FP32 guarded v1" : "Off");
        if (benchmark.tileMode == TrainingTileMode::Fixed16 &&
            benchmark.pixelMode == TrainingPixelTo2DGSMode::VkSplatTensor && !benchmark.checkpointWarpRequested) {
            ImGui::Text("Benchmark Tensor reduction: %s",
                        benchmark.tensorSubgroupReduction ? "Subgroup" : "Shared");
            ImGui::Text("Benchmark Tensor pixel grad: %s",
                        benchmark.tensorGlobalPixelGrad ? "Global" : "Shared");
            if (benchmark.tensorGlobalPixelGrad) {
                ImGui::Text("Benchmark Tensor RGB schedule: %s",
                            benchmark.tensorRgbLatencyHiding ? "Latency Hiding" : "Baseline");
            }
        }
    }
    if (benchmark.complete && !state.trainingRunning) {
        if (ImGui::Button("Export Benchmark Gradients")) {
            actions.exportFixedBenchmarkGradients = true;
        }
        if (benchmark.tileMode == TrainingTileMode::Fixed16 &&
            benchmark.pixelMode == TrainingPixelTo2DGSMode::VkSplatTensor &&
            benchmark.tensorGlobalPixelGrad && !benchmark.tensorRgbLatencyHiding &&
            !benchmark.pixelCullMethod && !benchmark.checkpointDiagnosticsRequested) {
            if (ImGui::Button("Export Scanline Inputs")) actions.exportScanlineDiagnosticInputs = true;
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("One-time frozen input download for offline scanline feasibility diagnosis; no gradient or training changes.");
        }
        if (benchmark.tileMode == TrainingTileMode::AdaptiveSpatial) {
            if (ImGui::Button("Export Adaptive Tile Diagnostics")) {
                actions.exportAdaptiveTileDiagnostics = true;
            }
        }
    }
    if (!pureWorkerRunning) {
        TrainingDiagnosticsPanel::draw(snapshot, state.frameCount);
    }

    ImGui::Separator();
    if (state.trainingRunning) {
        ImGui::BeginDisabled();
    }
    ImGui::InputText("Output Folder", settings.paths.outputDirectory.data(),
                     settings.paths.outputDirectory.size());
    ImGui::SameLine();
    if (ImGui::Button("Browse##OutputFolder")) {
        actions.browseOutput = true;
    }
    ImGui::InputText("PLY Name", settings.paths.outputName.data(),
                     settings.paths.outputName.size());
    if (ImGui::Button("Save PLY")) {
        actions.savePly = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Save As")) {
        actions.savePlyAs = true;
    }
    if (state.trainingRunning) {
        ImGui::EndDisabled();
    }

    if (!state.status.empty()) {
        ImGui::TextWrapped("%.*s", static_cast<int>(state.status.size()), state.status.data());
    }
    if (state.errorPopupPending) {
        ImGui::OpenPopup("Training Error");
        actions.clearErrorPopupPending = true;
    }
    if (ImGui::BeginPopupModal("Training Error", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::TextWrapped("%.*s", static_cast<int>(state.error.size()), state.error.data());
        if (ImGui::Button("OK")) {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    return actions;
}

} // namespace vulkan3DGS
