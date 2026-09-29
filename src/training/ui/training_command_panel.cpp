#include "training/ui/training_command_panel.hpp"

#include "training/ui/training_diagnostics_panel.hpp"

#include <imgui.h>

#include <algorithm>

namespace vulkan3DGS {

TrainingCommandPanelActions TrainingCommandPanel::draw(
    TrainingSettings& settings,
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
    settings.initialization.downscale =
        std::clamp(settings.initialization.downscale, 1, 16);
    actions.downscaleChanged =
        settings.initialization.downscale != previousDownscale;

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
    actions.settings = TrainingSettingsPanel::drawTrainingAndDensification(
        settings, snapshot, state.trainingRunning);

    const bool hasDatasetPath = !textBufferString(settings.paths.dataset).empty();
    if (!hasDatasetPath) {
        ImGui::BeginDisabled();
    }
    const bool benchmarkActive = snapshot.fixedBenchmark.active;
    const char* primaryLabel = state.stopRequested
        ? "Stopping Training..."
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
    }
    if (!hasDatasetPath) {
        ImGui::EndDisabled();
    }

    ImGui::Text("Dataset valid %s, loaded %s",
                state.datasetValid ? "yes" : "no",
                state.datasetLoaded ? "yes" : "no");
    ImGui::Text("Training initialized %s, running %s",
                state.trainingInitialized ? "yes" : "no",
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
    ImGui::Text("Total training time %s%s",
                state.elapsedTime.data(),
                state.timerRunning ? " (running)" : "");
    if (!pureWorkerRunning) {
        ImGui::Text("Gaussians %u", snapshot.gaussianCount);
    } else {
        ImGui::TextDisabled(
            "Pure mode: detailed training data will be published after completion.");
    }

    if (!state.trainingRunning) {
        ImGui::InputScalar("Benchmark Frame", ImGuiDataType_U32,
                           &settings.run.benchmarkFrame);
        settings.run.benchmarkFrame = state.frameCount > 0u
            ? std::min(settings.run.benchmarkFrame, state.frameCount - 1u)
            : 0u;
        ImGui::InputScalar("Benchmark Warmup", ImGuiDataType_U32,
                           &settings.run.benchmarkWarmupSteps);
        ImGui::InputScalar("Benchmark Measured", ImGuiDataType_U32,
                           &settings.run.benchmarkMeasuredSteps);
        settings.run.benchmarkMeasuredSteps =
            std::max(settings.run.benchmarkMeasuredSteps, 1u);
    }
    const auto& benchmark = snapshot.fixedBenchmark;
    if (benchmark.active || benchmark.complete || benchmark.completedWarmupSteps > 0u ||
        benchmark.completedMeasuredSteps > 0u) {
        ImGui::Text("Fixed benchmark frame %u, warmup %u/%u, measured %u/%u, %s",
                    benchmark.frameIndex,
                    benchmark.completedWarmupSteps,
                    benchmark.warmupSteps,
                    benchmark.completedMeasuredSteps,
                    benchmark.measuredSteps,
                    benchmark.active ? "running" : (benchmark.complete ? "complete" : "stopped"));
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
        ImGui::TextWrapped("%s", state.status.data());
    }
    if (state.errorPopupPending) {
        ImGui::OpenPopup("Training Error");
        actions.clearErrorPopupPending = true;
    }
    if (ImGui::BeginPopupModal("Training Error", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::TextWrapped("%s", state.error.data());
        if (ImGui::Button("OK")) {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    return actions;
}

} // namespace vulkan3DGS
