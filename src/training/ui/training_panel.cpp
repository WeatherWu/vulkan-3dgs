#include "training/ui/training_panel.hpp"

#include "training/app/training_controller.hpp"
#include "training/ui/training_command_panel.hpp"

#include <imgui.h>

#include <algorithm>
#include <cfloat>
#include <filesystem>
#include <iomanip>
#include <sstream>

namespace vulkan3DGS {
namespace {

std::filesystem::path existingDirectoryOrCurrent(const std::string& value) {
    const std::filesystem::path path(value);
    return std::filesystem::is_directory(path)
        ? path
        : std::filesystem::current_path();
}

std::string gpuLabel(const TrainingGpuInfo& info) {
    constexpr double bytesPerGiB = 1024.0 * 1024.0 * 1024.0;
    std::ostringstream stream;
    stream << '[' << info.vulkanIndex << "] " << info.name << " ("
           << info.typeName << ", " << std::fixed << std::setprecision(1)
           << static_cast<double>(info.deviceLocalMemoryBytes) / bytesPerGiB
           << " GiB)";
    return stream.str();
}

} // namespace

void TrainingPanel::draw(TrainingController& controller) {
    ImGui::Begin("Training");
    drawGpuControl(controller);
    ImGui::Separator();

    const TrainingLifecycleState& lifecycle = controller.state();
    TrainingSettings displaySettings = controller.settings();
    TrainingSettings& settings = lifecycle.running
        ? displaySettings
        : controller.editableSettings();
    const TrainingUiSnapshot snapshot = controller.snapshotForUi();
    const std::string elapsed = controller.elapsedText();
    const TrainingCommandPanelState state{
        .datasetValid = lifecycle.datasetValid,
        .datasetLoaded = lifecycle.datasetLoaded,
        .trainingInitialized = lifecycle.initialized,
        .trainingRunning = lifecycle.running,
        .pureTrainingActive = lifecycle.pureActive,
        .workerActive = controller.isActive(),
        .stopRequested = controller.isStopRequested(),
        .timerRunning = controller.isTimerRunning(),
        .errorPopupPending = lifecycle.errorPopupPending,
        .frameCount = lifecycle.frameCount,
        .imageWidth = lifecycle.imageWidth,
        .imageHeight = lifecycle.imageHeight,
        .completedSteps = lifecycle.stepsDone,
        .elapsedTime = elapsed,
        .status = lifecycle.status,
        .error = lifecycle.error,
    };
    const TrainingCommandPanelActions actions =
        TrainingCommandPanel::draw(settings, state, snapshot);
    dispatch(controller, actions, snapshot);
    if (!controller.state().running) drawFileDialogs(controller);
    ImGui::End();
}

void TrainingPanel::drawGpuControl(TrainingController& controller) {
    const auto options = controller.gpuOptions();
    if (options.empty()) {
        ImGui::TextDisabled("Training GPU unavailable");
        return;
    }
    const auto selected = std::find_if(
        options.begin(), options.end(),
        [](const TrainingGpuInfo& info) { return info.selected; });
    if (selected != options.end()) {
        gpuSelection_ = static_cast<size_t>(std::distance(options.begin(), selected));
    }
    gpuSelection_ = std::min(gpuSelection_, options.size() - 1u);

    if (controller.state().running) ImGui::BeginDisabled();
    ImGui::SetNextItemWidth(-FLT_MIN);
    const std::string preview = gpuLabel(options[gpuSelection_]);
    if (ImGui::BeginCombo("Training GPU", preview.c_str())) {
        for (size_t index = 0; index < options.size(); ++index) {
            const bool isSelected = index == gpuSelection_;
            const std::string label = gpuLabel(options[index]);
            if (ImGui::Selectable(label.c_str(), isSelected)) {
                gpuSelection_ = index;
                if (!options[index].selected) {
                    controller.selectGpu(options[index].uuid.empty()
                        ? std::to_string(options[index].vulkanIndex)
                        : options[index].uuid);
                }
            }
            if (isSelected) ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }
    if (controller.state().running) ImGui::EndDisabled();
    if (ImGui::IsItemHovered()) {
        const auto& gpu = options[gpuSelection_];
        ImGui::BeginTooltip();
        ImGui::Text("Vulkan %s", gpu.apiVersion.c_str());
        ImGui::Text("Driver %s", gpu.driverVersion.c_str());
        if (!gpu.uuid.empty()) ImGui::Text("UUID %s", gpu.uuid.c_str());
        ImGui::EndTooltip();
    }
}

void TrainingPanel::dispatch(TrainingController& controller,
                             const TrainingCommandPanelActions& actions,
                             const TrainingUiSnapshot&) {
    const TrainingSettings& settings = controller.settings();
    if (actions.clearErrorPopupPending) controller.clearErrorPopup();
    if (actions.datasetPathChanged) {
        controller.invalidateDataset(
            "Dataset path changed. Validate or start training to load it.");
    }
    if (actions.downscaleChanged) {
        controller.invalidateDataset(
            "Dataset downscale changed. Validate or start training to load it.");
    }
    if (actions.browseDataset) {
        fileDialogs_.openDatasetFolder(existingDirectoryOrCurrent(
            textBufferString(settings.paths.dataset)).string());
    }
    if (actions.validateDataset) controller.validateDataset();
    if (actions.loadDataset) controller.loadDataset();

    if (actions.settings.pixelBackwardChanged ||
        actions.settings.subgroupUtilizationChanged ||
        actions.settings.forwardCompositeChanged ||
        actions.settings.validationIntervalChanged) {
        controller.applyLiveSettings();
    }
    if (actions.toggleTraining) controller.toggleTraining();
    if (actions.startFixedBenchmark) controller.startFixedBenchmark();

    if (actions.browseOutput) {
        fileDialogs_.openOutputFolder(existingDirectoryOrCurrent(
            textBufferString(settings.paths.outputDirectory)).string());
    }
    if (actions.savePly) controller.exportDefaultModel();
    if (actions.savePlyAs) {
        fileDialogs_.openSavePly(existingDirectoryOrCurrent(
            textBufferString(settings.paths.outputDirectory)).string(),
            textBufferString(settings.paths.outputName));
    }
}

void TrainingPanel::drawFileDialogs(TrainingController& controller) {
    const TrainingFileDialogResult result = fileDialogs_.draw();
    if (result.datasetDirectory) {
        controller.setDatasetPath(*result.datasetDirectory, true);
    }
    if (result.outputDirectory) {
        controller.setOutputDirectory(*result.outputDirectory);
    }
    if (result.exportPath) {
        controller.setOutputPath(*result.exportPath);
        controller.exportModelTo(*result.exportPath);
    }
}

} // namespace vulkan3DGS
