#include "training/ui/training_settings_panel.hpp"

#include <imgui.h>

#include <algorithm>

namespace vulkan3DGS {

void TrainingSettingsPanel::drawInitialization(TrainingSettings& settings, bool trainingRunning) {
    if (!ImGui::CollapsingHeader("Initialization", ImGuiTreeNodeFlags_DefaultOpen)) return;
    if (trainingRunning) ImGui::BeginDisabled();

    auto& init = settings.initialization;
    ImGui::InputScalar("Fallback Gaussians", ImGuiDataType_U32, &init.gaussianCount);
    init.gaussianCount = std::clamp(init.gaussianCount, 1u, 1000000u);
    ImGui::InputScalar("Random Seed", ImGuiDataType_U32, &init.randomSeed);
    ImGui::InputFloat("Initial Opacity", &init.opacity, 0.01f, 0.05f, "%.4f");
    init.opacity = std::clamp(init.opacity, 0.001f, 0.99f);
    ImGui::InputFloat("Scene Radius Scale", &init.sceneRadiusScale, 0.1f, 1.0f, "%.3f");
    init.sceneRadiusScale = std::clamp(init.sceneRadiusScale, 0.01f, 100.0f);

    if (trainingRunning) ImGui::EndDisabled();
}

TrainingSettingsPanelActions TrainingSettingsPanel::drawTrainingAndDensification(
    TrainingSettings& settings, const TrainingUiSnapshot& snapshot, bool trainingRunning) {
    TrainingSettingsPanelActions actions{};
    const bool canEdit = !trainingRunning;
    auto& schedule = settings.schedule;
    auto& run = settings.run;
    auto& optimizer = settings.optimizer;
    auto& densify = settings.densification;

    if (ImGui::CollapsingHeader("Training", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (!canEdit) ImGui::BeginDisabled();

        static constexpr const char* trainingModes[] = {"Sequential", "3DGS Random"};
        ImGui::Combo("Mode", &schedule.imageSelectionMode, trainingModes,
                     IM_ARRAYSIZE(trainingModes));
        schedule.imageSelectionMode = std::clamp(schedule.imageSelectionMode, 0, 1);

        static constexpr const char* pixelBackwardModes[] = {
            "Auto (Adaptive)",          "Direct",
            "Workgroup Shared",         "Subgroup",
            "Tile Gaussian Atomic",     "VkSplat Per-Splat",
            "VkSplat Tensor (Adapted)",
        };
        if (ImGui::Combo("Pixel Backward", &run.pixelTo2DGSMode, pixelBackwardModes,
                         IM_ARRAYSIZE(pixelBackwardModes))) {
            run.pixelTo2DGSMode = std::clamp(run.pixelTo2DGSMode, 0, 6);
            actions.pixelBackwardChanged = true;
        }
        if (ImGui::SliderFloat("Auto Min Subgroup Utilization",
                               &run.pixelTo2DGSMinSubgroupUtilization, 0.0f, 1.0f, "%.2f")) {
            actions.subgroupUtilizationChanged = true;
        }
        if (snapshot.rendererInitialized) {
            const int activeMode = std::clamp(static_cast<int>(snapshot.activePixelMode), 0, 6);
            ImGui::Text("Active Pixel Backward %s", pixelBackwardModes[activeMode]);
            if (run.pixelTo2DGSMode == 4 && !snapshot.tileGaussianSupported)
                ImGui::TextDisabled("Tile Gaussian unavailable on this GPU; using Direct.");
            if (run.pixelTo2DGSMode == 5 && !snapshot.vkSplatPerSplatSupported)
                ImGui::TextDisabled("VkSplat Per-Splat unavailable on this GPU; using Direct.");
            if (run.pixelTo2DGSMode == 6 && !snapshot.vkSplatTensorSupported)
                ImGui::TextDisabled("VkSplat Tensor requires 45 KiB shared memory; using Direct.");
        }

        static constexpr const char* forwardCompositeModes[] = {"Direct", "Workgroup Shared"};
        if (ImGui::Combo("Forward Composite", &run.forwardCompositeMode, forwardCompositeModes,
                         IM_ARRAYSIZE(forwardCompositeModes))) {
            run.forwardCompositeMode = std::clamp(run.forwardCompositeMode, 0, 1);
            actions.forwardCompositeChanged = true;
        }
        if (schedule.imageSelectionMode == 1) {
            schedule.totalIterations = 30000;
            ImGui::BeginDisabled();
            ImGui::InputScalar("Total Iterations", ImGuiDataType_U32, &schedule.totalIterations);
            ImGui::EndDisabled();
        }
        ImGui::InputScalar("Benchmark Steps/Frame", ImGuiDataType_U32, &run.benchmarkStepsPerFrame);
        run.benchmarkStepsPerFrame = std::max(run.benchmarkStepsPerFrame, 1u);
        ImGui::Checkbox("Pure Training", &run.pureTraining);
        if (run.pureTraining) {
            ImGui::TextDisabled(
                "Only elapsed time refreshes; the final PLY is exported automatically.");
        }
        if (ImGui::InputScalar("Validation Interval", ImGuiDataType_U32,
                               &schedule.validationInterval)) {
            actions.validationIntervalChanged = true;
        }

        ImGui::InputFloat("Position LR", &optimizer.positionLearningRate, 0.00001f, 0.0001f,
                          "%.6g");
        optimizer.positionLearningRate = std::max(optimizer.positionLearningRate, 0.0f);
        ImGui::InputFloat("Position LR Final", &optimizer.positionLearningRateFinal, 0.000001f,
                          0.00001f, "%.6g");
        optimizer.positionLearningRateFinal = std::max(optimizer.positionLearningRateFinal, 0.0f);
        ImGui::InputFloat("Position LR Delay Mult", &optimizer.positionLearningRateDelayMult, 0.01f,
                          0.05f, "%.6g");
        optimizer.positionLearningRateDelayMult =
            std::clamp(optimizer.positionLearningRateDelayMult, 0.0f, 1.0f);
        ImGui::InputFloat("Position LR Delay Steps", &optimizer.positionLearningRateDelaySteps,
                          100.0f, 1000.0f, "%.0f");
        optimizer.positionLearningRateDelaySteps =
            std::max(optimizer.positionLearningRateDelaySteps, 0.0f);
        ImGui::InputFloat("Position LR Steps", &optimizer.positionLearningRateMaxSteps, 1000.0f,
                          5000.0f, "%.0f");
        optimizer.positionLearningRateMaxSteps =
            std::max(optimizer.positionLearningRateMaxSteps, 1.0f);
        ImGui::InputFloat("Feature LR", &optimizer.featureLearningRate, 0.0001f, 0.001f, "%.6g");
        optimizer.featureLearningRate = std::max(optimizer.featureLearningRate, 0.0f);
        ImGui::InputFloat("Feature Rest LR", &optimizer.featureRestLearningRate, 0.00001f, 0.0001f,
                          "%.6g");
        optimizer.featureRestLearningRate = std::max(optimizer.featureRestLearningRate, 0.0f);
        ImGui::InputFloat("Opacity LR", &optimizer.opacityLearningRate, 0.001f, 0.01f, "%.6g");
        optimizer.opacityLearningRate = std::max(optimizer.opacityLearningRate, 0.0f);
        ImGui::InputFloat("Scale LR", &optimizer.scaleLearningRate, 0.0001f, 0.001f, "%.6g");
        optimizer.scaleLearningRate = std::max(optimizer.scaleLearningRate, 0.0f);
        ImGui::InputFloat("Rotation LR", &optimizer.rotationLearningRate, 0.0001f, 0.001f, "%.6g");
        optimizer.rotationLearningRate = std::max(optimizer.rotationLearningRate, 0.0f);
        ImGui::InputScalar("Max SH Degree", ImGuiDataType_U32, &optimizer.maxSHDegree);
        optimizer.maxSHDegree = std::min(optimizer.maxSHDegree, 3u);
        ImGui::InputScalar("SH Degree Interval", ImGuiDataType_U32, &optimizer.shDegreeInterval);
        optimizer.shDegreeInterval = std::max(optimizer.shDegreeInterval, 1u);
        ImGui::InputFloat("Adam Beta1", &optimizer.adamBeta1, 0.01f, 0.05f, "%.6g");
        optimizer.adamBeta1 = std::clamp(optimizer.adamBeta1, 0.0f, 0.999999f);
        ImGui::InputFloat("Adam Beta2", &optimizer.adamBeta2, 0.001f, 0.01f, "%.6g");
        optimizer.adamBeta2 = std::clamp(optimizer.adamBeta2, 0.0f, 0.999999f);
        ImGui::InputFloat("Adam Epsilon", &optimizer.adamEpsilon, 1e-15f, 1e-12f, "%.3e");
        optimizer.adamEpsilon = std::max(optimizer.adamEpsilon, 1e-15f);
        ImGui::InputFloat("Gradient Clip", &optimizer.gradientClip, 10.0f, 100.0f, "%.6g");
        optimizer.gradientClip = std::max(optimizer.gradientClip, 0.0f);
        ImGui::InputFloat("DSSIM Weight", &optimizer.lossDssimWeight, 0.01f, 0.05f, "%.6g");
        optimizer.lossDssimWeight = std::clamp(optimizer.lossDssimWeight, 0.0f, 1.0f);

        if (!canEdit) ImGui::EndDisabled();
    }

    if (ImGui::CollapsingHeader("Densification / Pruning", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (!canEdit) ImGui::BeginDisabled();

        ImGui::Checkbox("Enable Densification", &densify.enabled);
        ImGui::InputScalar("Densify From", ImGuiDataType_U32, &densify.fromIteration);
        ImGui::InputScalar("Densify Until", ImGuiDataType_U32, &densify.untilIteration);
        densify.untilIteration = std::max(densify.untilIteration, densify.fromIteration);
        ImGui::InputScalar("Densify Interval", ImGuiDataType_U32, &densify.interval);
        densify.interval = std::max(densify.interval, 1u);
        ImGui::InputScalar("Opacity Reset Interval", ImGuiDataType_U32,
                           &densify.opacityResetInterval);
        densify.opacityResetInterval = std::max(densify.opacityResetInterval, 1u);
        ImGui::InputScalar("Max Gaussians", ImGuiDataType_U32, &densify.maxGaussians);
        densify.maxGaussians = std::clamp(densify.maxGaussians, 1u, 10000000u);
        ImGui::InputScalar("Split Children", ImGuiDataType_U32, &densify.splitChildren);
        densify.splitChildren = std::clamp(densify.splitChildren, 2u, 8u);
        ImGui::InputFloat("Gradient Threshold", &densify.gradientThreshold, 0.00001f, 0.0001f,
                          "%.6g");
        densify.gradientThreshold = std::max(densify.gradientThreshold, 0.0f);
        ImGui::InputFloat("Min Opacity", &densify.minOpacity, 0.001f, 0.01f, "%.6g");
        densify.minOpacity = std::clamp(densify.minOpacity, 0.0f, 0.99f);
        ImGui::InputFloat("Percent Dense", &densify.percentDense, 0.001f, 0.01f, "%.6g");
        densify.percentDense = std::clamp(densify.percentDense, 0.0f, 1.0f);
        ImGui::InputFloat("Screen Prune Size", &densify.screenPruneSize, 1.0f, 10.0f, "%.3f");
        densify.screenPruneSize = std::max(densify.screenPruneSize, 0.0f);
        ImGui::InputFloat("World Prune Size", &densify.worldPruneSize, 0.001f, 0.01f, "%.6g");
        densify.worldPruneSize = std::max(densify.worldPruneSize, 0.0f);

        if (!canEdit) ImGui::EndDisabled();
    }
    return actions;
}

} // namespace vulkan3DGS
