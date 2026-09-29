#pragma once

#include "training/app/training_controller.hpp"
#include "training/app/training_settings.hpp"
#include "training/ui/training_settings_panel.hpp"

#include <cstdint>
#include <string_view>

namespace vulkan3DGS {

struct TrainingCommandPanelState {
    bool datasetValid = false;
    bool datasetLoaded = false;
    bool trainingInitialized = false;
    bool trainingRunning = false;
    bool pureTrainingActive = false;
    bool workerActive = false;
    bool stopRequested = false;
    bool timerRunning = false;
    bool errorPopupPending = false;
    uint32_t frameCount = 0;
    uint32_t imageWidth = 0;
    uint32_t imageHeight = 0;
    uint64_t completedSteps = 0;
    std::string_view elapsedTime;
    std::string_view status;
    std::string_view error;
};

struct TrainingCommandPanelActions {
    TrainingSettingsPanelActions settings{};
    bool datasetPathChanged = false;
    bool downscaleChanged = false;
    bool browseDataset = false;
    bool validateDataset = false;
    bool loadDataset = false;
    bool toggleTraining = false;
    bool startFixedBenchmark = false;
    bool browseOutput = false;
    bool savePly = false;
    bool savePlyAs = false;
    bool clearErrorPopupPending = false;
};

// Composes the training ImGui body from passive settings and immutable UI
// state. It returns typed intents for TrainingPanel to dispatch through the
// controller boundary.
class TrainingCommandPanel {
public:
    static TrainingCommandPanelActions draw(TrainingSettings& settings,
                                            const TrainingCommandPanelState& state,
                                            const TrainingUiSnapshot& snapshot);
};

} // namespace vulkan3DGS
