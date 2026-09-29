#pragma once

#include "training/app/training_controller.hpp"
#include "training/app/training_settings.hpp"

namespace vulkan3DGS {

struct TrainingSettingsPanelActions {
    bool pixelBackwardChanged = false;
    bool subgroupUtilizationChanged = false;
    bool forwardCompositeChanged = false;
    bool validationIntervalChanged = false;
};

// ImGui editor for passive TrainingSettings. It does not mutate
// GaussianTraining; TrainingPanel forwards returned actions through
// TrainingController while the asynchronous worker is stopped.
class TrainingSettingsPanel {
public:
    static void drawInitialization(TrainingSettings& settings, bool trainingRunning);
    static TrainingSettingsPanelActions drawTrainingAndDensification(
        TrainingSettings& settings,
        const TrainingUiSnapshot& snapshot,
        bool trainingRunning);
};

} // namespace vulkan3DGS
