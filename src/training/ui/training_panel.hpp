#pragma once

#include "training/ui/training_file_dialogs.hpp"

#include <cstddef>

namespace vulkan3DGS {

class TrainingController;
struct TrainingCommandPanelActions;
struct TrainingUiSnapshot;

// Owns all training presentation state and file dialogs. It translates ImGui
// actions into TrainingController commands and never touches GaussianTraining.
class TrainingPanel {
public:
    void draw(TrainingController& controller);

private:
    void drawGpuControl(TrainingController& controller);
    void dispatch(TrainingController& controller,
                  const TrainingCommandPanelActions& actions,
                  const TrainingUiSnapshot& snapshot);
    void drawFileDialogs(TrainingController& controller);

    size_t gpuSelection_ = 0;
    TrainingFileDialogs fileDialogs_;
};

} // namespace vulkan3DGS
