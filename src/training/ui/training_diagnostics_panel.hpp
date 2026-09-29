#pragma once

#include "training/app/training_controller.hpp"

#include <cstdint>

namespace vulkan3DGS {

// ImGui presentation for immutable worker diagnostics. This module deliberately
// has no access to GaussianTraining, worker lifecycle, or editable settings.
class TrainingDiagnosticsPanel {
public:
    static void draw(const TrainingUiSnapshot& snapshot, uint32_t frameCount);
};

} // namespace vulkan3DGS
