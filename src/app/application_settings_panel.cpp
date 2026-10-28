#include "app/application_settings_panel.hpp"

#include <imgui.h>

namespace vulkan3DGS {

bool ApplicationSettingsPanel::draw(ApplicationSettings& settings,
                                    ApplicationValidationSelection selection, bool actualValidation,
                                    const std::string& saveError) {
    bool changed = false;
    const auto* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(
        ImVec2(viewport->WorkPos.x + viewport->WorkSize.x - 12, viewport->WorkPos.y + 12),
        ImGuiCond_FirstUseEver, ImVec2(1, 0));
    ImGui::SetNextWindowSize(ImVec2(430, 0), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Application Settings")) {
        changed = ImGui::Checkbox("Vulkan Validation Layers (next launch)",
                                  &settings.validationLayersEnabled);
        const char* source = "Saved setting";
        if (selection.source == ApplicationValidationSource::CommandLine) source = "Command line";
        if (selection.source == ApplicationValidationSource::Smoke) source = "Smoke requirement";
        ImGui::Text("Current request: %s (%s)", selection.requested ? "On" : "Off", source);
        ImGui::Text("Actually enabled: %s", actualValidation ? "On" : "Off");
        if (selection.requested && !actualValidation)
            ImGui::TextWrapped("Requested validation is unavailable; inspect the startup log.");
        if (settings.validationLayersEnabled != selection.requested)
            ImGui::TextWrapped(
                "Restart to apply the saved setting. The current Vulkan instance is unchanged.");
        if (selection.source != ApplicationValidationSource::SavedSetting)
            ImGui::TextWrapped("This launch has an override. Saved settings apply to an ordinary "
                               "launch without overrides.");
        if (!saveError.empty())
            ImGui::TextColored(ImVec4(1, 0.35f, 0.25f, 1), "Settings save/load failed: %s",
                               saveError.c_str());
    }
    ImGui::End();
    return changed;
}

} // namespace vulkan3DGS
