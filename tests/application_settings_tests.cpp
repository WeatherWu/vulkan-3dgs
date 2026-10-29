#include "app/application_settings.hpp"
#include "app/application_settings_panel.hpp"
#include "app/command_line_options.hpp"
#include "context/device.hpp"
#include "training/app/training_controller.hpp"
#include "training/app/training_settings.hpp"
#include "viewer/viewer_settings.hpp"
#include <imgui.h>
#include <imgui_internal.h>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>

using namespace vulkan3DGS;
namespace {
void check(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
std::string contents(const std::filesystem::path& path) {
    std::ifstream file(path);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}
void checkDeviceIsolation() {
    Device ordinary(vk::SurfaceKHR{}, {}, DeviceRole::Training);
    Device requested(vk::SurfaceKHR{}, {}, DeviceRole::Training, TrainingDeviceOptions{true});
    check(!ordinary.trainingStorage16Requested() && !ordinary.trainingStorage16Enabled(),
          "Baseline device requested 16-bit features");
    check(requested.trainingStorage16Requested() && !requested.trainingStorage16Enabled(),
          "Physical support mistaken for enabled features");
    bool failReplacement = false;
    uint32_t creates = 0;
    TrainingController controller([&](std::optional<std::string> selector, bool key16) {
        ++creates;
        if (failReplacement) throw std::runtime_error("injected device creation failure");
        return std::make_unique<Device>(vk::SurfaceKHR{}, std::move(selector), DeviceRole::Training,
                                        TrainingDeviceOptions{key16});
    });
    auto& settings = controller.editableSettings();
    settings.run.physicalTileKey16 = false;
    controller.initializeDevice(std::string("test"));
    check(!controller.requiresDatasetReload(), "Fresh baseline needs reload");
    settings.run.physicalTileKey16 = true;
    check(controller.requiresDatasetReload(), "Changed request did not require reload");
    controller.toggleTraining();
    check(!controller.state().running && creates == 1 &&
              controller.state().error.find("Load Dataset") != std::string::npos,
          "Pending setting advanced training or rebuilt device");
    settings.run.physicalTileKey16 = false;
    check(!controller.requiresDatasetReload(), "Reverted setting did not cancel pending reload");
    settings.run.physicalTileKey16 = true;
    failReplacement = true;
    const auto before = controller.state();
    controller.loadDataset();
    check(creates == 2 && !controller.storage16Requested() && controller.requiresDatasetReload() &&
              controller.state().stepsDone == before.stepsDone &&
              controller.state().initialized == before.initialized &&
              controller.state().datasetLoaded == before.datasetLoaded &&
              controller.state().error.find("injected device") != std::string::npos,
          "Failed replacement changed existing device or lifecycle");
    TrainingController unsupported([](std::optional<std::string> selector, bool key16) {
        return std::make_unique<Device>(vk::SurfaceKHR{}, std::move(selector), DeviceRole::Training,
                                        TrainingDeviceOptions{key16});
    });
    unsupported.editableSettings().run.physicalTileKey16 = true;
    unsupported.initializeDevice(std::string("test"));
    check(unsupported.storage16Requested() && !unsupported.storage16Enabled() &&
              !unsupported.requiresDatasetReload(),
          "Unavailable capability caused a repeated reload request");
}
void checkResolution() {
    const ApplicationSettings enabled, disabled{false};
    check(resolveApplicationValidation(enabled, {}, false).requested, "Default validation off");
    check(!resolveApplicationValidation(disabled, {}, false).requested, "Saved off ignored");
    for (const auto saved : {enabled, disabled}) {
        for (bool command : {false, true}) {
            const auto selection = resolveApplicationValidation(saved, command, false);
            check(selection.requested == command &&
                      selection.source == ApplicationValidationSource::CommandLine,
                  "CLI must override saved preference");
        }
        const auto smoke = resolveApplicationValidation(saved, {}, true);
        check(smoke.requested && smoke.source == ApplicationValidationSource::Smoke,
              "Smoke must enable validation without changing saved preference");
        check(resolveApplicationValidation(saved, true, true).requested, "Explicit smoke on");
        bool rejected = false;
        try {
            (void)resolveApplicationValidation(saved, false, true);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        check(rejected, "Smoke accepted explicit validation off");
    }
    const auto ordinary = parseApplicationCommandLine({});
    check(
        !resolveApplicationValidation(disabled, ordinary.validationLayersOverride, false).requested,
        "Ordinary launch must use saved off");
}

void checkValidationUi() {
    ImGui::CreateContext();
    auto& io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.DisplaySize = ImVec2(1280, 720);
    io.DeltaTime = 1.0f / 60.0f;
    unsigned char* pixels = nullptr;
    int width = 0, height = 0;
    io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
    ApplicationSettings settings;
    const auto selection = resolveApplicationValidation(settings, {}, false);
    const auto frame = [&] {
        ImGui::NewFrame();
        const bool changed = ApplicationSettingsPanel::draw(settings, selection, true, {});
        ImGui::Render();
        return changed;
    };
    (void)frame();
    (void)frame();
    const auto* window = ImGui::FindWindowByName("Application Settings");
    const bool exists = window != nullptr;
    bool changed = false;
    if (window) {
        // Click the first checkbox through ImGui input; no GLFW window or Vulkan instance.
        io.AddMousePosEvent(window->DC.CursorStartPos.x + 5,
                            window->DC.CursorStartPos.y + ImGui::GetFrameHeight() / 2);
        (void)frame();
        io.AddMouseButtonEvent(0, true);
        changed |= frame();
        io.AddMouseButtonEvent(0, false);
        changed |= frame();
    }
    const bool toggledPreference = !settings.validationLayersEnabled;
    ImGui::DestroyContext();
    check(exists && changed && toggledPreference,
          "Validation UI checkbox did not change preference");
    check(selection.requested, "UI changed the current launch request");
}
} // namespace

int main() {
    try {
        checkDeviceIsolation();
        checkResolution();
        checkValidationUi();
        const auto root =
            std::filesystem::temp_directory_path() /
            ("vulkan-3dgs-app-settings-" +
             std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directory(root);
        const auto path = root / "settings.cfg";
        ApplicationSettings application;
        ApplicationSettingsStore::load(path, application);
        check(application.validationLayersEnabled, "Missing config must default on");
        {
            std::ofstream file(path);
            file << "version 1\n# shared configuration\nvulkan_validation 0\n\n"
                    "dataset \"a dataset\"\noutput_dir \"output\"\noutput_name \"model.ply\"\n"
                    "faster_sort 1\nmorton_ordering 1\ndirect_intrinsics_projection "
                    "1\nphysical_tile_key16 1\nrender_profile 1\ncustom_key keep\n";
        }
        ApplicationSettingsStore::load(path, application);
        check(!application.validationLayersEnabled, "Blank line changed saved off");
        ApplicationSettingsStore::save(path, application);
        check(contents(path).find("custom_key keep") != std::string::npos &&
                  contents(path).find("# shared configuration") != std::string::npos,
              "Application save discarded unowned data");
        TrainingSettings training;
        TrainingSettingsStore::load(path, training);
        check(training.run.fasterSort, "Training sort preference missing");
        check(training.run.directIntrinsicsProjection && training.run.physicalTileKey16,
              "New method preferences missing");
        check(training.run.mortonOrdering, "Morton preference missing");
        const auto run = makeTrainingRunConfig(training);
        training.run.fasterSort = false;
        training.run.mortonOrdering = false;
        training.run.directIntrinsicsProjection = false;
        training.run.physicalTileKey16 = false;
        check(run.run.fasterSort, "Run configuration observed later UI edits");
        check(run.run.directIntrinsicsProjection && run.run.physicalTileKey16,
              "New method run settings were mutable");
        check(run.run.mortonOrdering, "Morton run configuration observed later UI edits");
        check(TrainingSettingsStore::describeRun(run).find("morton_ordering 1") !=
                  std::string::npos,
              "Run description omitted Morton choice");
        check(TrainingSettingsStore::describeRun(run).find("faster_sort 1") != std::string::npos,
              "Run description omitted sorting choice");
        TrainingSettingsStore::save(path, training);
        ViewerSettings viewer;
        ViewerSettingsStore::load(path, viewer);
        ViewerSettingsStore::save(path, viewer);
        ApplicationSettings reloaded;
        ApplicationSettingsStore::load(path, reloaded);
        check(!reloaded.validationLayersEnabled, "Domain saves lost application preference");
        TrainingSettings reloadedTraining;
        TrainingSettingsStore::load(path, reloadedTraining);
        check(!reloadedTraining.run.fasterSort, "Stale sorting preference survived save");
        check(!reloadedTraining.run.directIntrinsicsProjection &&
                  !reloadedTraining.run.physicalTileKey16,
              "New method save did not persist off");
        check(!reloadedTraining.run.mortonOrdering, "Stale Morton preference survived save");
        application.validationLayersEnabled = true;
        ApplicationSettingsStore::save(path, application);
        ApplicationSettingsStore::save(path, application);
        const auto text = contents(path);
        const auto first = text.find("vulkan_validation ");
        check(first != std::string::npos &&
                  text.find("vulkan_validation ", first + 1) == std::string::npos,
              "Repeated save duplicated application key");
        check(text.find("custom_key keep") != std::string::npos,
              "Cross-domain round trip lost custom key");
        {
            std::ofstream file(path);
            file << "vulkan_validation invalid\n";
        }
        reloaded.validationLayersEnabled = false;
        ApplicationSettingsStore::load(path, reloaded);
        check(reloaded.validationLayersEnabled, "Malformed preference must not disable validation");
        bool failed = false;
        try {
            ApplicationSettingsStore::save(root, application);
        } catch (const std::exception&) {
            failed = true;
        }
        check(failed, "Unwritable destination was reported as saved");
        std::filesystem::remove(path);
        std::filesystem::remove(root);
        std::cout << "Application settings and validation resolution passed\n";
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    } catch (...) {
        std::fputs("Unknown application settings test failure\n", stderr);
        return 1;
    }
}
