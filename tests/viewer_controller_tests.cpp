#include "training/app/training_settings.hpp"
#include "viewer/viewer_controller.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>

#include <glm/glm.hpp>

using namespace vulkan3DGS;

namespace {

bool testInputEdgesAndViewport() {
    ViewerController controller;
    controller.cameraController().setDampingSeconds(0.0f);
    controller.resetCamera(1920.0f, 1080.0f);
    if (!controller.hasCamera()) return false;

    ViewerInputState input{};
    input.framebufferWidth = 1920;
    input.framebufferHeight = 1080;

    const float initialAzimuth = controller.cameraController().azimuthDegrees();
    input.orbitButton = true;
    input.mouseX = 100.0;
    input.mouseY = 100.0;
    controller.updateInput(input, 1.0f / 60.0f);
    input.mouseX = 140.0;
    controller.updateInput(input, 1.0f / 60.0f);
    if (controller.cameraController().azimuthDegrees() == initialAzimuth) {
        return false;
    }

    input.orbitButton = false;
    controller.updateInput(input, 1.0f / 60.0f);
    const glm::vec3 focalBeforePan = controller.cameraController().focalPoint();
    input.panButton = true;
    input.mouseX = 200.0;
    controller.updateInput(input, 1.0f / 60.0f);
    input.mouseX = 225.0;
    input.mouseY = 115.0;
    controller.updateInput(input, 1.0f / 60.0f);
    if (glm::length(controller.cameraController().focalPoint() - focalBeforePan) <=
        1e-5f) {
        return false;
    }

    input.panButton = false;
    input.captureMouse = true;
    input.orbitButton = true;
    const float capturedAzimuth =
        controller.cameraController().azimuthDegrees();
    input.mouseX = 400.0;
    controller.updateInput(input, 1.0f / 60.0f);
    input.mouseX = 500.0;
    controller.updateInput(input, 1.0f / 60.0f);
    if (controller.cameraController().azimuthDegrees() != capturedAzimuth) {
        return false;
    }

    input = ViewerInputState{};
    input.framebufferWidth = 1920;
    input.framebufferHeight = 1080;
    input.captureKeyboard = true;
    input.toggleMode = true;
    controller.updateInput(input, 1.0f / 60.0f);
    if (controller.cameraController().mode() != CameraControlMode::Orbit) return false;
    input.toggleMode = false;
    controller.updateInput(input, 1.0f / 60.0f);
    input.captureKeyboard = false;
    input.toggleMode = true;
    controller.updateInput(input, 1.0f / 60.0f);
    if (controller.cameraController().mode() != CameraControlMode::Fly) return false;

    input.toggleMode = false;
    input.moveForward = true;
    const glm::vec3 focalBeforeFly = controller.cameraController().focalPoint();
    controller.updateInput(input, 1.0f);
    if (glm::length(controller.cameraController().focalPoint() - focalBeforeFly) <=
        1e-5f) {
        return false;
    }

    input.moveForward = false;
    input.focus = true;
    input.resetFocus = true;
    controller.updateInput(input, 1.0f / 60.0f);
    if (controller.cameraController().mode() != CameraControlMode::Orbit) return false;
    input.focus = false;
    input.resetFocus = false;
    controller.updateInput(input, 1.0f / 60.0f);

    // V remains edge-triggered: holding the key must not toggle twice.
    input.toggleMode = true;
    controller.updateInput(input, 1.0f / 60.0f);
    if (controller.cameraController().mode() != CameraControlMode::Fly) return false;
    input.toggleMode = false;
    controller.updateInput(input, 1.0f / 60.0f);
    input.toggleMode = true;
    controller.updateInput(input, 1.0f / 60.0f);
    return controller.cameraController().mode() == CameraControlMode::Orbit;
}

bool testViewerSettingsRoundTripAndTrainingPreservation() {
    const auto path = std::filesystem::temp_directory_path() /
        "vulkan-3dgs-viewer-settings-test.cfg";
    TrainingSettings training{};
    setTextBuffer(training.paths.dataset, "preserved dataset");
    TrainingSettingsStore::save(path, training);

    ViewerController source;
    source.setRenderProfile(GaussianRenderProfile::SuperSplatCompatible);
    source.setSHBands(2);
    source.setPresentMode(PresentModePreference::VSync);
    source.setFlipY(true);
    source.cameraController().setMode(CameraControlMode::Fly);
    source.cameraController().setFovDegrees(61.0f);
    source.saveSettings(path);

    ViewerController loaded;
    loaded.loadSettings(path);
    TrainingSettings preserved{};
    TrainingSettingsStore::load(path, preserved);
    preserved.run.pureTraining = true;
    TrainingSettingsStore::save(path, preserved);
    ViewerController preservedViewer;
    preservedViewer.loadSettings(path);
    std::filesystem::remove(path);
    return loaded.renderProfile() ==
               GaussianRenderProfile::SuperSplatCompatible &&
           loaded.shBands() == 2 &&
           loaded.presentMode() == PresentModePreference::VSync &&
           loaded.flipY() &&
           loaded.cameraController().mode() == CameraControlMode::Fly &&
           loaded.cameraController().fovDegrees() == 61.0f &&
           textBufferString(preserved.paths.dataset) == "preserved dataset" &&
           preservedViewer.renderProfile() ==
               GaussianRenderProfile::SuperSplatCompatible &&
           preservedViewer.cameraController().fovDegrees() == 61.0f;
}

} // namespace

int main() {
    if (!testInputEdgesAndViewport()) {
        std::cerr << "Viewer input edge handling failed\n";
        return 1;
    }
    if (!testViewerSettingsRoundTripAndTrainingPreservation()) {
        std::cerr << "Viewer settings round-trip or shared-file preservation failed\n";
        return 1;
    }
    return 0;
}
