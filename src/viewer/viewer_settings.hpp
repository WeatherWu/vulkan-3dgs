#pragma once

#include "graphics/render_profile.hpp"
#include "viewer/camera_controller.hpp"
#include "vulkan/swapchain.hpp"

#include <cstdint>
#include <filesystem>

namespace vulkan3DGS {

struct ViewerSettings {
    GaussianRenderProfile renderProfile = GaussianRenderProfile::Legacy;
    uint32_t shBands = 3;
    PresentModePreference presentMode = PresentModePreference::MaxFps;
    bool flipY = false;
    bool flipZ = false;
    CameraControlMode cameraMode = CameraControlMode::Orbit;
    float cameraFov = 75.0f;
    float cameraDamping = 0.12f;
    float orbitSensitivity = 0.005f;
    float panSensitivity = 1.0f;
    float zoomSensitivity = 0.12f;
    float flySpeed = 1.0f;
};

class ViewerSettingsStore {
public:
    static std::filesystem::path defaultPath();
    static void load(const std::filesystem::path& path, ViewerSettings& settings);
    static void save(const std::filesystem::path& path, const ViewerSettings& settings);
};

} // namespace vulkan3DGS
