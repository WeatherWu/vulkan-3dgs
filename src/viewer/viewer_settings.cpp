#include "viewer/viewer_settings.hpp"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace vulkan3DGS {
namespace {

std::optional<std::filesystem::path> environmentDirectory(const char* name) {
#ifdef _WIN32
    char* value = nullptr;
    size_t length = 0;
    if (_dupenv_s(&value, &length, name) != 0 || !value || length <= 1u) {
        std::free(value);
        return std::nullopt;
    }
    std::filesystem::path path(value);
    std::free(value);
    return path;
#else
    const char* value = std::getenv(name);
    return value && value[0] != '\0' ? std::optional<std::filesystem::path>(value) : std::nullopt;
#endif
}

const std::unordered_set<std::string>& viewerKeys() {
    static const std::unordered_set<std::string> keys{"render_profile",
                                                      "gaussian_sh_bands",
                                                      "present_mode",
                                                      "flip_y",
                                                      "flip_z",
                                                      "camera_mode",
                                                      "supersplat_camera_mode",
                                                      "camera_fov",
                                                      "supersplat_camera_fov",
                                                      "camera_damping",
                                                      "supersplat_camera_damping",
                                                      "camera_orbit_sensitivity",
                                                      "supersplat_orbit_sensitivity",
                                                      "camera_pan_sensitivity",
                                                      "supersplat_pan_sensitivity",
                                                      "camera_zoom_sensitivity",
                                                      "supersplat_zoom_sensitivity",
                                                      "camera_fly_speed",
                                                      "supersplat_fly_speed"};
    return keys;
}

std::vector<std::string> unownedLines(const std::filesystem::path& path) {
    std::ifstream input(path);
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(input, line)) {
        std::istringstream row(line);
        std::string key;
        row >> key;
        if (!key.empty() && key != "version" && !viewerKeys().contains(key)) {
            lines.push_back(line);
        }
    }
    return lines;
}

} // namespace

std::filesystem::path ViewerSettingsStore::defaultPath() {
#ifdef _WIN32
    if (auto appData = environmentDirectory("APPDATA")) {
        return *appData / "vulkan-3dgs" / "training-settings.cfg";
    }
#else
    if (auto configHome = environmentDirectory("XDG_CONFIG_HOME")) {
        return *configHome / "vulkan-3dgs" / "training-settings.cfg";
    }
    if (auto home = environmentDirectory("HOME")) {
        return *home / ".config" / "vulkan-3dgs" / "training-settings.cfg";
    }
#endif
    return std::filesystem::current_path() / ".vulkan-3dgs-training-settings.cfg";
}

void ViewerSettingsStore::load(const std::filesystem::path& path, ViewerSettings& settings) {
    std::ifstream input(path);
    if (!input.is_open()) return;

    std::string line;
    while (std::getline(input, line)) {
        std::istringstream row(line);
        std::string key;
        row >> key;
        if (key == "render_profile") {
            int value = 0;
            if (row >> value)
                settings.renderProfile =
                    static_cast<GaussianRenderProfile>(std::clamp(value, 0, 1));
        } else if (key == "gaussian_sh_bands") {
            row >> settings.shBands;
            settings.shBands = std::min(settings.shBands, 3u);
        } else if (key == "present_mode") {
            int value = 0;
            if (row >> value)
                settings.presentMode = static_cast<PresentModePreference>(std::clamp(value, 0, 2));
        } else if (key == "flip_y") {
            int value = 0;
            if (row >> value) settings.flipY = value != 0;
        } else if (key == "flip_z") {
            int value = 0;
            if (row >> value) settings.flipZ = value != 0;
        } else if (key == "camera_mode" || key == "supersplat_camera_mode") {
            int value = 0;
            if (row >> value)
                settings.cameraMode = static_cast<CameraControlMode>(std::clamp(value, 0, 1));
        } else if (key == "camera_fov" || key == "supersplat_camera_fov") {
            row >> settings.cameraFov;
        } else if (key == "camera_damping" || key == "supersplat_camera_damping") {
            row >> settings.cameraDamping;
        } else if (key == "camera_orbit_sensitivity" || key == "supersplat_orbit_sensitivity") {
            row >> settings.orbitSensitivity;
        } else if (key == "camera_pan_sensitivity" || key == "supersplat_pan_sensitivity") {
            row >> settings.panSensitivity;
        } else if (key == "camera_zoom_sensitivity" || key == "supersplat_zoom_sensitivity") {
            row >> settings.zoomSensitivity;
        } else if (key == "camera_fly_speed" || key == "supersplat_fly_speed") {
            row >> settings.flySpeed;
        }
    }
}

void ViewerSettingsStore::save(const std::filesystem::path& path, const ViewerSettings& settings) {
    const auto preserved = unownedLines(path);
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::trunc);
    if (!output.is_open()) {
        throw std::runtime_error("Could not save viewer settings to " + path.string());
    }
    output << "version 1\n";
    for (const auto& line : preserved)
        output << line << '\n';
    output << std::setprecision(std::numeric_limits<float>::max_digits10);
    output << "render_profile " << static_cast<int>(settings.renderProfile) << '\n';
    output << "gaussian_sh_bands " << settings.shBands << '\n';
    output << "present_mode " << static_cast<int>(settings.presentMode) << '\n';
    output << "flip_y " << (settings.flipY ? 1 : 0) << '\n';
    output << "flip_z " << (settings.flipZ ? 1 : 0) << '\n';
    output << "camera_mode " << static_cast<int>(settings.cameraMode) << '\n';
    output << "camera_fov " << settings.cameraFov << '\n';
    output << "camera_damping " << settings.cameraDamping << '\n';
    output << "camera_orbit_sensitivity " << settings.orbitSensitivity << '\n';
    output << "camera_pan_sensitivity " << settings.panSensitivity << '\n';
    output << "camera_zoom_sensitivity " << settings.zoomSensitivity << '\n';
    output << "camera_fly_speed " << settings.flySpeed << '\n';
}

} // namespace vulkan3DGS
