#pragma once

#include "graphics/gaussian_model.hpp"
#include "graphics/render_profile.hpp"
#include "viewer/camera.hpp"
#include "viewer/camera_controller.hpp"
#include "vulkan/swapchain.hpp"
#include "viewer/viewer_settings.hpp"

#include <filesystem>
#include <memory>
#include <string>
#include <algorithm>

namespace vulkan3DGS {
struct ViewerRenderData {
    const GaussianModel* model = nullptr;
    glm::mat4 view{1.0f};
    glm::mat4 projection{1.0f};
    glm::mat4 modelMatrix{1.0f};
    Camera camera{};
};
struct ViewerInputState {
    double mouseX=0.0, mouseY=0.0;
    bool orbitButton=false, panButton=false;
    bool captureMouse=false, captureKeyboard=false;
    bool toggleMode=false, focus=false, resetFocus=false;
    bool moveLeft=false, moveRight=false, moveUp=false, moveDown=false;
    bool moveForward=false, moveBackward=false;
    float speedMultiplier=1.0f;
    int framebufferWidth=1, framebufferHeight=1;
};
class ViewerController {
public:
    void loadSettings();
    void loadSettings(const std::filesystem::path& path);
    void saveSettings();
    void saveSettings(const std::filesystem::path& path);
    void setExternalModel(const GaussianModel* model);
    bool loadModel(const std::string& path);
    void setFlipY(bool value); void setFlipZ(bool value);
    [[nodiscard]] bool flipY() const { return settings_.flipY; }
    [[nodiscard]] bool flipZ() const { return settings_.flipZ; }
    void setRenderProfile(GaussianRenderProfile value) { settings_.renderProfile = value; }
    [[nodiscard]] GaussianRenderProfile renderProfile() const { return settings_.renderProfile; }
    void setSHBands(uint32_t value) { settings_.shBands = std::min(value, 3u); }
    [[nodiscard]] uint32_t shBands() const { return settings_.shBands; }
    void setPresentMode(PresentModePreference value) {
        settings_.presentMode = value;
        presentDirty_ = true;
    }
    [[nodiscard]] PresentModePreference presentMode() const { return settings_.presentMode; }
    bool consumePresentDirty() { const bool value=presentDirty_; presentDirty_=false; return value; }
    [[nodiscard]] const GaussianModel* model() const { return model_; }
    const glm::mat4& viewMatrix() const { return viewMatrix_; }
    const glm::mat4& projectionMatrix() const { return projectionMatrix_; }
    const glm::mat4& modelMatrix() const { return modelMatrix_; }
    Camera& camera() { return camera_; }
    const Camera& camera() const { return camera_; }
    bool hasCamera() const { return hasCamera_; }
    CameraController& cameraController() { return cameraController_; }
    const CameraController& cameraController() const { return cameraController_; }
    float nearPlane() const { return nearPlane_; }
    float farPlane() const { return farPlane_; }
    void setCamera(const Camera& camera);
    void syncCameraToModelFocus();
    void resetCamera(float viewportWidth, float viewportHeight);
    void focusCamera(float viewportWidth, float viewportHeight);
    void dolly(float wheelDelta) { cameraController_.dolly(wheelDelta); }
    void beginDrag(double x, double y, int button) { dragging_=true; dragButton_=button; lastMouseX_=x; lastMouseY_=y; }
    void endDrag() { dragging_=false; dragButton_=-1; }
    [[nodiscard]] bool dragging() const { return dragging_; }
    void resetDrag() { dragging_=false; dragButton_=-1; }
    bool updateDrag(double x,double y,int button,float& dx,float& dy) { if(!dragging_||dragButton_!=button){beginDrag(x,y,button);dx=dy=0.0f;return false;} dx=static_cast<float>(x-lastMouseX_);dy=static_cast<float>(y-lastMouseY_);lastMouseX_=x;lastMouseY_=y;return true; }
    void orbit(float x, float y) { cameraController_.orbit(x, y); }
    void look(float x, float y) { cameraController_.look(x, y); }
    void pan(float x, float y, float w, float h) { cameraController_.pan(x, y, w, h); }
    void fly(const glm::vec3& motion, float dt, float speed) { cameraController_.fly(motion, dt, speed); }
    void toggleCameraMode() { cameraController_.toggleMode(); }
    bool toggleKeyPressed(bool down) { const bool pressed=down&&!toggleKeyDown_; toggleKeyDown_=down; return pressed; }
    bool focusKeyPressed(bool down) { const bool pressed=down&&!focusKeyDown_; focusKeyDown_=down; return pressed; }
    void updateCamera(float deltaTime, int framebufferWidth, int framebufferHeight);
    void updateInput(const ViewerInputState& input, float deltaTime);
    void setCameraMatrices(const glm::mat4& view, const glm::mat4& projection);
    [[nodiscard]] ViewerRenderData renderData() const;
private:
    void updateModelMatrix();
    std::unique_ptr<GaussianModel> ownedModel_;
    const GaussianModel* model_ = nullptr;
    ViewerSettings settings_{};
    bool presentDirty_ = false;
    glm::mat4 modelMatrix_{1.0f};
    Camera camera_;
    bool hasCamera_ = false;
    CameraController cameraController_;
    glm::mat4 viewMatrix_{1.0f};
    glm::mat4 projectionMatrix_{1.0f};
    float nearPlane_ = 0.1f, farPlane_ = 100.0f;
    bool dragging_ = false;
    int dragButton_ = -1;
    double lastMouseX_ = 0.0, lastMouseY_ = 0.0;
    bool toggleKeyDown_ = false, focusKeyDown_ = false;
};
} // namespace vulkan3DGS
