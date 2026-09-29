#include "viewer/viewer_controller.hpp"
#include <glm/gtc/matrix_transform.hpp>
namespace vulkan3DGS {
void ViewerController::loadSettings() {
    loadSettings(ViewerSettingsStore::defaultPath());
}
void ViewerController::loadSettings(const std::filesystem::path& path) {
    ViewerSettingsStore::load(path, settings_);
    cameraController_.setMode(settings_.cameraMode);
    cameraController_.setFovDegrees(settings_.cameraFov);
    cameraController_.setDampingSeconds(settings_.cameraDamping);
    cameraController_.setOrbitSensitivity(settings_.orbitSensitivity);
    cameraController_.setPanSensitivity(settings_.panSensitivity);
    cameraController_.setZoomSensitivity(settings_.zoomSensitivity);
    cameraController_.setFlySpeed(settings_.flySpeed);
    updateModelMatrix();
}
void ViewerController::saveSettings() {
    saveSettings(ViewerSettingsStore::defaultPath());
}
void ViewerController::saveSettings(const std::filesystem::path& path) {
    settings_.cameraMode = cameraController_.mode();
    settings_.cameraFov = cameraController_.fovDegrees();
    settings_.cameraDamping = cameraController_.dampingSeconds();
    settings_.orbitSensitivity = cameraController_.orbitSensitivity();
    settings_.panSensitivity = cameraController_.panSensitivity();
    settings_.zoomSensitivity = cameraController_.zoomSensitivity();
    settings_.flySpeed = cameraController_.flySpeed();
    ViewerSettingsStore::save(path, settings_);
}
void ViewerController::setExternalModel(const GaussianModel* model) { ownedModel_.reset(); model_=model; updateModelMatrix(); }
bool ViewerController::loadModel(const std::string& path) { auto model=std::make_unique<GaussianModel>(); if(!model->loadFromFile(path)) return false; ownedModel_=std::move(model); model_=ownedModel_.get(); updateModelMatrix(); return true; }
void ViewerController::setFlipY(bool value) { settings_.flipY=value; updateModelMatrix(); }
void ViewerController::setFlipZ(bool value) { settings_.flipZ=value; updateModelMatrix(); }
void ViewerController::setCamera(const Camera& camera) { camera_=camera; hasCamera_=true; syncCameraToModelFocus(); }
void ViewerController::syncCameraToModelFocus() {
    const glm::vec3 focal = model_ ? model_->get_focus_center() : camera_.get_position()+camera_.get_front()*5.0f;
    const float radius = model_ ? model_->get_focus_radius() : 2.0f;
    cameraController_.syncFromCamera(camera_, focal, radius);
}
void ViewerController::resetCamera(float width, float height) {
    if (!model_ || model_->isEmpty()) cameraController_.reset(glm::vec3(0.0f), 1.0f, width, height);
    else cameraController_.reset(model_->get_focus_center(), model_->get_focus_radius(), width, height);
    hasCamera_=true;
}
void ViewerController::focusCamera(float width, float height) {
    if (model_ && !model_->isEmpty()) {
        cameraController_.focus(model_->get_focus_center(), model_->get_focus_radius(), width, height);
        hasCamera_=true;
    }
}
void ViewerController::updateCamera(float deltaTime, int width, int height) {
    cameraController_.update(deltaTime, camera_);
    if (width <= 0 || height <= 0) return;
    nearPlane_=0.1f; farPlane_=100.0f;
    if (model_ && !model_->isEmpty()) { const auto clip=cameraController_.fitClippingPlanes(camera_,model_->get_clip_center(),model_->get_clip_radius()); nearPlane_=clip.first; farPlane_=clip.second; }
    const float aspect=static_cast<float>(width)/static_cast<float>(height);
    float vertical=camera_.get_fov();
    if(aspect>1.0f) vertical=glm::degrees(2.0f*std::atan(std::tan(glm::radians(camera_.get_fov())*0.5f)/aspect));
    viewMatrix_=camera_.get_view_matrix(); projectionMatrix_=camera_.get_projection_matrix(aspect,vertical,nearPlane_,farPlane_);
}
void ViewerController::updateInput(const ViewerInputState& in, float dt) {
    const int button=in.panButton?1:(in.orbitButton?0:-1);
    if(in.captureMouse||button<0) resetDrag();
    else { float dx=0,dy=0; if(updateDrag(in.mouseX,in.mouseY,button,dx,dy)) { if(button==1) pan(dx,dy,(float)in.framebufferWidth,(float)in.framebufferHeight); else if(cameraController_.mode()==CameraControlMode::Fly) look(dx,dy); else orbit(dx,dy); } }
    if(!in.captureKeyboard&&toggleKeyPressed(in.toggleMode)) toggleCameraMode();
    if(!in.captureKeyboard&&focusKeyPressed(in.focus)) { if(in.resetFocus) resetCamera((float)in.framebufferWidth,(float)in.framebufferHeight); else focusCamera((float)in.framebufferWidth,(float)in.framebufferHeight); }
    if(!in.captureKeyboard&&cameraController_.mode()==CameraControlMode::Fly) { glm::vec3 m(0); if(in.moveRight)m.x+=1; if(in.moveLeft)m.x-=1; if(in.moveUp)m.y+=1; if(in.moveDown)m.y-=1; if(in.moveForward)m.z+=1; if(in.moveBackward)m.z-=1; fly(m,dt,in.speedMultiplier); }
}
void ViewerController::updateModelMatrix() { modelMatrix_=glm::mat4(1.0f); if(!model_||model_->isEmpty()) return; const glm::vec3 c=model_->get_center(); modelMatrix_=glm::translate(glm::mat4(1.0f),c)*glm::scale(glm::mat4(1.0f),glm::vec3(1.0f,settings_.flipY?-1.0f:1.0f,settings_.flipZ?-1.0f:1.0f))*glm::translate(glm::mat4(1.0f),-c); }
void ViewerController::setCameraMatrices(const glm::mat4& view, const glm::mat4& projection) { viewMatrix_=view; projectionMatrix_=projection; }
ViewerRenderData ViewerController::renderData() const { return {model_,viewMatrix_,projectionMatrix_,modelMatrix_,camera_}; }
} // namespace vulkan3DGS
