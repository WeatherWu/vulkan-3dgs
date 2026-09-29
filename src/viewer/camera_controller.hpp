#pragma once

#include <utility>

#include <glm/glm.hpp>

#include "camera.hpp"

namespace vulkan3DGS {

enum class CameraControlMode {
    Orbit = 0,
    Fly = 1,
};

// Focus-point camera controller modelled after SuperSplat's orbit/fly camera.
// The controller always reconstructs the camera with world +Y as its up axis,
// preventing the unintended roll that a free-form trackball can accumulate.
class CameraController {
public:
    void reset(const glm::vec3& focalPoint, float sceneRadius,
               float viewportWidth = 1.0f,
               float viewportHeight = 1.0f,
               float azimuthDegrees = -45.0f,
               float elevationDegrees = -10.0f);
    void focus(const glm::vec3& focalPoint, float radius,
               float viewportWidth = 1.0f,
               float viewportHeight = 1.0f);
    void syncFromCamera(const Camera& camera, const glm::vec3& focalPoint,
                        float sceneRadius);

    void update(float deltaTime, Camera& camera);
    void orbit(float deltaX, float deltaY);
    void look(float deltaX, float deltaY);
    void pan(float deltaX, float deltaY, float viewportWidth, float viewportHeight);
    void dolly(float wheelDelta);
    void fly(const glm::vec3& localMotion, float deltaTime,
             float speedMultiplier = 1.0f);

    std::pair<float, float> fitClippingPlanes(
        const Camera& camera,
        const glm::vec3& boundsCenter,
        float boundsRadius) const;
    float fitDistanceForBounds(float radius,
                               float viewportWidth,
                               float viewportHeight,
                               float padding = 1.1f) const;

    CameraControlMode mode() const { return mode_; }
    void setMode(CameraControlMode mode) { mode_ = mode; }
    void toggleMode();

    const glm::vec3& focalPoint() const { return targetFocalPoint_; }
    float distance() const { return targetDistance_; }
    float azimuthDegrees() const;
    float elevationDegrees() const;
    float fovDegrees() const { return fovDegrees_; }
    float dampingSeconds() const { return dampingSeconds_; }
    float orbitSensitivity() const { return orbitSensitivity_; }
    float panSensitivity() const { return panSensitivity_; }
    float zoomSensitivity() const { return zoomSensitivity_; }
    float flySpeed() const { return flySpeed_; }

    void setDistance(float distance);
    void setAzimuthDegrees(float degrees);
    void setElevationDegrees(float degrees);
    void setFovDegrees(float degrees);
    void setDampingSeconds(float seconds);
    void setOrbitSensitivity(float radiansPerPixel);
    void setPanSensitivity(float scale);
    void setZoomSensitivity(float scale);
    void setFlySpeed(float scale);

private:
    static glm::vec3 forwardFromAngles(float azimuth, float elevation);
    static float approachAngle(float current, float target, float amount);
    float smoothingAmount(float deltaTime) const;
    void clampTargets();

    CameraControlMode mode_ = CameraControlMode::Orbit;
    glm::vec3 focalPoint_{0.0f};
    glm::vec3 targetFocalPoint_{0.0f};
    float azimuth_ = 0.0f;
    float targetAzimuth_ = 0.0f;
    float elevation_ = 0.0f;
    float targetElevation_ = 0.0f;
    float distance_ = 5.0f;
    float targetDistance_ = 5.0f;
    float sceneRadius_ = 1.0f;

    float fovDegrees_ = 75.0f;
    float dampingSeconds_ = 0.12f;
    float orbitSensitivity_ = 0.005f;
    float panSensitivity_ = 1.0f;
    float zoomSensitivity_ = 0.12f;
    float flySpeed_ = 1.0f;
};

} // namespace vulkan3DGS
