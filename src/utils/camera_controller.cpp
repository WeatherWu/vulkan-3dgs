#include "camera_controller.hpp"

#include <algorithm>
#include <cmath>

#include <glm/gtc/constants.hpp>

namespace vulkan3DGS {

namespace {

constexpr float kMinDistance = 0.0001f;
constexpr float kMaxElevation = glm::radians(89.0f);
constexpr float kMinClip = 0.00001f;

} // namespace

void CameraController::reset(const glm::vec3& focalPoint,
                             float sceneRadius,
                             float viewportWidth,
                             float viewportHeight,
                             float azimuthDegrees,
                             float elevationDegrees) {
    sceneRadius_ = std::max(sceneRadius, 0.001f);
    targetFocalPoint_ = focalPoint;
    focalPoint_ = focalPoint;
    targetAzimuth_ = glm::radians(azimuthDegrees);
    azimuth_ = targetAzimuth_;
    targetElevation_ = glm::radians(elevationDegrees);
    elevation_ = targetElevation_;
    targetDistance_ = fitDistanceForBounds(
        sceneRadius_, viewportWidth, viewportHeight);
    distance_ = targetDistance_;
    mode_ = CameraControlMode::Orbit;
    clampTargets();
}

void CameraController::focus(const glm::vec3& focalPoint,
                             float radius,
                             float viewportWidth,
                             float viewportHeight) {
    sceneRadius_ = std::max(radius, 0.001f);
    targetFocalPoint_ = focalPoint;
    targetDistance_ = fitDistanceForBounds(
        sceneRadius_, viewportWidth, viewportHeight);
    mode_ = CameraControlMode::Orbit;
    clampTargets();
}

void CameraController::syncFromCamera(const Camera& camera,
                                      const glm::vec3& focalPoint,
                                      float sceneRadius) {
    sceneRadius_ = std::max(sceneRadius, 0.001f);
    glm::vec3 offset = camera.get_position() - focalPoint;
    float distance = glm::length(offset);
    if (distance <= kMinDistance) {
        reset(focalPoint, sceneRadius_);
        return;
    }

    offset /= distance;
    targetFocalPoint_ = focalPoint;
    focalPoint_ = focalPoint;
    targetDistance_ = distance;
    distance_ = distance;
    targetAzimuth_ = std::atan2(offset.x, offset.z);
    azimuth_ = targetAzimuth_;
    targetElevation_ = -std::asin(std::clamp(offset.y, -1.0f, 1.0f));
    elevation_ = targetElevation_;
    fovDegrees_ = camera.get_fov();
    clampTargets();
}

void CameraController::update(float deltaTime, Camera& camera) {
    const float amount = smoothingAmount(deltaTime);
    focalPoint_ = glm::mix(focalPoint_, targetFocalPoint_, amount);
    azimuth_ = approachAngle(azimuth_, targetAzimuth_, amount);
    elevation_ = glm::mix(elevation_, targetElevation_, amount);
    distance_ = glm::mix(distance_, targetDistance_, amount);

    const glm::vec3 forward = forwardFromAngles(azimuth_, elevation_);
    camera.set_position(focalPoint_ - forward * distance_);
    camera.look_at(focalPoint_, glm::vec3(0.0f, 1.0f, 0.0f));
    camera.set_fov(fovDegrees_);
}

void CameraController::orbit(float deltaX, float deltaY) {
    targetAzimuth_ -= deltaX * orbitSensitivity_;
    targetElevation_ -= deltaY * orbitSensitivity_;
    clampTargets();
}

void CameraController::look(float deltaX, float deltaY) {
    const glm::vec3 oldForward = forwardFromAngles(targetAzimuth_, targetElevation_);
    const glm::vec3 cameraPosition = targetFocalPoint_ - oldForward * targetDistance_;

    targetAzimuth_ -= deltaX * orbitSensitivity_;
    targetElevation_ -= deltaY * orbitSensitivity_;
    clampTargets();

    const glm::vec3 newForward = forwardFromAngles(targetAzimuth_, targetElevation_);
    targetFocalPoint_ = cameraPosition + newForward * targetDistance_;
}

void CameraController::pan(float deltaX, float deltaY,
                           float viewportWidth, float viewportHeight) {
    const glm::vec3 forward = forwardFromAngles(targetAzimuth_, targetElevation_);
    glm::vec3 right = glm::cross(forward, glm::vec3(0.0f, 1.0f, 0.0f));
    if (glm::length(right) <= 1e-6f) {
        right = glm::vec3(1.0f, 0.0f, 0.0f);
    } else {
        right = glm::normalize(right);
    }
    const glm::vec3 up = glm::normalize(glm::cross(right, forward));
    const float safeWidth = std::max(viewportWidth, 1.0f);
    const float safeHeight = std::max(viewportHeight, 1.0f);
    const float aspect = safeWidth / safeHeight;
    const float verticalFov = aspect > 1.0f
        ? 2.0f * std::atan(std::tan(glm::radians(fovDegrees_) * 0.5f) / aspect)
        : glm::radians(fovDegrees_);
    const float worldPerPixel =
        2.0f * targetDistance_ * std::tan(verticalFov * 0.5f) /
        safeHeight;
    targetFocalPoint_ +=
        (-right * deltaX + up * deltaY) * worldPerPixel * panSensitivity_;
}

void CameraController::dolly(float wheelDelta) {
    targetDistance_ *= std::exp(-wheelDelta * zoomSensitivity_);
    clampTargets();
}

void CameraController::fly(const glm::vec3& localMotion,
                           float deltaTime,
                           float speedMultiplier) {
    if (glm::dot(localMotion, localMotion) <= 1e-8f || deltaTime <= 0.0f) {
        return;
    }

    const glm::vec3 forward = forwardFromAngles(targetAzimuth_, targetElevation_);
    glm::vec3 right = glm::cross(forward, glm::vec3(0.0f, 1.0f, 0.0f));
    if (glm::length(right) <= 1e-6f) {
        right = glm::vec3(1.0f, 0.0f, 0.0f);
    } else {
        right = glm::normalize(right);
    }

    glm::vec3 motion = right * localMotion.x +
                       glm::vec3(0.0f, 1.0f, 0.0f) * localMotion.y +
                       forward * localMotion.z;
    if (glm::length(motion) > 1.0f) {
        motion = glm::normalize(motion);
    }

    const float worldSpeed = flySpeed_ * std::max(sceneRadius_, 0.01f);
    targetFocalPoint_ += motion * worldSpeed * speedMultiplier * deltaTime;
}

std::pair<float, float> CameraController::fitClippingPlanes(
    const Camera& camera,
    const glm::vec3& boundsCenter,
    float boundsRadius) const {
    // GaussianModel currently bounds centers, not full ellipsoid extents.
    // Keep a conservative margin until the resource bound includes scale.
    const float radius = std::max(boundsRadius * 1.25f, 0.01f);
    const float distanceAlongView =
        glm::dot(boundsCenter - camera.get_position(), camera.get_front());
    float farPlane = std::max(distanceAlongView + radius, radius * 2.0f);
    farPlane = std::max(farPlane, 0.01f);
    float nearPlane = std::max(distanceAlongView - radius,
                               farPlane / (1024.0f * 16.0f));
    nearPlane = std::min(nearPlane, 1.0f);
    nearPlane = std::clamp(nearPlane, kMinClip, farPlane * 0.5f);
    return {nearPlane, farPlane};
}

float CameraController::fitDistanceForBounds(float radius,
                                             float viewportWidth,
                                             float viewportHeight,
                                             float padding) const {
    const float safeRadius = std::max(radius, 0.001f);
    const float safeWidth = std::max(viewportWidth, 1.0f);
    const float safeHeight = std::max(viewportHeight, 1.0f);
    const float aspect = safeWidth / safeHeight;
    const float largerAxisFov = glm::radians(fovDegrees_);

    float horizontalFov = largerAxisFov;
    float verticalFov = largerAxisFov;
    if (aspect > 1.0f) {
        verticalFov = 2.0f * std::atan(
            std::tan(largerAxisFov * 0.5f) / aspect);
    } else if (aspect < 1.0f) {
        horizontalFov = 2.0f * std::atan(
            std::tan(largerAxisFov * 0.5f) * aspect);
    }

    const float limitingHalfFov =
        std::max(0.5f * std::min(horizontalFov, verticalFov),
                 glm::radians(0.5f));
    return safeRadius /
        std::max(std::sin(limitingHalfFov), 1e-4f) *
        std::max(padding, 1.0f);
}

void CameraController::toggleMode() {
    mode_ = mode_ == CameraControlMode::Orbit
        ? CameraControlMode::Fly
        : CameraControlMode::Orbit;
}

float CameraController::azimuthDegrees() const {
    return glm::degrees(targetAzimuth_);
}

float CameraController::elevationDegrees() const {
    return glm::degrees(targetElevation_);
}

void CameraController::setDistance(float distance) {
    targetDistance_ = distance;
    clampTargets();
}

void CameraController::setAzimuthDegrees(float degrees) {
    targetAzimuth_ = glm::radians(degrees);
}

void CameraController::setElevationDegrees(float degrees) {
    targetElevation_ = glm::radians(degrees);
    clampTargets();
}

void CameraController::setFovDegrees(float degrees) {
    fovDegrees_ = std::clamp(degrees, 10.0f, 120.0f);
}

void CameraController::setDampingSeconds(float seconds) {
    dampingSeconds_ = std::clamp(seconds, 0.0f, 1.0f);
}

void CameraController::setOrbitSensitivity(float radiansPerPixel) {
    orbitSensitivity_ = std::clamp(radiansPerPixel, 0.0001f, 0.05f);
}

void CameraController::setPanSensitivity(float scale) {
    panSensitivity_ = std::clamp(scale, 0.01f, 10.0f);
}

void CameraController::setZoomSensitivity(float scale) {
    zoomSensitivity_ = std::clamp(scale, 0.01f, 1.0f);
}

void CameraController::setFlySpeed(float scale) {
    flySpeed_ = std::clamp(scale, 0.01f, 30.0f);
}

glm::vec3 CameraController::forwardFromAngles(float azimuth, float elevation) {
    const float cosElevation = std::cos(elevation);
    return glm::normalize(glm::vec3(
        -std::sin(azimuth) * cosElevation,
        std::sin(elevation),
        -std::cos(azimuth) * cosElevation));
}

float CameraController::approachAngle(float current, float target, float amount) {
    const float delta = std::remainder(target - current, glm::two_pi<float>());
    return current + delta * amount;
}

float CameraController::smoothingAmount(float deltaTime) const {
    if (dampingSeconds_ <= 0.0f || deltaTime <= 0.0f) {
        return dampingSeconds_ <= 0.0f ? 1.0f : 0.0f;
    }
    return 1.0f - std::exp(-deltaTime / dampingSeconds_);
}

void CameraController::clampTargets() {
    targetElevation_ = std::clamp(targetElevation_, -kMaxElevation, kMaxElevation);
    const float minimumDistance = std::max(sceneRadius_ * 1e-5f, kMinDistance);
    const float maximumDistance = std::max(sceneRadius_ * 100.0f, 100.0f);
    targetDistance_ = std::clamp(targetDistance_, minimumDistance, maximumDistance);
}

} // namespace vulkan3DGS
