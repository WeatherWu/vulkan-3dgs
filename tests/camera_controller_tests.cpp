#include <algorithm>
#include <cmath>
#include <iostream>

#include "viewer/camera_controller.hpp"

namespace {

bool near(float lhs, float rhs, float tolerance = 1e-4f) {
    return std::abs(lhs - rhs) <= tolerance *
        std::max({1.0f, std::abs(lhs), std::abs(rhs)});
}

bool testOrbitHasNoRoll() {
    vulkan3DGS::Camera camera;
    vulkan3DGS::CameraController controller;
    controller.setDampingSeconds(0.0f);
    controller.reset(glm::vec3(1.0f, 2.0f, 3.0f), 2.0f);
    controller.orbit(250.0f, -120.0f);
    controller.update(1.0f / 60.0f, camera);

    return std::abs(camera.get_right().y) < 1e-4f &&
           glm::dot(camera.get_up(), glm::vec3(0.0f, 1.0f, 0.0f)) > 0.0f &&
           near(glm::length(camera.get_position() - controller.focalPoint()),
                controller.distance());
}

bool testDollyAndFly() {
    vulkan3DGS::Camera camera;
    vulkan3DGS::CameraController controller;
    controller.setDampingSeconds(0.0f);
    controller.reset(glm::vec3(0.0f), 2.0f);
    const float previousDistance = controller.distance();
    controller.dolly(1.0f);
    if (!(controller.distance() < previousDistance)) {
        return false;
    }

    controller.setMode(vulkan3DGS::CameraControlMode::Fly);
    controller.update(1.0f / 60.0f, camera);
    const glm::vec3 previousPosition = camera.get_position();
    controller.fly(glm::vec3(0.0f, 0.0f, 1.0f), 1.0f);
    controller.update(1.0f / 60.0f, camera);
    return glm::length(camera.get_position() - previousPosition) > 0.1f;
}

bool testDynamicClipping() {
    vulkan3DGS::Camera camera;
    vulkan3DGS::CameraController controller;
    controller.setDampingSeconds(0.0f);
    controller.reset(glm::vec3(0.0f), 4.0f);
    controller.update(1.0f / 60.0f, camera);
    const auto clipping = controller.fitClippingPlanes(
        camera, glm::vec3(0.0f), 4.0f);
    return clipping.first > 0.0f && clipping.second > clipping.first;
}

bool testAspectAwareFramingAndModeReset() {
    vulkan3DGS::CameraController controller;
    controller.setFovDegrees(75.0f);
    const float widescreenDistance = controller.fitDistanceForBounds(
        1.0f, 1920.0f, 1080.0f);
    const float ultrawideDistance = controller.fitDistanceForBounds(
        1.0f, 2520.0f, 1080.0f);
    if (!(widescreenDistance > 2.5f) ||
        !(ultrawideDistance > widescreenDistance)) {
        return false;
    }

    controller.setMode(vulkan3DGS::CameraControlMode::Fly);
    controller.reset(glm::vec3(0.0f), 1.0f, 1920.0f, 1080.0f);
    if (controller.mode() != vulkan3DGS::CameraControlMode::Orbit ||
        !near(controller.distance(), widescreenDistance)) {
        return false;
    }

    controller.dolly(2.0f);
    controller.focus(glm::vec3(0.0f), 1.0f, 1920.0f, 1080.0f);
    return near(controller.distance(), widescreenDistance);
}

} // namespace

int main() {
    if (!testOrbitHasNoRoll()) {
        std::cerr << "Orbit camera accumulated roll or lost its distance\n";
        return 1;
    }
    if (!testDollyAndFly()) {
        std::cerr << "Dolly or fly camera motion failed\n";
        return 1;
    }
    if (!testDynamicClipping()) {
        std::cerr << "Dynamic clipping planes are invalid\n";
        return 1;
    }
    if (!testAspectAwareFramingAndModeReset()) {
        std::cerr << "Aspect-aware framing or mode reset failed\n";
        return 1;
    }
    return 0;
}
