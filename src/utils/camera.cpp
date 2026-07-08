#include "camera.hpp"
#include "utils/logger.hpp"

#include <algorithm>
#include <cmath>

namespace vulkan3DGS {

Camera::Camera() {
    update_vectors();
}

void Camera::update(float delta_time) {
    // 可以在这里添加平滑移动或动画逻辑
}

void Camera::set_target(const glm::vec3& target) {
    look_at(target, world_up_);
}

void Camera::look_at(const glm::vec3& target, const glm::vec3& up_hint) {
    glm::vec3 direction = target - position_;
    if (glm::length(direction) <= 1e-6f) {
        return;
    }

    set_orientation_from_basis(glm::normalize(direction), up_hint);
    world_up_ = up_;
}

void Camera::set_up(const glm::vec3& up) {
    world_up_ = glm::normalize(up);
    set_orientation_from_basis(front_, world_up_);
}

void Camera::set_yaw(float yaw) {
    yaw_ = yaw;
    update_vectors();
}

void Camera::set_pitch(float pitch) {
    pitch_ = pitch;
    update_vectors();
}

void Camera::add_yaw(float delta) {
    set_yaw(yaw_ + delta);
}

void Camera::add_pitch(float delta) {
    set_pitch(pitch_ + delta);
}

void Camera::move_forward(float distance) {
    position_ += front_ * distance * movement_speed_;
}

void Camera::move_backward(float distance) {
    position_ -= front_ * distance * movement_speed_;
}

void Camera::move_left(float distance) {
    position_ -= right_ * distance * movement_speed_;
}

void Camera::move_right(float distance) {
    position_ += right_ * distance * movement_speed_;
}

void Camera::move_up(float distance) {
    position_ += world_up_ * distance * movement_speed_;
}

void Camera::move_down(float distance) {
    position_ -= world_up_ * distance * movement_speed_;
}

glm::mat4 Camera::get_view_matrix() const {
    return glm::lookAt(position_, position_ + front_, up_);
}

glm::mat4 Camera::get_projection_matrix(float aspect_ratio, float fov, 
                                       float near_plane, float far_plane) {
    fov_ = fov;
    glm::mat4 projection = glm::perspective(glm::radians(fov_), aspect_ratio, near_plane, far_plane);
    projection[1][1] *= -1.0f; // Vulkan framebuffer coordinates have inverted Y relative to GLM's OpenGL projection.
    return projection;
}

void Camera::update_vectors() {
    // Compatibility path for callers that still use yaw/pitch.
    if (pitch_ > 89.0f) pitch_ = 89.0f;
    if (pitch_ < -89.0f) pitch_ = -89.0f;
    
    // 计算新的前向量
    glm::vec3 new_front;
    new_front.x = cos(glm::radians(yaw_)) * cos(glm::radians(pitch_));
    new_front.y = sin(glm::radians(pitch_));
    new_front.z = sin(glm::radians(yaw_)) * cos(glm::radians(pitch_));
    set_orientation_from_basis(glm::normalize(new_front), world_up_);
}

void Camera::set_orientation_from_basis(const glm::vec3& front, const glm::vec3& up_hint) {
    glm::vec3 normalizedFront = glm::length(front) > 1e-6f ? glm::normalize(front) : glm::vec3(0.0f, 0.0f, -1.0f);
    glm::vec3 upHint = glm::length(up_hint) > 1e-6f ? glm::normalize(up_hint) : glm::vec3(0.0f, 1.0f, 0.0f);

    glm::vec3 right = glm::cross(normalizedFront, upHint);
    if (glm::length(right) <= 1e-6f) {
        glm::vec3 fallbackUp = std::abs(normalizedFront.y) < 0.9f ? glm::vec3(0.0f, 1.0f, 0.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
        right = glm::cross(normalizedFront, fallbackUp);
    }
    right = glm::normalize(right);
    glm::vec3 up = glm::normalize(glm::cross(right, normalizedFront));

    glm::mat3 basis(right, up, -normalizedFront);
    orientation_ = glm::normalize(glm::quat_cast(basis));
    sync_vectors_from_orientation();
    sync_euler_from_front();
}

void Camera::sync_vectors_from_orientation() {
    orientation_ = glm::normalize(orientation_);
    front_ = glm::normalize(orientation_ * glm::vec3(0.0f, 0.0f, -1.0f));
    right_ = glm::normalize(orientation_ * glm::vec3(1.0f, 0.0f, 0.0f));
    up_ = glm::normalize(orientation_ * glm::vec3(0.0f, 1.0f, 0.0f));
}

void Camera::sync_euler_from_front() {
    yaw_ = glm::degrees(std::atan2(front_.z, front_.x));
    pitch_ = glm::degrees(std::asin(std::clamp(front_.y, -1.0f, 1.0f)));
}

} // namespace vulkan3DGS
