#include "camera.hpp"
#include "utils/logger.hpp"

namespace vk_gs {

Camera::Camera() {
    update_vectors();
}

void Camera::update(float delta_time) {
    // 可以在这里添加平滑移动或动画逻辑
}

void Camera::set_target(const glm::vec3& target) {
    glm::vec3 direction = glm::normalize(target - position_);
    
    // 计算偏航角和俯仰角
    yaw_ = glm::degrees(atan2(direction.z, direction.x));
    pitch_ = glm::degrees(asin(direction.y));
    
    update_vectors();
}

void Camera::set_up(const glm::vec3& up) {
    world_up_ = glm::normalize(up);
    update_vectors();
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
    // 限制俯仰角，避免万向节锁
    if (pitch_ > 89.0f) pitch_ = 89.0f;
    if (pitch_ < -89.0f) pitch_ = -89.0f;
    
    // 计算新的前向量
    glm::vec3 new_front;
    new_front.x = cos(glm::radians(yaw_)) * cos(glm::radians(pitch_));
    new_front.y = sin(glm::radians(pitch_));
    new_front.z = sin(glm::radians(yaw_)) * cos(glm::radians(pitch_));
    front_ = glm::normalize(new_front);
    
    // 重新计算右向量和上向量
    right_ = glm::normalize(glm::cross(front_, world_up_));
    up_ = glm::normalize(glm::cross(right_, front_));
}

} // namespace vk_gs
