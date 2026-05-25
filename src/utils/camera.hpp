#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

namespace vk_gs {

class Camera {
public:
    Camera();
    
    void update(float delta_time);
    
    // 位置和方向设置
    void set_position(const glm::vec3& position) { position_ = position; }
    void set_target(const glm::vec3& target);
    void look_at(const glm::vec3& target, const glm::vec3& up_hint);
    void set_up(const glm::vec3& up);
    
    // 欧拉角设置
    void set_yaw(float yaw);
    void set_pitch(float pitch);
    void add_yaw(float delta);
    void add_pitch(float delta);
    
    // 移动控制
    void move_forward(float distance);
    void move_backward(float distance);
    void move_left(float distance);
    void move_right(float distance);
    void move_up(float distance);
    void move_down(float distance);
    
    // 获取矩阵
    glm::mat4 get_view_matrix() const;
    glm::mat4 get_projection_matrix(float aspect_ratio, float fov, 
                                   float near_plane = 0.1f, float far_plane = 100.0f);
    
    // 获取属性
    const glm::vec3& get_position() const { return position_; }
    const glm::vec3& get_front() const { return front_; }
    const glm::vec3& get_right() const { return right_; }
    const glm::vec3& get_up() const { return up_; }
    float get_yaw() const { return yaw_; }
    float get_pitch() const { return pitch_; }
    float get_fov() const { return fov_; }
    
private:
    void update_vectors();
    void set_orientation_from_basis(const glm::vec3& front, const glm::vec3& up_hint);
    void sync_vectors_from_orientation();
    void sync_euler_from_front();
    
    glm::vec3 position_ = glm::vec3(0.0f, 0.0f, 3.0f);
    glm::quat orientation_ = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    glm::vec3 front_ = glm::vec3(0.0f, 0.0f, -1.0f);
    glm::vec3 up_ = glm::vec3(0.0f, 1.0f, 0.0f);
    glm::vec3 right_ = glm::vec3(1.0f, 0.0f, 0.0f);
    glm::vec3 world_up_ = glm::vec3(0.0f, 1.0f, 0.0f);
    
    float yaw_ = -90.0f;   // 偏航角
    float pitch_ = 0.0f;   // 俯仰角
    
    float movement_speed_ = 2.5f;
    float mouse_sensitivity_ = 0.1f;

    float fov_ = 45.0f;
};

} // namespace vk_gs
