#include "gaussian_renderer_compute/gaussian_model.hpp"
#include "application.hpp"
#include "utils/camera.hpp"
#include "utils/logger.hpp"
#include <glm/gtc/matrix_transform.hpp>

int main(){
    // 加载高斯模型
    vk_gs::GaussianModel model;
    if (!model.loadFromFile("./../../../insect.ply")) {
        LOG_ERROR("Failed to load insect.ply");
        return -1;
    }
    
    // 创建应用
    vk_gs::Application app("Vulkan Gaussian Splatting", 1280, 720, vk_gs::RenderMode::GaussianGraphics);
    
    // 设置模型
    app.setModel(&model);
    
    // 创建相机并设置初始位置
    vk_gs::Camera camera;
    glm::vec3 modelCenter = model.get_center();
    camera.set_position(modelCenter + glm::vec3(-5.0f, -5.0f, -5.0f));
    camera.set_up(glm::vec3(0.0f, -1.0f, 0.0f));
    camera.set_target(modelCenter);
    app.setCamera(camera.get_view_matrix(), camera.get_projection_matrix(1280.0f / 720.0f, 45.0f));
    app.setTrueCamera(camera);
    
    // 运行主循环
    app.run();
    
    return 0;
}
