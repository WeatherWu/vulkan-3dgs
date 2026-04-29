#include "application.hpp"
#include "gs/gaussian_model.hpp"
#include "gs/gaussian_renderer.hpp"
#include "utils/camera.hpp"
#include "utils/logger.hpp"
#include "utils/path_utils.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace vk_gs {

class DemoApplication : public Application {
public:
    DemoApplication() : Application("Vulkan 3DGS Demo", 1200, 800) {}
    
protected:
    void initialize() override {
        Application::initialize();
        
        LOG_INFO("Initializing demo application");
        
        // 创建相机
        camera_ = std::make_unique<Camera>();
        camera_->set_position(glm::vec3(0.0f, 0.0f, 3.0f));
        
        // 创建高斯散射模型
        gaussian_model_ = std::make_unique<GaussianModel>();
        
        // 加载示例数据（使用相对路径）
        std::string modelPath = PathUtils::resolvePath("data/example.ply");
        LOG_INFO("Attempting to load model from: {}", modelPath);
        
        if (PathUtils::fileExists(modelPath)) {
            if (!gaussian_model_->loadFromFile(modelPath)) {
                LOG_WARN("Failed to load {}, creating test data", modelPath);
                create_test_data();
            }
        } else {
            LOG_WARN("Model file not found: {}, creating test data", modelPath);
            create_test_data();
        }
        
        // 创建渲染器
        renderer_ = std::make_unique<GaussianRenderer>(vulkan_context_->get_device());
        renderer_->initialize();
        renderer_->set_resolution(window_->get_width(), window_->get_height());
        
        // 设置窗口回调
        setup_callbacks();
        
        LOG_INFO("Demo application initialized successfully");
    }
    
    void update(float delta_time) override {
        // 更新相机
        camera_->update(delta_time);
        
        // 更新动画
        time_ += delta_time;
        
        // 简单的旋转动画
        if (animate_) {
            float rotation_speed = 0.5f;
            camera_->set_yaw(camera_->get_yaw() + rotation_speed * delta_time);
        }
    }
    
    void render() override {
        // 获取视图和投影矩阵
        glm::mat4 view_matrix = camera_->get_view_matrix();
        glm::mat4 projection_matrix = camera_->get_projection_matrix(
            static_cast<float>(window_->get_width()) / window_->get_height()
        );
        
        // 渲染高斯散射
        renderer_->render(*gaussian_model_, view_matrix, projection_matrix);
    }
    
    void cleanup() override {
        LOG_INFO("Cleaning up demo application");
        
        if (renderer_) {
            renderer_->cleanup();
        }
        
        Application::cleanup();
    }
    
private:
    void setup_callbacks() {
        window_->set_key_callback([this](int key, int action) {
            if (action == GLFW_PRESS) {
                handle_key_press(key);
            }
        });
        
        window_->set_mouse_callback([this](double x, double y) {
            handle_mouse_move(x, y);
        });
        
        window_->set_resize_callback([this](int width, int height) {
            handle_resize(width, height);
        });
    }
    
    void handle_key_press(int key) {
        switch (key) {
            case GLFW_KEY_ESCAPE:
                running_ = false;
                break;
            case GLFW_KEY_SPACE:
                animate_ = !animate_;
                LOG_INFO("Animation {}", animate_ ? "enabled" : "disabled");
                break;
            case GLFW_KEY_W:
                camera_->move_forward(0.1f);
                break;
            case GLFW_KEY_S:
                camera_->move_backward(0.1f);
                break;
            case GLFW_KEY_A:
                camera_->move_left(0.1f);
                break;
            case GLFW_KEY_D:
                camera_->move_right(0.1f);
                break;
        }
    }
    
    void handle_mouse_move(double x, double y) {
        static double last_x = x, last_y = y;
        
        double x_offset = x - last_x;
        double y_offset = last_y - y; // 反转Y轴
        
        last_x = x;
        last_y = y;
        
        // 只在鼠标按下时旋转相机
        if (glfwGetMouseButton(window_->get_handle(), GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
            float sensitivity = 0.1f;
            camera_->add_yaw(static_cast<float>(x_offset) * sensitivity);
            camera_->add_pitch(static_cast<float>(y_offset) * sensitivity);
        }
    }
    
    void handle_resize(int width, int height) {
        if (renderer_) {
            renderer_->set_resolution(width, height);
        }
    }
    
    void create_test_data() {
        // 创建一些测试高斯点
        for (int i = 0; i < 1000; ++i) {
            GaussianPoint point;
            
            // 在球面上随机分布
            float theta = static_cast<float>(i) * 0.1f;
            float phi = static_cast<float>(i) * 0.05f;
            float radius = 2.0f;
            
            point.position = glm::vec3(
                radius * sin(theta) * cos(phi),
                radius * cos(theta),
                radius * sin(theta) * sin(phi)
            );
            
            // 基于位置的颜色（存储为SH DC系数）
            float r = (sin(theta) + 1.0f) * 0.5f;
            float g = (cos(phi) + 1.0f) * 0.5f;
            float b = (sin(theta + phi) + 1.0f) * 0.5f;
            point.color.sh0 = glm::vec3(r, g, b);
            
            point.covariance = glm::mat3(0.1f);
            point.alpha = 0.7f + 0.3f * sin(theta);
            
            point.scale = glm::vec3(0.05f + 0.03f * sin(theta));
            point.rotation = glm::quat(cos(theta * 0.5f), 0.0f, sin(theta * 0.5f), 0.0f);
            
            gaussian_model_->addPoint(point);
        }
        
        LOG_INFO("Created {} test Gaussian points", gaussian_model_->get_statistics().total_points);
    }
    
    std::unique_ptr<Camera> camera_;
    std::unique_ptr<GaussianModel> gaussian_model_;
    std::unique_ptr<GaussianRenderer> renderer_;
    
    float time_ = 0.0f;
    bool animate_ = true;
};

} // namespace vk_gs

int main() {
    try {
        vk_gs::DemoApplication app;
        app.run();
    } catch (const std::exception& e) {
        LOG_ERROR("Application failed: {}", e.what());
        return -1;
    }
    
    return 0;
}