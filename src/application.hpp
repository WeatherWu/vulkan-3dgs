#pragma once

#include <memory>
#include <string>
#include <vector>
#include <array>
#include <filesystem>
#include <glm/glm.hpp>

#include "gaussian_training/gaussian_training.hpp"
#include "utils/camera.hpp"
#include "vulkan/swapchain.hpp"

namespace vk_gs {

class Window;
class Context;
class Renderer;
class GaussianModel;

// 渲染模式枚举
enum class RenderMode {
    GaussianGraphics,  // 高斯图形渲染
    GaussianCompute,   // 高斯计算渲染
    StopthePoping
    // 未来可以添加其他渲染模式
    // RayTracing,      // 光线追踪
    // ForwardRendering,// 前向渲染
};

class Application {
public:
    Application(const std::string& title, int width, int height, RenderMode mode = RenderMode::GaussianGraphics);
    virtual ~Application();
    
    void run();
    
    // 单帧更新（方便外部自由控制渲染循环）
    void tick();
    
    // 运行时切换渲染模式
    void switchRenderMode(RenderMode mode);
    RenderMode getCurrentRenderMode() const { return current_mode_; }
    
    // 设置模型数据（由外部调用）
    void setModel(const GaussianModel* model);
    bool loadModelFromFile(const std::string& filename);
    
    // 设置相机参数（由外部调用）
    void setCamera(const glm::mat4& view, const glm::mat4& projection);

    void setTrueCamera(const vk_gs::Camera& camera);
    
protected:
    virtual void initialize();
    virtual void update(float delta_time);
    virtual void render();
    virtual void cleanup();
    
protected:
    std::unique_ptr<Window> window_;
    Context* vulkan_context_ = nullptr;  // 使用指针指向单例
    std::unique_ptr<Renderer> renderer_;
    
    // 渲染数据
    const GaussianModel* current_model_ = nullptr;
    glm::mat4 view_matrix_ = glm::mat4(1.0f);
    glm::mat4 projection_matrix_ = glm::mat4(1.0f);

    vk_gs::Camera camera_;
    bool has_true_camera_ = false;

    RenderMode current_mode_;
    bool running_ = true;
    
private:
    // 工厂方法：根据渲染模式创建对应的渲染器实例
    std::unique_ptr<Renderer> createRenderer(RenderMode mode);
    void drawImGuiControls();
    void resetOrbitFromModel();
    void updateOrbitCamera(float delta_time);
    void updateOrbitInput(float delta_time);
    void syncOrbitAnglesFromOffset();
    void rebuildOrbitOffsetFromAngles();
    void handleScroll(double xoffset, double yoffset);
    void handleDroppedFiles(const std::vector<std::string>& paths);
    void updateModelMatrix();
    void drawTrainingControls();
    void validateTrainingDatasetFromUi();
    void loadTrainingDatasetFromUi();
    void initializeTrainingIfNeeded();
    void runTrainingStepFromUi();
    void exportTrainingModelFromUi();
    void exportTrainingModelToPath(const std::filesystem::path& path);
    void chooseTrainingDatasetFolderFromUi();
    void chooseTrainingOutputFolderFromUi();
    void saveTrainingPlyAsFromUi();
    void drawTrainingFileDialogs();
    void setTrainingStatus(const std::string& message);
    void setTrainingError(const std::string& message);
    std::filesystem::path outputPlyPath() const;
    void syncDefaultOutputNameFromDataset();

    std::unique_ptr<GaussianModel> owned_model_;
    std::string owned_model_path_;
    bool flip_model_y_ = false;
    bool flip_model_z_ = false;
    glm::mat4 model_matrix_ = glm::mat4(1.0f);

    bool orbit_camera_enabled_ = true;
    float orbit_mouse_sensitivity_ = 0.005f;
    float orbit_zoom_sensitivity_ = 0.12f;
    float orbit_radius_ = 5.0f;
    float orbit_angle_ = 0.0f;
    float orbit_pitch_ = 0.0f;
    glm::vec3 orbit_center_ = glm::vec3(0.0f);
    glm::vec3 orbit_offset_ = glm::vec3(5.0f, 0.0f, 0.0f);
    glm::vec3 orbit_up_ = glm::vec3(0.0f, 1.0f, 0.0f);

    bool orbit_dragging_ = false;
    double last_mouse_x_ = 0.0;
    double last_mouse_y_ = 0.0;

    bool has_last_tick_time_ = false;
    double last_tick_time_ = 0.0;
    PresentModePreference present_mode_preference_ = PresentModePreference::MaxFps;
    bool present_mode_dirty_ = false;

    GaussianTraining training_;
    bool training_initialized_ = false;
    bool training_dataset_valid_ = false;
    bool training_dataset_loaded_ = false;
    bool training_running_ = false;
    bool training_error_popup_pending_ = false;
    int training_downscale_ = 4;
    uint32_t training_frame_count_ = 0;
    uint32_t training_width_ = 0;
    uint32_t training_height_ = 0;
    uint32_t training_steps_per_frame_ = 1;
    uint64_t training_steps_done_ = 0;
    std::array<char, 512> training_dataset_path_{};
    std::array<char, 512> training_output_dir_{};
    std::array<char, 256> training_output_name_{};
    std::string training_status_;
    std::string training_error_;
};

} // namespace vk_gs
