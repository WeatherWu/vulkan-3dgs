#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <array>
#include <filesystem>
#include <glm/glm.hpp>

#include "gaussian_training/gaussian_training.hpp"
#include "utils/camera.hpp"
#include "vulkan/swapchain.hpp"

namespace vulkan3DGS {

class Window;
class Context;
class Device;
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
    Application(const std::string& title,
                int width,
                int height,
                RenderMode mode = RenderMode::GaussianGraphics,
                std::optional<std::string> gpuSelector = std::nullopt);
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

    void setTrueCamera(const vulkan3DGS::Camera& camera);
    
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

    vulkan3DGS::Camera camera_;
    bool has_true_camera_ = false;

    RenderMode current_mode_;
    bool running_ = true;
    
private:
    // 工厂方法：根据渲染模式创建对应的渲染器实例
    std::unique_ptr<Renderer> createRenderer(RenderMode mode);
    void drawImGuiControls();
    void drawTrainingGpuControl();
    void applyPendingTrainingGpuSelection();
    void syncTrainingGpuSelection();
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
    void applyTrainingConfigFromUi();
    void applyTrainingDensificationConfigFromUi();
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
    std::unique_ptr<Device> training_device_;
    size_t training_gpu_ui_selection_ = 0;
    std::optional<std::string> pending_training_gpu_selector_;

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
    uint32_t training_initial_gaussians_ = 10000;
    uint32_t training_random_seed_ = 0;
    float training_initial_opacity_ = 0.1f;
    float training_scene_radius_scale_ = 1.0f;
    int training_mode_ = 1;
    uint32_t training_total_iterations_ = 30000;
    uint32_t training_steps_per_frame_ = 1;
    float training_position_lr_ = 0.00016f;
    float training_position_lr_final_ = 0.0000016f;
    float training_position_lr_delay_mult_ = 0.01f;
    float training_position_lr_delay_steps_ = 0.0f;
    float training_position_lr_max_steps_ = 30000.0f;
    float training_feature_lr_ = 0.0025f;
    float training_feature_rest_lr_ = 0.000125f;
    float training_opacity_lr_ = 0.025f;
    float training_scale_lr_ = 0.005f;
    float training_rotation_lr_ = 0.001f;
    float training_adam_beta1_ = 0.9f;
    float training_adam_beta2_ = 0.999f;
    float training_adam_epsilon_ = 1e-15f;
    float training_grad_clip_ = 0.0f;
    float training_loss_dssim_weight_ = 0.2f;
    uint32_t training_max_sh_degree_ = 3;
    uint32_t training_sh_degree_interval_ = 1000;
    uint64_t training_steps_done_ = 0;
    bool training_densification_enabled_ = true;
    uint32_t training_densify_from_iteration_ = 500;
    uint32_t training_densify_until_iteration_ = 15000;
    uint32_t training_densification_interval_ = 100;
    uint32_t training_opacity_reset_interval_ = 3000;
    uint32_t training_max_gaussians_ = 10000000;
    uint32_t training_split_children_ = 2;
    float training_densify_grad_threshold_ = 0.0002f;
    float training_min_opacity_ = 0.005f;
    float training_percent_dense_ = 0.01f;
    float training_screen_size_prune_threshold_ = 20.0f;
    float training_world_size_prune_threshold_ = 0.1f;
    std::array<char, 512> training_dataset_path_{};
    std::array<char, 512> training_output_dir_{};
    std::array<char, 256> training_output_name_{};
    std::string training_status_;
    std::string training_error_;
};

} // namespace vulkan3DGS
