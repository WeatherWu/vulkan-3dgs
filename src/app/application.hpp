#pragma once

#include "training/app/training_controller.hpp"
#include "training/ui/training_panel.hpp"
#include "viewer/viewer_controller.hpp"

#include <glm/glm.hpp>

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace vulkan3DGS {

class Window;
class Context;
class Renderer;
class GaussianModel;

enum class RenderMode {
    GaussianGraphics,
    GaussianCompute,
    StopthePoping,
};

// Application is the composition root: it owns presentation routing and
// delegates viewer and training policy to their controllers/panels.
class Application {
public:
    Application(const std::string& title,
                int width,
                int height,
                RenderMode mode = RenderMode::GaussianGraphics,
                std::optional<std::string> gpuSelector = std::nullopt);
    virtual ~Application();

    void run();
    void tick();
    void switchRenderMode(RenderMode mode);
    [[nodiscard]] RenderMode getCurrentRenderMode() const { return currentMode_; }

    void setModel(const GaussianModel* model);
    bool loadModelFromFile(const std::string& filename);
    void setCamera(const glm::mat4& view, const glm::mat4& projection);
    void setTrueCamera(const Camera& camera);

protected:
    virtual void initialize();
    virtual void update(float deltaTime);
    virtual void render();
    virtual void cleanup();

    std::unique_ptr<Window> window_;
    Context* vulkanContext_ = nullptr;
    std::unique_ptr<Renderer> renderer_;

private:
    std::unique_ptr<Renderer> createRenderer(RenderMode mode);
    void configureRenderer();
    void drawImGuiControls();
    void updateViewer(float deltaTime);
    void handleScroll(double xoffset, double yoffset);
    void handleDroppedFiles(const std::vector<std::string>& paths);

    ViewerController viewerController_;
    TrainingController trainingController_;
    TrainingPanel trainingPanel_;
    RenderMode currentMode_;
    bool running_ = true;
    bool cleanedUp_ = false;
    bool hasLastTickTime_ = false;
    double lastTickTime_ = 0.0;
    double pureLastUiRenderTime_ = 0.0;
};

} // namespace vulkan3DGS
