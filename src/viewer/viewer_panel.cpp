#include "viewer/viewer_panel.hpp"
#include <imgui.h>
#include <algorithm>
namespace vulkan3DGS {
void ViewerPanel::draw(ViewerController& controller, float frameRate,
                       int framebufferWidth, int framebufferHeight) {
    ImGui::Begin("Camera");
    ImGui::Text("FPS %.1f", frameRate);
    if (frameRate > 0.0f) ImGui::Text("Frame %.2f ms", 1000.0f / frameRate);
    ImGui::Separator();
    int profile = static_cast<int>(controller.renderProfile());
    const char* labels[] = {"Legacy", "SuperSplat Compatible"};
    if (ImGui::Combo("Render Profile", &profile, labels, IM_ARRAYSIZE(labels)))
        controller.setRenderProfile(static_cast<GaussianRenderProfile>(profile));
    int bands = static_cast<int>(controller.shBands());
    if (ImGui::SliderInt("SH Bands", &bands, 0, 3)) controller.setSHBands(static_cast<uint32_t>(bands));
    int present = static_cast<int>(controller.presentMode());
    const char* presentLabels[] = {"最高帧率", "低延迟", "垂直同步"};
    if (ImGui::Combo("帧率模式", &present, presentLabels, 3))
        controller.setPresentMode(static_cast<PresentModePreference>(std::clamp(present, 0, 2)));
    bool flipY = controller.flipY(), flipZ = controller.flipZ();
    if (ImGui::Checkbox("Flip Y", &flipY)) controller.setFlipY(flipY);
    ImGui::SameLine();
    if (ImGui::Checkbox("Flip Z", &flipZ)) controller.setFlipZ(flipZ);
    int mode=static_cast<int>(controller.cameraController().mode());
    const char* modes[]={"Orbit","Fly"};
    if(ImGui::Combo("Camera Mode",&mode,modes,IM_ARRAYSIZE(modes))) controller.cameraController().setMode(static_cast<CameraControlMode>(mode));
    float fov=controller.cameraController().fovDegrees(); if(ImGui::SliderFloat("Field of View",&fov,10.0f,120.0f,"%.1f deg")) controller.cameraController().setFovDegrees(fov);
    float damping=controller.cameraController().dampingSeconds(); if(ImGui::SliderFloat("Damping",&damping,0.0f,0.5f,"%.3f s")) controller.cameraController().setDampingSeconds(damping);
    float distance=controller.cameraController().distance(); if(ImGui::SliderFloat("Distance",&distance,0.01f,std::max(distance*3.0f,20.0f),"%.3f")) controller.cameraController().setDistance(distance);
    const float width=static_cast<float>(std::max(framebufferWidth,1));
    const float height=static_cast<float>(std::max(framebufferHeight,1));
    if(ImGui::Button("Focus")) controller.focusCamera(width,height); ImGui::SameLine(); if(ImGui::Button("Reset")) controller.resetCamera(width,height);
    ImGui::Text("Clip %.6g .. %.6g",controller.nearPlane(),controller.farPlane());
    ImGui::End();
}
} // namespace vulkan3DGS
