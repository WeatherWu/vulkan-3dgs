#include "training/ui/training_file_dialogs.hpp"

#include <imgui.h>
#include <ImGuiFileDialog.h>

namespace vulkan3DGS {
namespace {
constexpr const char* kDatasetDialog = "TrainingDatasetFolderDialog";
constexpr const char* kOutputDialog = "TrainingOutputFolderDialog";
constexpr const char* kSaveDialog = "TrainingSavePlyDialog";
}

void TrainingFileDialogs::openDatasetFolder(const std::string& initialDirectory) {
    IGFD::FileDialogConfig config{}; config.path = initialDirectory;
    config.flags = ImGuiFileDialogFlags_Modal;
    ImGuiFileDialog::Instance()->OpenDialog(kDatasetDialog, "Select Training Dataset Folder", nullptr, config);
}
void TrainingFileDialogs::openOutputFolder(const std::string& initialDirectory) {
    IGFD::FileDialogConfig config{}; config.path = initialDirectory;
    config.flags = ImGuiFileDialogFlags_Modal;
    ImGuiFileDialog::Instance()->OpenDialog(kOutputDialog, "Select Output Folder", nullptr, config);
}
void TrainingFileDialogs::openSavePly(const std::string& initialDirectory, const std::string& fileName) {
    IGFD::FileDialogConfig config{}; config.path = initialDirectory; config.fileName = fileName;
    config.flags = ImGuiFileDialogFlags_Modal;
    ImGuiFileDialog::Instance()->OpenDialog(kSaveDialog, "Save Training Result", ".ply", config);
}
TrainingFileDialogResult TrainingFileDialogs::draw() {
    TrainingFileDialogResult result{};
    const ImVec2 minSize(620.0f, 360.0f), maxSize(FLT_MAX, FLT_MAX);
    if (ImGuiFileDialog::Instance()->Display(kDatasetDialog, ImGuiWindowFlags_NoCollapse, minSize, maxSize)) {
        if (ImGuiFileDialog::Instance()->IsOk()) result.datasetDirectory = ImGuiFileDialog::Instance()->GetCurrentPath();
        ImGuiFileDialog::Instance()->Close();
    }
    if (ImGuiFileDialog::Instance()->Display(kOutputDialog, ImGuiWindowFlags_NoCollapse, minSize, maxSize)) {
        if (ImGuiFileDialog::Instance()->IsOk()) result.outputDirectory = ImGuiFileDialog::Instance()->GetCurrentPath();
        ImGuiFileDialog::Instance()->Close();
    }
    if (ImGuiFileDialog::Instance()->Display(kSaveDialog, ImGuiWindowFlags_NoCollapse, minSize, maxSize)) {
        if (ImGuiFileDialog::Instance()->IsOk()) result.exportPath = std::filesystem::path(ImGuiFileDialog::Instance()->GetFilePathName());
        ImGuiFileDialog::Instance()->Close();
    }
    return result;
}
} // namespace vulkan3DGS
