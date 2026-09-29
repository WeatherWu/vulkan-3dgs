#pragma once

#include <filesystem>
#include <optional>
#include <string>

namespace vulkan3DGS {

struct TrainingFileDialogResult {
    std::optional<std::string> datasetDirectory;
    std::optional<std::string> outputDirectory;
    std::optional<std::filesystem::path> exportPath;
};

class TrainingFileDialogs {
public:
    void openDatasetFolder(const std::string& initialDirectory);
    void openOutputFolder(const std::string& initialDirectory);
    void openSavePly(const std::string& initialDirectory, const std::string& fileName);
    [[nodiscard]] TrainingFileDialogResult draw();
};

} // namespace vulkan3DGS
