#include "training_scanline_reference.hpp"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string_view>

using namespace vulkan3DGS;
namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template <class F> void rejects(F operation) {
    try {
        operation();
    } catch (const std::runtime_error&) {
        return;
    }
    throw std::runtime_error("Invalid capture was accepted");
}
TrainingScanlineCapture makeData() {
    TrainingScanlineCapture data;
    data.header.width = 17;
    data.header.height = 19;
    data.header.tileCount = 4;
    data.header.gaussianCount = 1;
    data.header.itemCount = 33;
    ProjectedGaussian g{};
    g.centerRadius = {0, 0, 1, 3};
    g.conicOpacity = {1, 0, 1, 0.01f};
    g.color = {64, 0.001f, 0, 0};
    data.projected.push_back(g);
    data.items.resize(33, 0u);
    data.ranges.resize(4, {0, 0, 0, 0});
    data.ranges[0] = {0, 33, 0, 0};
    data.blends.resize(17 * 19);
    data.pixelGrads.resize(data.blends.size());
    data.blends[0].processedCount = 33;
    return data;
}
} // namespace
int main(int argc, char** argv) {
    if (argc == 3 && std::string_view(argv[1]) == "--write-fixture") {
        try {
            auto data = makeData();
            data.projected[0].centerRadius.x = data.projected[0].centerRadius.y = 8.0f;
            for (uint32_t y = 0; y < 16u; ++y)
                for (uint32_t x = 0; x < 16u; ++x)
                    data.blends[y * 17u + x].processedCount = 33u;
            writeTrainingScanlineCapture(std::filesystem::path(argv[2]), data);
            return 0;
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
    if (argc != 1) return 2;
    const auto path =
        std::filesystem::temp_directory_path() / "vulkan-3dgs-scanline-capture-test.bin";
    try {
        auto data = makeData();
        const auto counts = scanline_reference::oracle(data);
        require(counts[1] == 33 && counts[2] == 33 * 255 && counts[4] == 33 && counts[11] == 3,
                "Prefix guard or reverse-aligned batch contract failed");
        require(counts[5] == 33 && counts[6] == 0 && counts[12] == 0,
                "Already guarded rows were counted as spatial opportunities");
        data.projected[0].conicOpacity.w = 0.001f;
        const auto empty = scanline_reference::oracle(data);
        require(empty[6] == 33 && empty[7] == 33 && empty[9] == 33 && empty[12] == 3,
                "Empty row/Gaussian/batch oracle failed");
        data.blends[0].processedCount = 1;
        require(scanline_reference::oracle(data)[29] == 32, "Trimmed suffix counted as new work");
        if (std::filesystem::exists(path)) std::filesystem::remove(path);
        writeTrainingScanlineCapture(path, data);
        require(std::filesystem::file_size(path) == trainingScanlineCaptureBytes(data.header),
                "Capture byte count failed");
        const auto roundtrip = readTrainingScanlineCapture(path);
        require(roundtrip.items == data.items && roundtrip.ranges == data.ranges &&
                    roundtrip.blends[0].processedCount == 1 && roundtrip.projected[0].color.x == 64,
                "Capture roundtrip failed");
        rejects([&] { writeTrainingScanlineCapture(path, data); });
        {
            std::ofstream file(path, std::ios::binary | std::ios::app);
            file.put('x');
        }
        rejects([&] { (void)readTrainingScanlineCapture(path); });
        std::filesystem::resize_file(path, 10);
        rejects([&] { (void)readTrainingScanlineCapture(path); });
        auto invalid = data;
        invalid.header.pixelCullMethod = 1;
        rejects([&] { validateTrainingScanlineCapture(invalid); });
        invalid = data;
        invalid.header.globalPixelGrad = 0;
        rejects([&] { validateTrainingScanlineCapture(invalid); });
        invalid = data;
        invalid.ranges[0].y = 34;
        rejects([&] { validateTrainingScanlineCapture(invalid); });
        invalid = data;
        invalid.header.width = UINT32_MAX;
        rejects([&] { (void)trainingScanlineCaptureBytes(invalid.header); });
        std::filesystem::remove(path);
        std::cout << "S3-2 capture/guard/batch/oracle tests passed\n";
    } catch (const std::exception& error) {
        std::filesystem::remove(path);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
