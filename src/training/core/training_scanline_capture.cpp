#include "training/core/training_scanline_capture.hpp"

#include <bit>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace vulkan3DGS {
namespace {
void validateHeader(const TrainingScanlineHeader& h) {
    const TrainingScanlineHeader defaults;
    const uint64_t pixels = uint64_t(h.width) * h.height;
    const uint64_t tiles = ((uint64_t(h.width) + 15u) / 16u) * ((uint64_t(h.height) + 15u) / 16u);
    if (h.magic != defaults.magic || h.version != 1u || !pixels ||
        pixels > std::numeric_limits<uint32_t>::max() || tiles != h.tileCount ||
        h.compositeMode != 1u || h.pixelMode != 6u || h.subgroupReduction != 1u ||
        h.globalPixelGrad != 1u || h.rgbLatency || h.pixelCullMethod || h.ssimMode > 2u ||
        h.reserved0 || h.reserved1)
        throw std::runtime_error("Invalid S3-2 capture header; requires unculled Fixed16 Tensor "
                                 "Subgroup/Global Baseline");
}
template <class T> void readVector(std::istream& stream, std::vector<T>& values, size_t count) {
    values.resize(count);
    if (count)
        stream.read(reinterpret_cast<char*>(values.data()),
                    static_cast<std::streamsize>(count * sizeof(T)));
    if (!stream) throw std::runtime_error("Truncated S3-2 capture payload");
}
template <class T> void writeVector(std::ostream& stream, const std::vector<T>& values) {
    if (!values.empty())
        stream.write(reinterpret_cast<const char*>(values.data()),
                     static_cast<std::streamsize>(values.size() * sizeof(T)));
}
} // namespace

uint64_t trainingScanlineCaptureBytes(const TrainingScanlineHeader& h) {
    validateHeader(h);
    return sizeof(h) + uint64_t(h.gaussianCount) * sizeof(ProjectedGaussian) +
           uint64_t(h.itemCount) * sizeof(uint32_t) + uint64_t(h.tileCount) * sizeof(glm::uvec4) +
           uint64_t(h.width) * h.height * (sizeof(PixelBlendState) + sizeof(PixelGrad));
}

void validateTrainingScanlineCapture(const TrainingScanlineCapture& data) {
    const auto& h = data.header;
    validateHeader(h);
    if (data.projected.size() != h.gaussianCount || data.items.size() != h.itemCount ||
        data.ranges.size() != h.tileCount || data.blends.size() != uint64_t(h.width) * h.height ||
        data.pixelGrads.size() != data.blends.size())
        throw std::runtime_error("S3-2 capture payload dimensions do not match header");
    for (const auto& range : data.ranges) {
        if (uint64_t(range.x) + range.y > h.itemCount || range.y > UINT32_MAX / 256u)
            throw std::runtime_error(
                "S3-2 range is out of bounds or per-tile counters would overflow");
    }
}

void writeTrainingScanlineCapture(const std::filesystem::path& path,
                                  const TrainingScanlineCapture& data) {
    static_assert(std::endian::native == std::endian::little);
    static_assert(sizeof(ProjectedGaussian) == 48 && sizeof(PixelBlendState) == 16 &&
                  sizeof(PixelGrad) == 16);
    validateTrainingScanlineCapture(data);
    if (std::filesystem::exists(path))
        throw std::runtime_error("Refusing to overwrite S3-2 capture: " + path.string());
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    std::ofstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open S3-2 capture: " + path.string());
    file.write(reinterpret_cast<const char*>(&data.header), sizeof(data.header));
    writeVector(file, data.projected);
    writeVector(file, data.items);
    writeVector(file, data.ranges);
    writeVector(file, data.blends);
    writeVector(file, data.pixelGrads);
    file.flush();
    if (!file) throw std::runtime_error("Failed to write S3-2 capture: " + path.string());
}

TrainingScanlineCapture readTrainingScanlineCapture(const std::filesystem::path& path) {
    static_assert(std::endian::native == std::endian::little);
    std::ifstream file(path, std::ios::binary);
    TrainingScanlineCapture data;
    file.read(reinterpret_cast<char*>(&data.header), sizeof(data.header));
    if (!file || std::filesystem::file_size(path) != trainingScanlineCaptureBytes(data.header))
        throw std::runtime_error("S3-2 capture has truncated/trailing data or an invalid header");
    const auto& h = data.header;
    readVector(file, data.projected, h.gaussianCount);
    readVector(file, data.items, h.itemCount);
    readVector(file, data.ranges, h.tileCount);
    readVector(file, data.blends, size_t(h.width) * h.height);
    readVector(file, data.pixelGrads, data.blends.size());
    validateTrainingScanlineCapture(data);
    return data;
}
} // namespace vulkan3DGS
