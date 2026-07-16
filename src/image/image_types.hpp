#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace vulkan3DGS {

using ImageId = uint64_t;

enum class ImagePixelFormat : uint32_t {
    Rgba8Unorm = 0,
};

enum class ImageColorSpace : uint32_t {
    LinearUnorm = 0,
    Srgb = 1,
};

struct ImageSourceDesc {
    ImageId id = 0;
    std::filesystem::path path;
    uint32_t expectedWidth = 0;
    uint32_t expectedHeight = 0;
    ImagePixelFormat format = ImagePixelFormat::Rgba8Unorm;
    ImageColorSpace colorSpace = ImageColorSpace::LinearUnorm;
};

struct ImageInfo {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t channels = 0;
};

struct ImageRgba8 {
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<uint8_t> pixels;

    size_t byteSize() const { return pixels.size(); }
    bool empty() const { return pixels.empty(); }
};

} // namespace vulkan3DGS
