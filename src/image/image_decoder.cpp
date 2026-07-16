#include "image/image_decoder.hpp"

#include <memory>
#include <stdexcept>
#include <string>

#include <stb_image.h>

namespace vulkan3DGS {

namespace {

struct StbiImageDeleter {
    void operator()(stbi_uc* data) const {
        stbi_image_free(data);
    }
};

} // namespace

ImageInfo ImageDecoder::probe(const std::filesystem::path& path) {
    int width = 0;
    int height = 0;
    int channels = 0;
    if (!stbi_info(path.string().c_str(), &width, &height, &channels)) {
        throw std::runtime_error("Failed to read image metadata: " + path.string());
    }
    if (width <= 0 || height <= 0 || channels <= 0) {
        throw std::runtime_error("Image metadata has invalid dimensions or channels: " + path.string());
    }
    return ImageInfo{static_cast<uint32_t>(width),
                     static_cast<uint32_t>(height),
                     static_cast<uint32_t>(channels)};
}

ImageRgba8 ImageDecoder::decodeRgba8(const ImageSourceDesc& source) {
    int width = 0;
    int height = 0;
    int channels = 0;
    std::unique_ptr<stbi_uc, StbiImageDeleter> data(
        stbi_load(source.path.string().c_str(), &width, &height, &channels, STBI_rgb_alpha));
    if (!data) {
        throw std::runtime_error("Failed to decode image: " + source.path.string());
    }
    if (width <= 0 || height <= 0) {
        throw std::runtime_error("Decoded image has invalid dimensions: " + source.path.string());
    }
    if ((source.expectedWidth != 0u && source.expectedWidth != static_cast<uint32_t>(width)) ||
        (source.expectedHeight != 0u && source.expectedHeight != static_cast<uint32_t>(height))) {
        throw std::runtime_error("Decoded image dimensions do not match metadata: " + source.path.string());
    }

    ImageRgba8 image{};
    image.width = static_cast<uint32_t>(width);
    image.height = static_cast<uint32_t>(height);
    const size_t byteCount = static_cast<size_t>(width) * static_cast<size_t>(height) * 4u;
    image.pixels.assign(data.get(), data.get() + byteCount);
    return image;
}

} // namespace vulkan3DGS
