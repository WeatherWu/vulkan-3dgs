#pragma once

#include "image/image_types.hpp"

namespace vulkan3DGS {

class ImageDecoder {
public:
    static ImageInfo probe(const std::filesystem::path& path);
    static ImageRgba8 decodeRgba8(const ImageSourceDesc& source);
};

} // namespace vulkan3DGS
