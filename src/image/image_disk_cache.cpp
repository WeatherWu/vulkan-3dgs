#include "image/image_disk_cache.hpp"

#include "image/image_decoder.hpp"
#include "utils/logger.hpp"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>

#include <ktx.h>
#include <vulkan/vulkan_core.h>

namespace vulkan3DGS {

namespace {

constexpr uint32_t kDiskCacheFormatVersion = 2u;

struct KtxTexture2Deleter {
    void operator()(ktxTexture2* texture) const {
        if (texture) {
            ktxTexture2_Destroy(texture);
        }
    }
};

using KtxTexture2Ptr = std::unique_ptr<ktxTexture2, KtxTexture2Deleter>;

void checkKtx(KTX_error_code result, const std::string& operation) {
    if (result != KTX_SUCCESS) {
        throw std::runtime_error(operation + ": " + ktxErrorString(result));
    }
}

uint64_t fnv1a64(const void* data, size_t size, uint64_t hash = 14695981039346656037ull) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

template <typename T>
uint64_t hashValue(uint64_t hash, const T& value) {
    return fnv1a64(&value, sizeof(value), hash);
}

uint64_t sourceFingerprint(const ImageSourceDesc& source) {
    const std::filesystem::path canonicalPath = std::filesystem::weakly_canonical(source.path);
    const std::string pathString = canonicalPath.generic_string();
    uint64_t hash = fnv1a64(pathString.data(), pathString.size());
    hash = hashValue(hash, kDiskCacheFormatVersion);
    hash = hashValue(hash, source.expectedWidth);
    hash = hashValue(hash, source.expectedHeight);
    hash = hashValue(hash, source.format);
    hash = hashValue(hash, source.colorSpace);

    std::error_code error;
    const uintmax_t fileSize = std::filesystem::file_size(source.path, error);
    if (!error) {
        hash = hashValue(hash, fileSize);
    }
    error.clear();
    const auto modified = std::filesystem::last_write_time(source.path, error);
    if (!error) {
        const auto ticks = modified.time_since_epoch().count();
        hash = hashValue(hash, ticks);
    }
    return hash;
}

bool chunkCompatible(const ImageSourceDesc& lhs, const ImageSourceDesc& rhs) {
    return lhs.expectedWidth == rhs.expectedWidth &&
           lhs.expectedHeight == rhs.expectedHeight &&
           lhs.format == rhs.format &&
           lhs.colorSpace == rhs.colorSpace;
}

std::string hashString(uint64_t value) {
    std::ostringstream stream;
    stream << std::hex << std::setw(16) << std::setfill('0') << value;
    return stream.str();
}

std::filesystem::path environmentPath(const char* name) {
#ifdef _WIN32
    char* value = nullptr;
    size_t length = 0;
    if (_dupenv_s(&value, &length, name) != 0 || !value) {
        return {};
    }
    const std::filesystem::path path(value);
    std::free(value);
    return path;
#else
    if (const char* value = std::getenv(name)) {
        return std::filesystem::path(value);
    }
    return {};
#endif
}

} // namespace

ImageDiskCache::ImageDiskCache(ImageDiskCacheConfig config)
    : config_(std::move(config)) {
    if (config_.rootDirectory.empty()) {
        config_.rootDirectory = defaultRootDirectory();
    }
    config_.chunkLayerCount = std::max(config_.chunkLayerCount, 1u);
}

void ImageDiskCache::setConfig(ImageDiskCacheConfig config) {
    std::lock_guard lock(mutex_);
    config_ = std::move(config);
    if (config_.rootDirectory.empty()) {
        config_.rootDirectory = defaultRootDirectory();
    }
    config_.chunkLayerCount = std::max(config_.chunkLayerCount, 1u);
    enforceQuota();
}

void ImageDiskCache::setSources(const std::vector<ImageSourceDesc>& sources) {
    std::lock_guard lock(mutex_);
    chunks_.clear();
    locations_.clear();

    std::vector<ImageSourceDesc> ordered = sources;
    std::sort(ordered.begin(), ordered.end(), [](const ImageSourceDesc& lhs, const ImageSourceDesc& rhs) {
        return lhs.id < rhs.id;
    });
    const uint32_t chunkLayerCount = std::max(config_.chunkLayerCount, 1u);
    for (const ImageSourceDesc& source : ordered) {
        if (chunks_.empty() ||
            chunks_.back().sources.size() >= chunkLayerCount ||
            !chunkCompatible(chunks_.back().sources.front(), source)) {
            chunks_.push_back(Chunk{});
        }
        Chunk& chunk = chunks_.back();
        const uint32_t layer = static_cast<uint32_t>(chunk.sources.size());
        chunk.sources.push_back(source);
        locations_.insert_or_assign(source.id, ChunkLocation{chunks_.size() - 1u, layer});
    }
    enforceQuota();
}

ImageDiskCacheConfig ImageDiskCache::config() const {
    std::lock_guard lock(mutex_);
    return config_;
}

ImageRgba8 ImageDiskCache::loadOrCreate(const ImageSourceDesc& source) {
    std::lock_guard lock(mutex_);
    if (!config_.enabled) {
        return ImageDecoder::decodeRgba8(source);
    }

    Chunk fallbackChunk{};
    const Chunk* chunk = nullptr;
    uint32_t requestedLayer = 0;
    const auto location = locations_.find(source.id);
    if (location != locations_.end() && location->second.chunkIndex < chunks_.size()) {
        chunk = &chunks_[location->second.chunkIndex];
        requestedLayer = location->second.layer;
    } else {
        fallbackChunk.sources.push_back(source);
        chunk = &fallbackChunk;
    }

    const std::filesystem::path path = cachePath(*chunk);
    if (std::filesystem::is_regular_file(path)) {
        try {
            ImageRgba8 image = loadKtx2(path,
                                        source,
                                        requestedLayer,
                                        static_cast<uint32_t>(chunk->sources.size()));
            std::error_code touchError;
            std::filesystem::last_write_time(path, std::filesystem::file_time_type::clock::now(), touchError);
            ++stats_.hits;
            return image;
        } catch (const std::exception& error) {
            ++stats_.readFailures;
            ++stats_.recoveries;
            LOG_WARN("Ignoring invalid image disk cache {}: {}", path.string(), error.what());
            std::error_code removeError;
            std::filesystem::remove(path, removeError);
        }
    }

    ++stats_.misses;
    std::vector<ImageRgba8> images;
    images.reserve(chunk->sources.size());
    for (const ImageSourceDesc& chunkSource : chunk->sources) {
        images.push_back(ImageDecoder::decodeRgba8(chunkSource));
    }
    ImageRgba8 image = images.at(requestedLayer);
    try {
        std::filesystem::create_directories(path.parent_path());
        std::filesystem::path temporaryPath = path;
        temporaryPath += ".tmp";
        std::error_code staleTemporaryError;
        std::filesystem::remove(temporaryPath, staleTemporaryError);
        writeKtx2(temporaryPath, images);

        std::error_code destinationRemoveError;
        std::filesystem::remove(path, destinationRemoveError);
        if (destinationRemoveError) {
            std::error_code temporaryRemoveError;
            std::filesystem::remove(temporaryPath, temporaryRemoveError);
            throw std::runtime_error("Failed to replace KTX2 cache file: " + destinationRemoveError.message());
        }
        std::error_code renameError;
        std::filesystem::rename(temporaryPath, path, renameError);
        if (renameError) {
            std::error_code removeError;
            std::filesystem::remove(temporaryPath, removeError);
            throw std::runtime_error("Failed to publish KTX2 cache file: " + renameError.message());
        }
        ++stats_.writes;
        enforceQuota(path);
    } catch (const std::exception& error) {
        LOG_WARN("Failed to write image disk cache {}: {}", path.string(), error.what());
    }
    return image;
}

ImageDiskCacheStats ImageDiskCache::stats() const {
    std::lock_guard lock(mutex_);
    return stats_;
}

std::filesystem::path ImageDiskCache::cachePath(const Chunk& chunk) const {
    uint64_t hash = hashValue(14695981039346656037ull, kDiskCacheFormatVersion);
    const uint32_t layerCount = static_cast<uint32_t>(chunk.sources.size());
    hash = hashValue(hash, layerCount);
    for (const ImageSourceDesc& source : chunk.sources) {
        const uint64_t fingerprint = sourceFingerprint(source);
        hash = hashValue(hash, fingerprint);
    }
    return config_.rootDirectory / ("chunk-" + hashString(hash) + ".ktx2");
}

ImageRgba8 ImageDiskCache::loadKtx2(const std::filesystem::path& path,
                                    const ImageSourceDesc& source,
                                    uint32_t layer,
                                    uint32_t layerCount) const {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open KTX2 image cache file");
    }
    const std::streamsize fileSize = file.tellg();
    if (fileSize <= 0) {
        throw std::runtime_error("KTX2 image cache file is empty");
    }
    std::vector<uint8_t> fileBytes(static_cast<size_t>(fileSize));
    file.seekg(0, std::ios::beg);
    if (!file.read(reinterpret_cast<char*>(fileBytes.data()), fileSize)) {
        throw std::runtime_error("Failed to read KTX2 image cache file");
    }
    file.close();

    ktxTexture2* rawTexture = nullptr;
    const KTX_error_code createResult = ktxTexture2_CreateFromMemory(
        fileBytes.data(),
        fileBytes.size(),
        KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT,
        &rawTexture);
    KtxTexture2Ptr texture(rawTexture);
    checkKtx(createResult, "Failed to open KTX2 image cache");

    if (texture->vkFormat != VK_FORMAT_R8G8B8A8_UNORM ||
        texture->baseWidth != source.expectedWidth ||
        texture->baseHeight != source.expectedHeight ||
        texture->numDimensions != 2u ||
        texture->numLevels != 1u ||
        texture->numLayers != layerCount ||
        texture->numFaces != 1u) {
        throw std::runtime_error("KTX2 cache metadata does not match the requested image");
    }

    const size_t expectedBytes = static_cast<size_t>(texture->baseWidth) *
                                 static_cast<size_t>(texture->baseHeight) * 4u;
    const ktx_size_t imageSize = ktxTexture_GetImageSize(ktxTexture(texture.get()), 0u);
    if (imageSize != expectedBytes) {
        throw std::runtime_error("KTX2 cache image size is invalid");
    }

    ktx_size_t imageOffset = 0;
    checkKtx(ktxTexture_GetImageOffset(ktxTexture(texture.get()), 0u, layer, 0u, &imageOffset),
             "Failed to locate KTX2 image data");
    const ktx_uint8_t* data = ktxTexture_GetData(ktxTexture(texture.get()));
    if (!data || imageOffset + expectedBytes > texture->dataSize) {
        throw std::runtime_error("KTX2 cache image data is incomplete");
    }

    ImageRgba8 image{};
    image.width = texture->baseWidth;
    image.height = texture->baseHeight;
    image.pixels.assign(data + imageOffset, data + imageOffset + expectedBytes);
    return image;
}

void ImageDiskCache::writeKtx2(const std::filesystem::path& path,
                               const std::vector<ImageRgba8>& images) const {
    if (images.empty()) {
        throw std::runtime_error("Cannot write an empty KTX2 image chunk");
    }
    const ImageRgba8& firstImage = images.front();
    for (const ImageRgba8& image : images) {
        if (image.width != firstImage.width || image.height != firstImage.height) {
            throw std::runtime_error("KTX2 image chunk contains mismatched dimensions");
        }
    }

    ktxTextureCreateInfo createInfo{};
    createInfo.vkFormat = VK_FORMAT_R8G8B8A8_UNORM;
    createInfo.baseWidth = firstImage.width;
    createInfo.baseHeight = firstImage.height;
    createInfo.baseDepth = 1u;
    createInfo.numDimensions = 2u;
    createInfo.numLevels = 1u;
    createInfo.numLayers = static_cast<ktx_uint32_t>(images.size());
    createInfo.numFaces = 1u;
    createInfo.isArray = images.size() > 1u ? KTX_TRUE : KTX_FALSE;
    createInfo.generateMipmaps = KTX_FALSE;

    ktxTexture2* rawTexture = nullptr;
    const KTX_error_code createResult = ktxTexture2_Create(
        &createInfo, KTX_TEXTURE_CREATE_ALLOC_STORAGE, &rawTexture);
    KtxTexture2Ptr texture(rawTexture);
    checkKtx(createResult, "Failed to create KTX2 image cache");
    for (uint32_t layer = 0; layer < images.size(); ++layer) {
        const ImageRgba8& image = images[layer];
        checkKtx(ktxTexture_SetImageFromMemory(ktxTexture(texture.get()),
                                               0u,
                                               layer,
                                               0u,
                                               image.pixels.data(),
                                               image.pixels.size()),
                 "Failed to populate KTX2 image cache layer");
    }
    checkKtx(ktxTexture2_WriteToNamedFile(texture.get(), path.string().c_str()),
             "Failed to write KTX2 image cache");
}

void ImageDiskCache::enforceQuota(const std::filesystem::path& protectedPath) {
    if (!config_.enabled || config_.diskQuotaBytes == 0 ||
        !std::filesystem::is_directory(config_.rootDirectory)) {
        refreshDiskUsageStats();
        return;
    }

    struct CacheFile {
        std::filesystem::path path;
        uint64_t size = 0;
        std::filesystem::file_time_type lastUse{};
    };
    std::vector<CacheFile> files;
    uint64_t totalBytes = 0;
    std::error_code iterateError;
    for (std::filesystem::directory_iterator iterator(config_.rootDirectory, iterateError), end;
         !iterateError && iterator != end;
         iterator.increment(iterateError)) {
        std::error_code entryError;
        if (!iterator->is_regular_file(entryError) || iterator->path().extension() != ".ktx2") {
            continue;
        }
        const uint64_t size = iterator->file_size(entryError);
        if (entryError) {
            continue;
        }
        const auto lastUse = iterator->last_write_time(entryError);
        if (entryError) {
            continue;
        }
        files.push_back(CacheFile{iterator->path(), size, lastUse});
        totalBytes += size;
    }

    std::sort(files.begin(), files.end(), [](const CacheFile& lhs, const CacheFile& rhs) {
        return lhs.lastUse < rhs.lastUse;
    });
    for (const CacheFile& file : files) {
        if (totalBytes <= config_.diskQuotaBytes) {
            break;
        }
        if (!protectedPath.empty() && file.path == protectedPath) {
            continue;
        }
        std::error_code removeError;
        if (std::filesystem::remove(file.path, removeError)) {
            totalBytes -= std::min(totalBytes, file.size);
            ++stats_.evictions;
            stats_.evictedBytes += file.size;
        }
    }
    stats_.cachedBytes = totalBytes;
    stats_.cachedFiles = static_cast<uint64_t>(std::count_if(
        files.begin(), files.end(), [](const CacheFile& file) {
            std::error_code error;
            return std::filesystem::is_regular_file(file.path, error);
        }));
}

void ImageDiskCache::refreshDiskUsageStats() {
    stats_.cachedBytes = 0;
    stats_.cachedFiles = 0;
    if (!std::filesystem::is_directory(config_.rootDirectory)) {
        return;
    }
    std::error_code iterateError;
    for (std::filesystem::directory_iterator iterator(config_.rootDirectory, iterateError), end;
         !iterateError && iterator != end;
         iterator.increment(iterateError)) {
        std::error_code entryError;
        if (!iterator->is_regular_file(entryError) || iterator->path().extension() != ".ktx2") {
            continue;
        }
        const uint64_t size = iterator->file_size(entryError);
        if (!entryError) {
            stats_.cachedBytes += size;
            ++stats_.cachedFiles;
        }
    }
}

std::filesystem::path ImageDiskCache::defaultRootDirectory() {
#ifdef _WIN32
    const std::filesystem::path localAppData = environmentPath("LOCALAPPDATA");
    if (!localAppData.empty()) {
        return localAppData / "vulkan-3dgs" / "image-cache";
    }
#else
    const std::filesystem::path xdgCacheHome = environmentPath("XDG_CACHE_HOME");
    if (!xdgCacheHome.empty()) {
        return xdgCacheHome / "vulkan-3dgs" / "image-cache";
    }
    const std::filesystem::path home = environmentPath("HOME");
    if (!home.empty()) {
        return home / ".cache" / "vulkan-3dgs" / "image-cache";
    }
#endif
    return std::filesystem::temp_directory_path() / "vulkan-3dgs" / "image-cache";
}

} // namespace vulkan3DGS
