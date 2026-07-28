#pragma once

#include <cstdint>
#include <filesystem>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "image/image_types.hpp"

namespace vulkan3DGS {

struct ImageDiskCacheConfig {
    bool enabled = true;
    std::filesystem::path rootDirectory;
    uint32_t chunkLayerCount = 16;
    uint64_t diskQuotaBytes = 20ull * 1024ull * 1024ull * 1024ull;
};

struct ImageDiskCacheStats {
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t writes = 0;
    uint64_t readFailures = 0;
    uint64_t recoveries = 0;
    uint64_t evictions = 0;
    uint64_t evictedBytes = 0;
    uint64_t cachedBytes = 0;
    uint64_t cachedFiles = 0;
    uint64_t activeBytes = 0;
    uint64_t activeFiles = 0;
    uint64_t historicalBytes = 0;
    uint64_t historicalFiles = 0;
};

struct ImageDiskCacheChunk {
    std::vector<ImageId> imageIds;
    std::vector<ImageRgba8> images;
};

class ImageDiskCache {
public:
    explicit ImageDiskCache(ImageDiskCacheConfig config = {});

    ImageRgba8 loadOrCreate(const ImageSourceDesc& source);
    ImageDiskCacheChunk loadOrCreateChunk(const ImageSourceDesc& source);
    void setConfig(ImageDiskCacheConfig config);
    void setSources(const std::vector<ImageSourceDesc>& sources);

    ImageDiskCacheConfig config() const;
    ImageDiskCacheStats stats() const;

    static std::filesystem::path defaultRootDirectory();

private:
    struct Chunk {
        std::vector<ImageSourceDesc> sources;
    };

    struct ChunkLocation {
        size_t chunkIndex = 0;
        uint32_t layer = 0;
    };

    std::filesystem::path cachePath(const Chunk& chunk) const;
    std::vector<ImageRgba8> loadKtx2(const std::filesystem::path& path,
                                     const ImageSourceDesc& source,
                                     uint32_t layerCount) const;
    void writeKtx2(const std::filesystem::path& path,
                   const std::vector<ImageRgba8>& images) const;
    void enforceQuota(const std::filesystem::path& protectedPath = {});
    void refreshDiskUsageStats();
    void refreshActiveCachePaths();

    ImageDiskCacheConfig config_;
    ImageDiskCacheStats stats_{};
    std::vector<Chunk> chunks_;
    std::unordered_map<ImageId, ChunkLocation> locations_;
    std::unordered_set<std::filesystem::path> activeCachePaths_;
    mutable std::mutex mutex_;
};

} // namespace vulkan3DGS
