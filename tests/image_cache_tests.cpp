#include "image/image_decoder.hpp"
#include "image/image_disk_cache.hpp"
#include "image/image_streamer.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

using namespace vulkan3DGS;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void writeTga(const std::filesystem::path& path,
              uint16_t width,
              uint16_t height,
              const std::vector<uint8_t>& rgba) {
    require(rgba.size() == static_cast<size_t>(width) * height * 4u,
            "TGA test pixels have the wrong size");
    std::array<uint8_t, 18> header{};
    header[2] = 2;
    header[12] = static_cast<uint8_t>(width & 0xffu);
    header[13] = static_cast<uint8_t>(width >> 8u);
    header[14] = static_cast<uint8_t>(height & 0xffu);
    header[15] = static_cast<uint8_t>(height >> 8u);
    header[16] = 32;
    header[17] = 0x28;

    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    require(file.is_open(), "Failed to create TGA test image");
    file.write(reinterpret_cast<const char*>(header.data()), header.size());
    for (size_t i = 0; i < rgba.size(); i += 4u) {
        const std::array<uint8_t, 4> bgra = {rgba[i + 2], rgba[i + 1], rgba[i], rgba[i + 3]};
        file.write(reinterpret_cast<const char*>(bgra.data()), bgra.size());
    }
    require(file.good(), "Failed to write TGA test image");
}

ImageSourceDesc source(ImageId id, const std::filesystem::path& path) {
    ImageSourceDesc result{};
    result.id = id;
    result.path = path;
    result.expectedWidth = 2;
    result.expectedHeight = 2;
    return result;
}

std::vector<uint8_t> solidPixels(uint8_t red, uint8_t green, uint8_t blue) {
    std::vector<uint8_t> pixels(16u);
    for (size_t i = 0; i < pixels.size(); i += 4u) {
        pixels[i] = red;
        pixels[i + 1] = green;
        pixels[i + 2] = blue;
        pixels[i + 3] = 255;
    }
    return pixels;
}

std::vector<std::filesystem::path> ktxFiles(const std::filesystem::path& directory) {
    std::vector<std::filesystem::path> files;
    if (!std::filesystem::is_directory(directory)) {
        return files;
    }
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (entry.is_regular_file() && entry.path().extension() == ".ktx2") {
            files.push_back(entry.path());
        }
    }
    return files;
}

void testKtxRoundTrip(const std::filesystem::path& root) {
    const std::filesystem::path imagePath = root / "source.tga";
    const std::filesystem::path cachePath = root / "ktx-cache";
    const std::vector<uint8_t> firstPixels = solidPixels(12, 34, 56);
    writeTga(imagePath, 2, 2, firstPixels);

    ImageDiskCache cache(ImageDiskCacheConfig{true, cachePath});
    const ImageSourceDesc imageSource = source(1, imagePath);
    const ImageRgba8 first = cache.loadOrCreate(imageSource);
    require(first.pixels == firstPixels, "Initial decoded RGBA8 data is incorrect");

    const ImageRgba8 cached = cache.loadOrCreate(imageSource);
    require(cached.pixels == firstPixels, "KTX2 round-trip changed RGBA8 data");
    ImageDiskCacheStats stats = cache.stats();
    require(stats.misses == 1 && stats.writes == 1 && stats.hits == 1,
            "Unexpected KTX2 cache hit/miss counters");

    const std::vector<uint8_t> secondPixels = solidPixels(90, 80, 70);
    writeTga(imagePath, 2, 2, secondPixels);
    const auto modified = std::filesystem::last_write_time(imagePath);
    std::filesystem::last_write_time(imagePath, modified + std::chrono::seconds(2));
    const ImageRgba8 refreshed = cache.loadOrCreate(imageSource);
    require(refreshed.pixels == secondPixels, "Source fingerprint did not invalidate KTX2 cache");
    stats = cache.stats();
    require(stats.misses == 2 && stats.writes == 2,
            "KTX2 cache invalidation counters are incorrect");
}

void testHostBudgetAndSourceReset(const std::filesystem::path& root) {
    std::vector<ImageSourceDesc> sources;
    for (uint64_t id = 0; id < 3; ++id) {
        const std::filesystem::path path = root / ("host-" + std::to_string(id) + ".tga");
        writeTga(path, 2, 2, solidPixels(static_cast<uint8_t>(id * 20u), 10, 30));
        sources.push_back(source(id, path));
    }

    ImageStreamerConfig config{};
    config.hostBudgetBytes = 16;
    config.diskCache.enabled = false;
    ImageStreamer streamer(config);
    streamer.setSources(sources);

    ImageHandle first = streamer.request(0);
    ImageHandle second = streamer.request(1);
    require(first.image().pixels[0] == 0 && second.image().pixels[0] == 20,
            "ImageStreamer returned the wrong image");
    first = {};
    second = {};
    ImageHandle third = streamer.request(2);
    require(third.image().pixels[0] == 40, "ImageStreamer returned the wrong replacement image");
    ImageStreamerStats stats = streamer.stats();
    require(stats.hostCachedBytes <= config.hostBudgetBytes,
            "ImageStreamer exceeded the unpinned host budget");
    require(stats.hostEvictions > 0, "ImageStreamer did not evict over-budget images");

    streamer.setSources({sources[0]});
    ImageHandle resetImage = streamer.request(0);
    require(resetImage.image().pixels[0] == 0, "ImageStreamer source reset returned stale data");
}

void testKtxChunkQuotaAndRecovery(const std::filesystem::path& root) {
    const std::filesystem::path cachePath = root / "chunk-cache";
    std::vector<ImageSourceDesc> sources;
    for (uint64_t id = 0; id < 4; ++id) {
        const std::filesystem::path path = root / ("chunk-" + std::to_string(id) + ".tga");
        writeTga(path, 2, 2, solidPixels(static_cast<uint8_t>(30u + id * 20u), 15, 25));
        sources.push_back(source(id, path));
    }

    ImageDiskCacheConfig config{};
    config.rootDirectory = cachePath;
    config.chunkLayerCount = 2;
    config.diskQuotaBytes = std::numeric_limits<uint64_t>::max();
    ImageDiskCache cache(config);
    cache.setSources(sources);

    const ImageRgba8 first = cache.loadOrCreate(sources[0]);
    const ImageRgba8 second = cache.loadOrCreate(sources[1]);
    require(first.pixels[0] == 30 && second.pixels[0] == 50,
            "KTX2 array chunk returned the wrong layer");
    std::vector<std::filesystem::path> files = ktxFiles(cachePath);
    require(files.size() == 1, "Two images in one chunk did not produce one KTX2 file");

    {
        std::ofstream corrupt(files.front(), std::ios::binary | std::ios::trunc);
        corrupt << "invalid";
    }
    const ImageRgba8 recovered = cache.loadOrCreate(sources[0]);
    require(recovered.pixels == first.pixels, "Corrupt KTX2 chunk was not rebuilt correctly");
    ImageDiskCacheStats stats = cache.stats();
    require(stats.readFailures == 1 && stats.recoveries == 1,
            "Corrupt KTX2 recovery counters are incorrect");

    files = ktxFiles(cachePath);
    require(files.size() == 1, "Recovered KTX2 chunk count is incorrect");
    const uint64_t firstChunkBytes = std::filesystem::file_size(files.front());
    config.diskQuotaBytes = firstChunkBytes + firstChunkBytes / 2u;
    cache.setConfig(config);
    const ImageRgba8 third = cache.loadOrCreate(sources[2]);
    require(third.pixels[0] == 70, "Second KTX2 chunk returned the wrong layer");
    stats = cache.stats();
    require(stats.evictions > 0, "KTX2 disk quota did not evict an old chunk");
    require(stats.cachedBytes <= config.diskQuotaBytes,
            "KTX2 disk cache remained above its quota");
}

void testStreamerPublishesWholeKtxChunks(const std::filesystem::path& root) {
    const std::filesystem::path cachePath = root / "streamer-chunk-cache";
    std::vector<ImageSourceDesc> sources;
    for (uint64_t id = 0; id < 4; ++id) {
        const std::filesystem::path path = root / ("streamer-chunk-" + std::to_string(id) + ".tga");
        writeTga(path, 2, 2, solidPixels(static_cast<uint8_t>(10u + id * 20u), 25, 35));
        sources.push_back(source(id, path));
    }

    ImageStreamerConfig config{};
    config.hostBudgetBytes = 4u * 2u * 2u * 4u;
    config.diskCache.rootDirectory = cachePath;
    config.diskCache.chunkLayerCount = 2;
    config.diskCache.diskQuotaBytes = std::numeric_limits<uint64_t>::max();
    ImageStreamer streamer(config);
    streamer.setSources(sources);

    for (uint64_t id = 0; id < sources.size(); ++id) {
        const ImageHandle image = streamer.request(id);
        require(image.image().pixels[0] == static_cast<uint8_t>(10u + id * 20u),
                "ImageStreamer published the wrong KTX2 chunk layer");
    }

    const ImageStreamerStats stats = streamer.stats();
    require(stats.hostCachedImages == sources.size(),
            "ImageStreamer did not publish every image in the loaded chunks");
    require(stats.disk.misses == 2 && stats.disk.writes == 2,
            "ImageStreamer loaded a KTX2 chunk more than once during full prefetch");
}

void testHistoricalChunksAreEvictedFirst(const std::filesystem::path& root) {
    const std::filesystem::path cachePath = root / "active-history-cache";
    std::vector<ImageSourceDesc> oldSources;
    std::vector<ImageSourceDesc> activeSources;
    for (uint64_t id = 0; id < 2; ++id) {
        const std::filesystem::path oldPath = root / ("history-" + std::to_string(id) + ".tga");
        const std::filesystem::path activePath = root / ("active-" + std::to_string(id) + ".tga");
        writeTga(oldPath, 2, 2, solidPixels(static_cast<uint8_t>(20u + id * 10u), 30, 40));
        writeTga(activePath, 2, 2, solidPixels(static_cast<uint8_t>(60u + id * 10u), 70, 80));
        oldSources.push_back(source(id, oldPath));
        activeSources.push_back(source(id, activePath));
    }

    ImageDiskCacheConfig config{};
    config.rootDirectory = cachePath;
    config.chunkLayerCount = 2;
    config.diskQuotaBytes = std::numeric_limits<uint64_t>::max();
    ImageDiskCache cache(config);
    cache.setSources(oldSources);
    (void)cache.loadOrCreateChunk(oldSources[0]);
    cache.setSources(activeSources);
    (void)cache.loadOrCreateChunk(activeSources[0]);

    ImageDiskCacheStats stats = cache.stats();
    require(stats.activeFiles == 1 && stats.historicalFiles == 1,
            "KTX2 cache did not distinguish active and historical chunks");

    config.diskQuotaBytes = stats.activeBytes + stats.activeBytes / 2u;
    cache.setConfig(config);
    stats = cache.stats();
    require(stats.activeFiles == 1 && stats.historicalFiles == 0,
            "KTX2 quota did not evict historical chunks before active chunks");
    require(stats.cachedBytes <= config.diskQuotaBytes,
            "KTX2 active/history cache remained above its quota");
}

void testPackedRgba8Normalization() {
    const std::array<uint8_t, 4> rgba = {17, 91, 203, 255};
    const uint32_t packed = static_cast<uint32_t>(rgba[0]) |
                            (static_cast<uint32_t>(rgba[1]) << 8u) |
                            (static_cast<uint32_t>(rgba[2]) << 16u) |
                            (static_cast<uint32_t>(rgba[3]) << 24u);
    constexpr float inverse255 = 1.0f / 255.0f;
    const std::array<float, 4> unpacked = {
        static_cast<float>(packed & 0xffu) * inverse255,
        static_cast<float>((packed >> 8u) & 0xffu) * inverse255,
        static_cast<float>((packed >> 16u) & 0xffu) * inverse255,
        static_cast<float>((packed >> 24u) & 0xffu) * inverse255,
    };
    for (size_t channel = 0; channel < rgba.size(); ++channel) {
        const float expected = static_cast<float>(rgba[channel]) * inverse255;
        require(std::abs(unpacked[channel] - expected) <= std::numeric_limits<float>::epsilon(),
                "Packed RGBA8 normalization differs from the float target baseline");
    }
}

} // namespace

int main() {
    const std::filesystem::path root = std::filesystem::temp_directory_path() /
                                       "vulkan-3dgs-image-cache-tests";
    std::error_code error;
    std::filesystem::remove_all(root, error);
    std::filesystem::create_directories(root);

    try {
        testKtxRoundTrip(root);
        testHostBudgetAndSourceReset(root);
        testKtxChunkQuotaAndRecovery(root);
        testStreamerPublishesWholeKtxChunks(root);
        testHistoricalChunksAreEvictedFirst(root);
        testPackedRgba8Normalization();
        std::filesystem::remove_all(root, error);
        std::cout << "image cache tests passed\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "image cache tests failed: " << exception.what() << '\n';
        return 1;
    }
}
