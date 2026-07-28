#pragma once

#include <condition_variable>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <queue>
#include <span>
#include <stop_token>
#include <thread>
#include <unordered_map>
#include <vector>

#include "image/image_disk_cache.hpp"
#include "image/image_types.hpp"

namespace vulkan3DGS {

struct ImageStreamerConfig {
    uint64_t hostBudgetBytes = 0;
    float automaticHostBudgetFraction = 0.1f;
    uint64_t hostReserveBytes = 2ull * 1024ull * 1024ull * 1024ull;
    ImageDiskCacheConfig diskCache{};
};

struct ImageStreamerStats {
    uint64_t hostBudgetBytes = 0;
    uint64_t hostCachedBytes = 0;
    uint64_t hostCachedImages = 0;
    uint64_t hostHits = 0;
    uint64_t hostMisses = 0;
    uint64_t hostWaits = 0;
    uint64_t hostEvictions = 0;
    uint64_t prefetchRequests = 0;
    ImageDiskCacheStats disk{};
};

class ImageHandle {
public:
    ImageHandle() = default;
    ImageHandle(ImageId id, std::shared_ptr<const ImageRgba8> image)
        : id_(id), image_(std::move(image)) {}

    explicit operator bool() const { return static_cast<bool>(image_); }
    ImageId id() const { return id_; }
    const ImageRgba8& image() const { return *image_; }

private:
    ImageId id_ = 0;
    std::shared_ptr<const ImageRgba8> image_;
};

class ImageStreamer {
public:
    explicit ImageStreamer(ImageStreamerConfig config = {});
    ~ImageStreamer();

    ImageStreamer(const ImageStreamer&) = delete;
    ImageStreamer& operator=(const ImageStreamer&) = delete;

    void setSources(std::vector<ImageSourceDesc> sources);
    ImageHandle request(ImageId id);
    void prefetch(std::span<const ImageId> ids);
    void refreshMemoryBudget();
    void clear();

    ImageStreamerStats stats() const;

private:
    enum class EntryState {
        Loading,
        Ready,
        Failed,
    };

    struct Entry {
        EntryState state = EntryState::Loading;
        uint64_t generation = 0;
        uint64_t lastUse = 0;
        std::shared_ptr<const ImageRgba8> image;
        std::exception_ptr error;
        std::condition_variable condition;
    };

    struct QueuedRequest {
        ImageId id = 0;
        uint64_t generation = 0;
        int priority = 0;
        uint64_t sequence = 0;
    };

    struct QueuedRequestLess {
        bool operator()(const QueuedRequest& lhs, const QueuedRequest& rhs) const {
            if (lhs.priority != rhs.priority) {
                return lhs.priority < rhs.priority;
            }
            return lhs.sequence > rhs.sequence;
        }
    };

    uint64_t resolveHostBudget() const;
    ImageDiskCacheChunk loadSourceChunk(const ImageSourceDesc& source);
    void workerMain(std::stop_token stopToken);
    void publishLoaded(ImageId id,
                       uint64_t generation,
                       std::shared_ptr<const ImageRgba8> image,
                       std::exception_ptr error);
    void trimUnlocked();

    ImageStreamerConfig config_;
    ImageDiskCache diskCache_;
    mutable std::mutex mutex_;
    std::condition_variable_any queueCondition_;
    std::unordered_map<ImageId, ImageSourceDesc> sources_;
    std::unordered_map<ImageId, std::shared_ptr<Entry>> entries_;
    std::priority_queue<QueuedRequest, std::vector<QueuedRequest>, QueuedRequestLess> queue_;
    std::jthread worker_;
    uint64_t generation_ = 1;
    uint64_t sequence_ = 0;
    uint64_t useCounter_ = 0;
    uint64_t hostBudgetBytes_ = 0;
    uint64_t cachedBytes_ = 0;
    uint64_t hostHits_ = 0;
    uint64_t hostMisses_ = 0;
    uint64_t hostWaits_ = 0;
    uint64_t hostEvictions_ = 0;
    uint64_t prefetchRequests_ = 0;
};

} // namespace vulkan3DGS
