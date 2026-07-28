#include "image/image_streamer.hpp"

#include "utils/system_memory.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace vulkan3DGS {

ImageStreamer::ImageStreamer(ImageStreamerConfig config)
    : config_(std::move(config)),
      diskCache_(config_.diskCache),
      hostBudgetBytes_(resolveHostBudget()),
      worker_([this](std::stop_token stopToken) { workerMain(stopToken); }) {}

ImageStreamer::~ImageStreamer() {
    worker_.request_stop();
    queueCondition_.notify_all();
}

void ImageStreamer::setSources(std::vector<ImageSourceDesc> sources) {
    std::lock_guard lock(mutex_);
    diskCache_.setSources(sources);
    ++generation_;
    for (const auto& [id, entry] : entries_) {
        (void)id;
        entry->condition.notify_all();
    }
    sources_.clear();
    entries_.clear();
    queue_ = {};
    cachedBytes_ = 0;
    hostHits_ = 0;
    hostMisses_ = 0;
    hostWaits_ = 0;
    hostEvictions_ = 0;
    prefetchRequests_ = 0;
    for (ImageSourceDesc& source : sources) {
        sources_.insert_or_assign(source.id, std::move(source));
    }

    uint64_t totalBytes = 0;
    for (const auto& [id, source] : sources_) {
        (void)id;
        const uint64_t imageBytes = static_cast<uint64_t>(source.expectedWidth) *
                                    static_cast<uint64_t>(source.expectedHeight) * 4ull;
        if (std::numeric_limits<uint64_t>::max() - totalBytes < imageBytes) {
            totalBytes = std::numeric_limits<uint64_t>::max();
            break;
        }
        totalBytes += imageBytes;
    }

    if (totalBytes <= hostBudgetBytes_) {
        for (const auto& [id, source] : sources_) {
            (void)source;
            auto entry = std::make_shared<Entry>();
            entry->generation = generation_;
            entries_.insert_or_assign(id, entry);
            queue_.push(QueuedRequest{id, generation_, 0, sequence_++});
        }
        prefetchRequests_ += sources_.size();
        queueCondition_.notify_all();
    }
}

ImageHandle ImageStreamer::request(ImageId id) {
    std::shared_ptr<Entry> entry;
    uint64_t requestGeneration = 0;
    {
        std::unique_lock lock(mutex_);
        const auto sourceIt = sources_.find(id);
        if (sourceIt == sources_.end()) {
            throw std::runtime_error("Requested image is not registered with ImageStreamer");
        }
        requestGeneration = generation_;

        const auto entryIt = entries_.find(id);
        if (entryIt != entries_.end()) {
            entry = entryIt->second;
            if (entry->state == EntryState::Ready) {
                ++hostHits_;
                entry->lastUse = ++useCounter_;
                return ImageHandle(id, entry->image);
            }
            if (entry->state == EntryState::Loading) {
                ++hostWaits_;
                queue_.push(QueuedRequest{id,
                                          requestGeneration,
                                          std::numeric_limits<int>::max(),
                                          sequence_++});
                queueCondition_.notify_all();
                entry->condition.wait(lock, [&] {
                    return entry->state != EntryState::Loading || entry->generation != generation_;
                });
                if (entry->generation != generation_) {
                    throw std::runtime_error("Image request was cancelled because the source set changed");
                }
                if (entry->state == EntryState::Ready) {
                    ++hostHits_;
                    entry->lastUse = ++useCounter_;
                    return ImageHandle(id, entry->image);
                }
                std::rethrow_exception(entry->error);
            }
            std::rethrow_exception(entry->error);
        }

        ++hostMisses_;
        entry = std::make_shared<Entry>();
        entry->generation = requestGeneration;
        entries_.insert_or_assign(id, entry);
        queue_.push(QueuedRequest{id,
                                  requestGeneration,
                                  std::numeric_limits<int>::max(),
                                  sequence_++});
        queueCondition_.notify_all();
        entry->condition.wait(lock, [&] {
            return entry->state != EntryState::Loading || entry->generation != generation_;
        });
        if (entry->generation != generation_) {
            throw std::runtime_error("Image request was cancelled because the source set changed");
        }
        if (entry->state == EntryState::Ready) {
            entry->lastUse = ++useCounter_;
            return ImageHandle(id, entry->image);
        }
        std::rethrow_exception(entry->error);
    }
}

void ImageStreamer::prefetch(std::span<const ImageId> ids) {
    std::lock_guard lock(mutex_);
    int priority = static_cast<int>(ids.size());
    for (ImageId id : ids) {
        if (!sources_.contains(id)) {
            --priority;
            continue;
        }
        const auto existing = entries_.find(id);
        if (existing != entries_.end()) {
            if (existing->second->state == EntryState::Loading) {
                queue_.push(QueuedRequest{id, generation_, priority, sequence_++});
            }
            --priority;
            continue;
        }
        auto entry = std::make_shared<Entry>();
        entry->generation = generation_;
        entries_.insert_or_assign(id, entry);
        queue_.push(QueuedRequest{id, generation_, priority--, sequence_++});
        ++prefetchRequests_;
    }
    queueCondition_.notify_all();
}

void ImageStreamer::refreshMemoryBudget() {
    std::lock_guard lock(mutex_);
    hostBudgetBytes_ = resolveHostBudget();
    trimUnlocked();
}

void ImageStreamer::clear() {
    std::lock_guard lock(mutex_);
    diskCache_.setSources({});
    ++generation_;
    for (const auto& [id, entry] : entries_) {
        (void)id;
        entry->condition.notify_all();
    }
    sources_.clear();
    entries_.clear();
    queue_ = {};
    cachedBytes_ = 0;
    queueCondition_.notify_all();
}

ImageStreamerStats ImageStreamer::stats() const {
    std::lock_guard lock(mutex_);
    ImageStreamerStats result{};
    result.hostBudgetBytes = hostBudgetBytes_;
    result.hostCachedBytes = cachedBytes_;
    result.hostCachedImages = static_cast<uint64_t>(std::count_if(
        entries_.begin(), entries_.end(), [](const auto& item) {
            return item.second->state == EntryState::Ready;
        }));
    result.hostHits = hostHits_;
    result.hostMisses = hostMisses_;
    result.hostWaits = hostWaits_;
    result.hostEvictions = hostEvictions_;
    result.prefetchRequests = prefetchRequests_;
    result.disk = diskCache_.stats();
    return result;
}

uint64_t ImageStreamer::resolveHostBudget() const {
    if (config_.hostBudgetBytes > 0u) {
        return config_.hostBudgetBytes;
    }
    const uint64_t available = availableSystemMemoryBytes();
    const float fraction = std::clamp(config_.automaticHostBudgetFraction, 0.0f, 1.0f);
    const uint64_t fractionalBudget = static_cast<uint64_t>(static_cast<long double>(available) * fraction);
    const uint64_t reserveBudget = available > config_.hostReserveBytes
        ? available - config_.hostReserveBytes
        : fractionalBudget;
    return std::min(fractionalBudget, reserveBudget);
}

ImageDiskCacheChunk ImageStreamer::loadSourceChunk(const ImageSourceDesc& source) {
    return diskCache_.loadOrCreateChunk(source);
}

void ImageStreamer::workerMain(std::stop_token stopToken) {
    while (!stopToken.stop_requested()) {
        QueuedRequest request{};
        ImageSourceDesc source{};
        {
            std::unique_lock lock(mutex_);
            queueCondition_.wait(lock, stopToken, [&] { return !queue_.empty(); });
            if (stopToken.stop_requested()) {
                return;
            }
            request = queue_.top();
            queue_.pop();
            if (request.generation != generation_) {
                continue;
            }
            const auto sourceIt = sources_.find(request.id);
            if (sourceIt == sources_.end()) {
                continue;
            }
            const auto entryIt = entries_.find(request.id);
            if (entryIt == entries_.end() || entryIt->second->state != EntryState::Loading) {
                continue;
            }
            source = sourceIt->second;
        }

        try {
            ImageDiskCacheChunk chunk = loadSourceChunk(source);
            if (chunk.imageIds.size() != chunk.images.size()) {
                throw std::runtime_error("Image disk cache returned mismatched chunk metadata");
            }
            for (size_t index = 0; index < chunk.imageIds.size(); ++index) {
                publishLoaded(chunk.imageIds[index],
                              request.generation,
                              std::make_shared<const ImageRgba8>(std::move(chunk.images[index])),
                              nullptr);
            }
        } catch (...) {
            publishLoaded(request.id, request.generation, nullptr, std::current_exception());
        }
    }
}

void ImageStreamer::publishLoaded(ImageId id,
                                  uint64_t generation,
                                  std::shared_ptr<const ImageRgba8> image,
                                  std::exception_ptr error) {
    std::lock_guard lock(mutex_);
    if (generation != generation_ || !sources_.contains(id)) {
        return;
    }

    auto entryIt = entries_.find(id);
    if (entryIt == entries_.end()) {
        if (error) {
            return;
        }
        auto entry = std::make_shared<Entry>();
        entry->generation = generation;
        entryIt = entries_.insert_or_assign(id, std::move(entry)).first;
    }
    if (entryIt->second->generation != generation || entryIt->second->state == EntryState::Ready) {
        return;
    }

    const std::shared_ptr<Entry>& entry = entryIt->second;
    if (error) {
        entry->state = EntryState::Failed;
        entry->error = error;
    } else {
        entry->state = EntryState::Ready;
        entry->image = std::move(image);
        entry->lastUse = ++useCounter_;
        cachedBytes_ += entry->image->byteSize();
    }
    entry->condition.notify_all();
    trimUnlocked();
}

void ImageStreamer::trimUnlocked() {
    while (cachedBytes_ > hostBudgetBytes_) {
        auto candidate = entries_.end();
        uint64_t oldestUse = std::numeric_limits<uint64_t>::max();
        for (auto it = entries_.begin(); it != entries_.end(); ++it) {
            const std::shared_ptr<Entry>& entry = it->second;
            if (entry->state != EntryState::Ready || !entry->image || entry->image.use_count() > 1) {
                continue;
            }
            if (entry->lastUse < oldestUse) {
                oldestUse = entry->lastUse;
                candidate = it;
            }
        }
        if (candidate == entries_.end()) {
            break;
        }
        cachedBytes_ -= candidate->second->image->byteSize();
        entries_.erase(candidate);
        ++hostEvictions_;
    }
}

} // namespace vulkan3DGS
