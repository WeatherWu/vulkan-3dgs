#pragma once

#include "training/core/training_types.hpp"

#include <array>
#include <filesystem>
#include <vector>

namespace vulkan3DGS {

// Explicit offline diagnostic input, NOT a training checkpoint or gradient export.
struct TrainingScanlineHeader {
    std::array<char, 8> magic{'V', '3', 'D', 'S', 'C', 'N', '0', '1'};
    uint32_t version = 1;
    uint32_t gaussianCount = 0;
    uint32_t iteration = 0;
    uint32_t frame = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t itemCount = 0;
    uint32_t tileCount = 0;
    uint32_t compositeMode = 1;
    uint32_t pixelMode = 6;
    uint32_t subgroupReduction = 1;
    uint32_t globalPixelGrad = 1;
    uint32_t rgbLatency = 0;
    uint32_t ssimMode = 0;
    uint32_t pixelCullMethod = 0;
    uint32_t vendorID = 0;
    uint32_t deviceID = 0;
    uint32_t driverVersion = 0;
    uint32_t reserved0 = 0;
    uint32_t reserved1 = 0;
};
static_assert(sizeof(TrainingScanlineHeader) == 88u);

struct TrainingScanlineCapture {
    TrainingScanlineHeader header;
    std::vector<ProjectedGaussian> projected;
    std::vector<uint32_t> items;
    std::vector<glm::uvec4> ranges;
    std::vector<PixelBlendState> blends;
    std::vector<PixelGrad> pixelGrads;
};

// Per-tile uint32 counters; aggregate on CPU with uint64 to avoid frame-wide overflow.
inline constexpr std::array<const char*, 32> kScanlineCounterNames = {
    "valid_prefix_pairs",
    "eligible_pairs",
    "already_guarded_pairs",
    "old_exp_pairs",
    "contributing_pairs",
    "eligible_rows",
    "oracle_empty_rows",
    "oracle_empty_row_pairs",
    "eligible_gaussian_items",
    "oracle_empty_gaussian_items",
    "oracle_empty_gaussian_pairs",
    "eligible_batches",
    "oracle_empty_batches",
    "oracle_empty_batch_pairs",
    "span_rejected_pairs",
    "span_rejected_exp_pairs",
    "span_empty_rows",
    "span_empty_row_pairs",
    "span_empty_gaussian_items",
    "span_empty_gaussian_pairs",
    "span_empty_batches",
    "span_empty_batch_pairs",
    "fallback_eligible_rows",
    "span_rows_built",
    "false_rejected_contributors",
    "invalid_id_items",
    "range_errors",
    "checksum",
    "processed_prefix_items",
    "already_trimmed_suffix_items",
    "span_retained_pixels",
    "clamped_processed_pixels"};
using TrainingScanlineTileCounters = std::array<uint32_t, 32>;

[[nodiscard]] uint64_t trainingScanlineCaptureBytes(const TrainingScanlineHeader& header);
void validateTrainingScanlineCapture(const TrainingScanlineCapture& capture);
void writeTrainingScanlineCapture(const std::filesystem::path& path,
                                  const TrainingScanlineCapture& capture);
[[nodiscard]] TrainingScanlineCapture
readTrainingScanlineCapture(const std::filesystem::path& path);

} // namespace vulkan3DGS
