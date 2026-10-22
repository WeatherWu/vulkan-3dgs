#pragma once

#include "training/core/training_types.hpp"
#include <vulkan/vulkan.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <span>
#include <string>

namespace vulkan3DGS {

// Independent optimizer capture protocol; gradient/scanline exports use different formats.
inline constexpr std::array<uint32_t, 9> optimizerCaptureStrides{240, 480, 48,  48, 16,
                                                                 8,   240, 480, 16};
inline constexpr std::array<uint32_t, 6> optimizerCaptureBindings{0, 2, 3, 13, 16, 32};
inline constexpr uint64_t optimizerHashSeed = 14695981039346656037ull;
inline constexpr uint64_t optimizerCaptureFileBudget = 8ull << 30;
inline constexpr size_t optimizerCaptureChunkBytes = 1u << 20;

struct OptimizerCaptureBlock {
    uint64_t bytes = 0;
    uint64_t checksum = 0;
};

// Version 1 is little endian, with explicit ABI sizes and bounded identity strings.
// FNV-1a-64 detects accidental corruption; this is not an authentication format.
struct alignas(16) OptimizerCaptureHeader {
    std::array<char, 8> magic{'V', '3', 'D', 'G', 'O', 'P', 'T', '1'};
    uint32_t version = 1;
    uint32_t headerBytes = sizeof(OptimizerCaptureHeader);
    uint32_t gaussianCount = 0;
    uint32_t capacity = 0;
    uint32_t frame = 0;
    uint32_t completedIteration = 0;
    uint32_t completedOptimizerStep = 0;
    uint32_t vendorID = 0;
    uint32_t deviceID = 0;
    uint32_t driverVersion = 0;
    uint32_t apiVersion = 0;
    uint32_t pixelMode = 0;
    uint32_t ssimMode = 0;
    uint32_t tileMode = 0;
    uint32_t compositeMode = 0;
    uint32_t tensorFlags = 0; // subgroup/global/RGB latency/actual EarlyCull
    uint32_t cameraBytes = sizeof(TrainingForwardCamera);
    uint32_t pushBytes = sizeof(TrainingPushConstants);
    uint32_t ordinaryFusedStep = 1;
    uint32_t synthetic = 0;
    std::array<uint32_t, 9> strides = optimizerCaptureStrides;
    uint32_t reserved = 0;
    std::array<uint8_t, 16> deviceUUID{};
    std::array<char, 128> deviceName{};
    std::array<char, 128> compiler{};
    uint64_t shaderHash = 0; // FNV-1a-64 of the exact dispatched SPIR-V
    uint64_t headerChecksum = 0;
    std::array<OptimizerCaptureBlock, 9> blocks{};
    TrainingForwardCamera camera{};
    TrainingPushConstants push{};
};
static_assert(sizeof(OptimizerCaptureHeader) % 16 == 0);
static_assert(sizeof(OptimizerCaptureHeader) == 928);
static_assert(offsetof(OptimizerCaptureHeader, camera) == 560);
static_assert(offsetof(OptimizerCaptureHeader, push) == 800);

uint64_t optimizerHash(std::span<const std::byte> bytes, uint64_t seed = optimizerHashSeed);
uint64_t optimizerShaderHash(const std::filesystem::path& path);
uint64_t optimizerCaptureBytes(uint32_t count);
uint64_t optimizerCaptureOffset(const OptimizerCaptureHeader& header, size_t block);
void validateOptimizerCaptureHeader(const OptimizerCaptureHeader& header, uint64_t fileBytes,
                                    uint64_t budget = optimizerCaptureFileBudget);
OptimizerCaptureHeader readOptimizerCaptureHeader(const std::filesystem::path& path,
                                                  bool verifyPayload = true);
void readOptimizerCaptureChunk(const std::filesystem::path& path,
                               const OptimizerCaptureHeader& header, size_t block, uint64_t offset,
                               std::span<std::byte> destination);
using OptimizerCaptureSource =
    std::function<void(size_t block, uint64_t offset, std::span<std::byte> destination)>;
void writeOptimizerCapture(const std::filesystem::path& path, OptimizerCaptureHeader header,
                           const OptimizerCaptureSource& source);
void fillOptimizerCaptureIdentity(OptimizerCaptureHeader& header, vk::PhysicalDevice physical,
                                  const std::filesystem::path& shader);
bool optimizerCaptureIdentityMatches(const OptimizerCaptureHeader& a,
                                     const OptimizerCaptureHeader& b);
std::string optimizerCaptureIdentityJson(const OptimizerCaptureHeader& header);

} // namespace vulkan3DGS
