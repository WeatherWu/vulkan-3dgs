#include "training/core/training_optimizer_capture.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace vulkan3DGS {
namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
uint64_t headerHash(OptimizerCaptureHeader header) {
    header.headerChecksum = 0;
    return optimizerHash(std::as_bytes(std::span{&header, 1}));
}
void validateFloatBytes(size_t block, uint64_t offset, std::span<const std::byte> bytes) {
    if (block == 5) return; // uint2 ranges
    for (size_t i = 0; i < bytes.size(); i += 4) {
        if ((block == 4 || block == 8) && (offset + i) % 16 != 0 && (offset + i) % 16 != 8)
            continue; // count/padding are uint
        float value;
        std::memcpy(&value, bytes.data() + i, 4);
        require(std::isfinite(value),
                "Optimizer capture contains non-finite data (no sanitization)");
        if ((block == 1 || block == 7) && (offset + i) % 480 >= 240)
            require(value >= 0, "Optimizer capture contains a negative second moment");
    }
}
std::string quote(const char* value) {
    std::string out = "\"";
    for (; *value; ++value) {
        if (*value == '\\' || *value == '"') out += '\\';
        if (static_cast<unsigned char>(*value) < 32)
            out += '?';
        else
            out += *value;
    }
    return out + '"';
}
} // namespace
uint64_t optimizerHash(std::span<const std::byte> bytes, uint64_t seed) {
    for (auto byte : bytes)
        seed = (seed ^ std::to_integer<uint8_t>(byte)) * 1099511628211ull;
    return seed;
}
uint64_t optimizerShaderHash(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    require(bool(file), "Cannot open original fused SPIR-V");
    std::array<std::byte, 16384> chunk{};
    uint64_t hash = optimizerHashSeed, bytes = 0;
    while (file.read(reinterpret_cast<char*>(chunk.data()), chunk.size()) || file.gcount()) {
        hash = optimizerHash(std::span(chunk).first(size_t(file.gcount())), hash);
        bytes += uint64_t(file.gcount());
    }
    require(file.eof() && bytes >= 20 && bytes % 4 == 0, "Invalid original fused SPIR-V file");
    return hash;
}
uint64_t optimizerCaptureBytes(uint32_t count) {
    // count is uint32; this product is bounded below 2^43.
    return sizeof(OptimizerCaptureHeader) + uint64_t(count) * 1576ull;
}
uint64_t optimizerCaptureOffset(const OptimizerCaptureHeader& h, size_t block) {
    require(block < h.blocks.size(), "Optimizer capture block out of bounds");
    uint64_t offset = sizeof(h);
    for (size_t i = 0; i < block; ++i)
        offset += h.blocks[i].bytes;
    return offset;
}
void validateOptimizerCaptureHeader(const OptimizerCaptureHeader& h, uint64_t bytes,
                                    uint64_t budget) {
    require(std::endian::native == std::endian::little, "Optimizer capture requires little endian");
    require(h.magic == OptimizerCaptureHeader{}.magic && h.version == 1 &&
                h.headerBytes == sizeof(h) && h.cameraBytes == sizeof(h.camera) &&
                h.pushBytes == sizeof(h.push) && h.strides == optimizerCaptureStrides &&
                h.reserved == 0,
            "Optimizer capture version/ABI mismatch");
    require(bytes == optimizerCaptureBytes(h.gaussianCount) && bytes <= budget,
            "Optimizer capture truncated, has trailing bytes, or exceeds budget");
    require(h.capacity >= h.gaussianCount && h.push.gaussianCount == h.gaussianCount &&
                h.ordinaryFusedStep == 1 && h.push.optimizerEnabled == 1 &&
                h.completedIteration == h.push.trainingIteration + 1ull &&
                h.completedOptimizerStep == h.push.optimizerStep && h.push.activeSHDegree <= 3 &&
                h.push.activeSHDegree <= h.push.maxSHDegree && h.push.maxSHDegree <= 3 &&
                h.synthetic <= 1 && h.pixelMode <= 6 && h.ssimMode <= 2 && h.tileMode <= 2 &&
                h.compositeMode <= 1 && h.tensorFlags <= 15,
            "Optimizer capture mixed pre/post metadata or unsupported mode");
    require(h.push.width && h.push.height &&
                uint64_t(h.push.width) * h.push.height == h.push.pixelCount,
            "Optimizer capture invalid image extent");
    require(h.deviceName.back() == 0 && h.compiler.back() == 0 && h.shaderHash != 0,
            "Optimizer capture invalid identity");
    const auto cameraBytes = std::as_bytes(std::span{&h.camera, 1});
    validateFloatBytes(0, 0, cameraBytes);
    const auto* push = reinterpret_cast<const std::byte*>(&h.push);
    validateFloatBytes(0, 0,
                       {push + offsetof(TrainingPushConstants, positionLearningRate),
                        offsetof(TrainingPushConstants, optimizerStep) -
                            offsetof(TrainingPushConstants, positionLearningRate)});
    require(std::isfinite(h.push.pixelTo2DGSMinSubgroupUtilization),
            "Optimizer capture contains non-finite push constants");
    for (size_t i = 0; i < h.blocks.size(); ++i)
        require(h.blocks[i].bytes == uint64_t(h.gaussianCount) * h.strides[i],
                "Optimizer capture invalid block length");
    require(h.headerChecksum == headerHash(h), "Optimizer capture header checksum mismatch");
}
void readOptimizerCaptureChunk(const std::filesystem::path& path, const OptimizerCaptureHeader& h,
                               size_t block, uint64_t offset, std::span<std::byte> dest) {
    require(block < h.blocks.size() && offset <= h.blocks[block].bytes &&
                dest.size() <= h.blocks[block].bytes - offset,
            "Optimizer capture read overflow");
    std::ifstream file(path, std::ios::binary);
    file.seekg(std::streamoff(optimizerCaptureOffset(h, block) + offset));
    file.read(reinterpret_cast<char*>(dest.data()), std::streamsize(dest.size()));
    require(bool(file), "Optimizer capture block read failed");
}
OptimizerCaptureHeader readOptimizerCaptureHeader(const std::filesystem::path& path, bool verify) {
    OptimizerCaptureHeader h{};
    std::ifstream file(path, std::ios::binary);
    file.read(reinterpret_cast<char*>(&h), sizeof(h));
    require(bool(file), "Optimizer capture missing/truncated header");
    validateOptimizerCaptureHeader(h, std::filesystem::file_size(path));
    if (verify) {
        std::vector<std::byte> chunk(optimizerCaptureChunkBytes);
        for (size_t index = 0; index < h.blocks.size(); ++index) {
            const auto& block = h.blocks[index];
            uint64_t hash = optimizerHashSeed;
            for (uint64_t offset = 0; offset < block.bytes;) {
                const auto n = size_t(std::min<uint64_t>(chunk.size(), block.bytes - offset));
                file.read(reinterpret_cast<char*>(chunk.data()), n);
                require(bool(file), "Optimizer capture payload read failed");
                validateFloatBytes(index, offset, std::span(chunk).first(n));
                hash = optimizerHash(std::span(chunk).first(n), hash);
                offset += n;
            }
            require(hash == block.checksum, "Optimizer capture payload checksum mismatch");
        }
    }
    return h;
}
void writeOptimizerCapture(const std::filesystem::path& path, OptimizerCaptureHeader h,
                           const OptimizerCaptureSource& source) {
    for (size_t i = 0; i < h.blocks.size(); ++i) {
        h.blocks[i].bytes = uint64_t(h.gaussianCount) * h.strides[i];
        h.blocks[i].checksum = optimizerHashSeed;
    }
    h.headerChecksum = headerHash(h);
    validateOptimizerCaptureHeader(h, optimizerCaptureBytes(h.gaussianCount));
    // Exclusive creation: an existing result must never be truncated, even on a name race.
    FILE* raw = nullptr;
#ifdef _WIN32
    _wfopen_s(&raw, path.c_str(), L"wbx");
#else
    raw = std::fopen(path.c_str(), "wbx");
#endif
    require(raw != nullptr, "Optimizer capture exists or cannot be created");
    std::unique_ptr<FILE, decltype(&std::fclose)> file(raw, &std::fclose);
    auto write = [&](const void* data, size_t bytes) {
        require(std::fwrite(data, 1, bytes, raw) == bytes, "Optimizer capture disk write failed");
    };
    write(&h, sizeof(h));
    std::vector<std::byte> chunk(optimizerCaptureChunkBytes);
    for (size_t i = 0; i < h.blocks.size(); ++i) {
        for (uint64_t offset = 0; offset < h.blocks[i].bytes;) {
            auto dest = std::span(chunk).first(
                size_t(std::min<uint64_t>(chunk.size(), h.blocks[i].bytes - offset)));
            source(i, offset, dest);
            validateFloatBytes(i, offset, dest);
            h.blocks[i].checksum = optimizerHash(dest, h.blocks[i].checksum);
            write(dest.data(), dest.size());
            offset += dest.size();
        }
    }
    h.headerChecksum = headerHash(h);
    require(std::fseek(raw, 0, SEEK_SET) == 0, "Optimizer capture header seek failed");
    write(&h, sizeof(h));
    require(std::fflush(raw) == 0, "Optimizer capture flush failed");
    require(std::fclose(file.release()) == 0, "Optimizer capture close failed");
    require(std::filesystem::file_size(path) == optimizerCaptureBytes(h.gaussianCount),
            "Optimizer capture final size mismatch");
    // Check disk bytes, not just the buffers that were fed to the writer.
    (void)readOptimizerCaptureHeader(path);
}
void fillOptimizerCaptureIdentity(OptimizerCaptureHeader& h, vk::PhysicalDevice physical,
                                  const std::filesystem::path& shader) {
    vk::PhysicalDeviceIDProperties id;
    vk::PhysicalDeviceProperties2 properties;
    properties.pNext = &id;
    physical.getProperties2(&properties);
    const auto& p = properties.properties;
    h.vendorID = p.vendorID;
    h.deviceID = p.deviceID;
    h.driverVersion = p.driverVersion;
    h.apiVersion = p.apiVersion;
    std::copy(id.deviceUUID.begin(), id.deviceUUID.end(), h.deviceUUID.begin());
    h.deviceName.fill(0);
    h.compiler.fill(0);
    std::copy_n(p.deviceName.data(),
                std::min(std::strlen(p.deviceName.data()), h.deviceName.size() - 1),
                h.deviceName.data());
    constexpr char compiler[] = OPTIMIZER_SHADER_COMPILER;
    static_assert(
        sizeof(compiler) <= 128,
        "Expand/version the identity field rather than silently truncating compiler flags");
    std::copy_n(compiler, sizeof(compiler), h.compiler.data());
    h.shaderHash = optimizerShaderHash(shader);
}
bool optimizerCaptureIdentityMatches(const OptimizerCaptureHeader& a,
                                     const OptimizerCaptureHeader& b) {
    return a.vendorID == b.vendorID && a.deviceID == b.deviceID && a.deviceUUID == b.deviceUUID &&
           a.driverVersion == b.driverVersion && a.apiVersion == b.apiVersion &&
           a.shaderHash == b.shaderHash && a.compiler == b.compiler;
}
std::string optimizerCaptureIdentityJson(const OptimizerCaptureHeader& h) {
    std::ostringstream s;
    s << "\"device\":" << quote(h.deviceName.data()) << ",\"vendor_id\":" << h.vendorID
      << ",\"device_id\":" << h.deviceID << ",\"driver_version\":" << h.driverVersion
      << ",\"api_version\":" << h.apiVersion << ",\"compiler\":" << quote(h.compiler.data())
      << ",\"shader_fnv1a64\":\"" << h.shaderHash << "\",\"device_uuid\":\"";
    constexpr char digits[] = "0123456789abcdef";
    for (auto byte : h.deviceUUID)
        s << digits[byte >> 4] << digits[byte & 15];
    s << '"';
    return s.str();
}
} // namespace vulkan3DGS
