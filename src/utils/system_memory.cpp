#include "utils/system_memory.hpp"

#ifdef _WIN32
#include <windows.h>
#else
#include <algorithm>
#include <fstream>
#include <limits>
#include <string>
#endif

namespace vulkan3DGS {

#ifndef _WIN32
namespace {

uint64_t readUnsignedFile(const char* path) {
    std::ifstream file(path);
    std::string value;
    if (!(file >> value) || value == "max") {
        return std::numeric_limits<uint64_t>::max();
    }
    try {
        return std::stoull(value);
    } catch (...) {
        return std::numeric_limits<uint64_t>::max();
    }
}

uint64_t cgroupAvailableMemoryBytes() {
    const uint64_t v2Limit = readUnsignedFile("/sys/fs/cgroup/memory.max");
    const uint64_t v2Usage = readUnsignedFile("/sys/fs/cgroup/memory.current");
    if (v2Limit != std::numeric_limits<uint64_t>::max() &&
        v2Usage != std::numeric_limits<uint64_t>::max()) {
        return v2Limit > v2Usage ? v2Limit - v2Usage : 0u;
    }

    const uint64_t v1Limit = readUnsignedFile("/sys/fs/cgroup/memory/memory.limit_in_bytes");
    const uint64_t v1Usage = readUnsignedFile("/sys/fs/cgroup/memory/memory.usage_in_bytes");
    if (v1Limit != std::numeric_limits<uint64_t>::max() &&
        v1Usage != std::numeric_limits<uint64_t>::max()) {
        return v1Limit > v1Usage ? v1Limit - v1Usage : 0u;
    }
    return std::numeric_limits<uint64_t>::max();
}

} // namespace
#endif

uint64_t availableSystemMemoryBytes() {
#ifdef _WIN32
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status)) {
        return status.ullAvailPhys;
    }
#else
    std::ifstream memInfo("/proc/meminfo");
    std::string key;
    uint64_t valueKiB = 0;
    std::string unit;
    while (memInfo >> key >> valueKiB >> unit) {
        if (key == "MemAvailable:") {
            const uint64_t systemAvailable = valueKiB * 1024ull;
            return std::min(systemAvailable, cgroupAvailableMemoryBytes());
        }
    }
#endif
    return 512ull * 1024ull * 1024ull;
}

} // namespace vulkan3DGS
