#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <filesystem>
#include <stdexcept>

#include "logger.hpp"

#ifdef _WIN32
    #include <windows.h>
#else
    #include <unistd.h>
    #include <limits.h>
#endif

namespace vk_gs {

class FileUtils {
public:
    // 获取可执行文件所在目录
    static std::string getExecutableDir() {
        #ifdef _WIN32
            char path[MAX_PATH];
            GetModuleFileNameA(NULL, path, MAX_PATH);
            std::filesystem::path exePath(path);
            return exePath.parent_path().string();
        #else
            char result[PATH_MAX];
            ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
            if (count != -1) {
                std::filesystem::path exePath(std::string(result, count));
                return exePath.parent_path().string();
            }
            return ".";
        #endif
    }
    
    // 解析相对路径为绝对路径（相对于可执行文件目录）
    static std::string resolvePath(const std::string& relativePath) {
        std::filesystem::path relPath(relativePath);
        
        // 如果已经是绝对路径，直接返回
        if (relPath.is_absolute()) {
            return relativePath;
        }
        
        // 相对于可执行文件目录
        std::string exeDir = getExecutableDir();
        std::filesystem::path fullPath = std::filesystem::path(exeDir) / relPath;
        
        // 规范化路径（处理../等）
        return std::filesystem::weakly_canonical(fullPath).string();
    }
    
    // 检查文件是否存在
    static bool fileExists(const std::string& path) {
        return std::filesystem::exists(path);
    }
    
    // 读取二进制文件
    static std::vector<char> readBinaryFile(const std::string& filename) {
        // 解析文件路径
        std::string resolvedPath = resolvePath(filename);
        
        std::ifstream file(resolvedPath, std::ios::ate | std::ios::binary);
    
        if (!file.is_open()) {
            LOG_ERROR("Failed to open file: {}", filename);
            throw std::runtime_error("Failed to open file: " + filename);
        }
        
        size_t file_size = static_cast<size_t>(file.tellg());
        std::vector<char> buffer(file_size);
        
        file.seekg(0);
        file.read(buffer.data(), static_cast<std::streamsize>(file_size));
        file.close();
        
        LOG_DEBUG("Loaded shader file: {} ({} bytes)", filename, file_size);
        return buffer;
    }

    template <typename T>
    static T readBinaryValue(std::istream& stream) {
        T value{};
        stream.read(reinterpret_cast<char*>(&value), sizeof(T));
        if (!stream) {
            throw std::runtime_error("Unexpected end of binary file");
        }
        return value;
    }

    static std::string readNullTerminatedString(std::istream& stream) {
        std::string value;
        char ch = '\0';
        while (stream.get(ch)) {
            if (ch == '\0') {
                return value;
            }
            value.push_back(ch);
        }

        throw std::runtime_error("Unexpected end of binary string");
    }
};

} // namespace vk_gs
