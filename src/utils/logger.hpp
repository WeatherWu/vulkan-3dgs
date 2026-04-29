#pragma once

// 在Windows上，需要在包含windows.h之前处理ERROR宏冲突
#ifdef _WIN32
    #ifdef ERROR
        #define VK_GS_LOGGER_ERROR_WAS_DEFINED
    #endif
#endif

#include <string>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <memory>
#include <utility>

namespace vk_gs {

enum class LogLevel {
    DEBUG_VKGS,
    INFO_VKGS,
    WARN_VKGS,
    ERROR_VKGS
};

class Logger {
public:
    static Logger& get_instance() {
        static Logger instance;
        return instance;
    }
    
    void set_level(LogLevel level) { level_ = level; }
    
    template<typename... Args>
    void log(LogLevel level, const std::string& format, Args&&... args) {
        if (level < level_) return;
        
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        
        std::stringstream ss;
        
        // 使用安全的本地时间获取方式
        std::tm tm_info;
        #ifdef _WIN32
            localtime_s(&tm_info, &time_t);
        #else
            localtime_r(&time_t, &tm_info);
        #endif
        
        // 使用stringstream进行时间格式化
        ss << "[" << std::setfill('0')
           << std::setw(4) << (tm_info.tm_year + 1900) << "-"
           << std::setw(2) << (tm_info.tm_mon + 1) << "-"
           << std::setw(2) << tm_info.tm_mday << " "
           << std::setw(2) << tm_info.tm_hour << ":"
           << std::setw(2) << tm_info.tm_min << ":"
           << std::setw(2) << tm_info.tm_sec << "] ";
        
        switch (level) {
            case LogLevel::DEBUG_VKGS: ss << "[DEBUG] "; break;
            case LogLevel::INFO_VKGS: ss << "[INFO]  "; break;
            case LogLevel::WARN_VKGS: ss << "[WARN]  "; break;
            case LogLevel::ERROR_VKGS: ss << "[ERROR] "; break;
        }
        
        // 使用可变参数模板展开进行格式化
        format_impl(ss, format, std::forward<Args>(args)...);
        
        ss << std::endl;
        std::cout << ss.str();
    }
    
private:
    Logger() : level_(LogLevel::INFO_VKGS) {}
    
    // 基础情况：没有参数时直接输出格式字符串
    void format_impl(std::stringstream& ss, const std::string& format) {
        ss << format;
    }
    
    // 递归展开可变参数模板
    template<typename T, typename... Args>
    void format_impl(std::stringstream& ss, const std::string& format, T&& first, Args&&... rest) {
        size_t pos = format.find("{}");
        if (pos != std::string::npos) {
            // 输出{}之前的内容
            ss << format.substr(0, pos);
            // 输出参数
            ss << std::forward<T>(first);
            // 递归处理剩余部分和参数
            format_impl(ss, format.substr(pos + 2), std::forward<Args>(rest)...);
        } else {
            // 没有找到{}，直接输出整个格式字符串
            ss << format;
        }
    }
    
    LogLevel level_;
};

// 便捷宏
#define LOG_DEBUG(format, ...) vk_gs::Logger::get_instance().log(vk_gs::LogLevel::DEBUG_VKGS, format, ##__VA_ARGS__)
#define LOG_INFO(format, ...) vk_gs::Logger::get_instance().log(vk_gs::LogLevel::INFO_VKGS, format, ##__VA_ARGS__)
#define LOG_WARN(format, ...) vk_gs::Logger::get_instance().log(vk_gs::LogLevel::WARN_VKGS, format, ##__VA_ARGS__)
#define LOG_ERROR(format, ...) vk_gs::Logger::get_instance().log(vk_gs::LogLevel::ERROR_VKGS, format, ##__VA_ARGS__)

} // namespace vk_gs