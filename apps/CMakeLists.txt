# Vulkan+3DGS应用程序配置
project(vk_gs_windows VERSION 1.0.0 LANGUAGES CXX)

# 创建示例应用程序
add_executable(vk_gs_windows main.cpp)

find_package(imgui CONFIG REQUIRED)

if(NOT imgui_FOUND)
    FetchContent_Declare(
        imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui.git
        GIT_TAG 1.91.91.91.9
    )
    FetchContent_MakeAvailable(imgui)
endif()

# 链接核心库
target_link_libraries(vk_gs_windows PRIVATE vk_gs_core)
target_link_libraries(vk_gs_windows PRIVATE imgui::imgui)

# 包含目录
target_include_directories(vk_gs_windows PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src/include
)

# 安装配置
install(TARGETS vk_gs_windows
    RUNTIME DESTINATION bin
)