# Vulkan+3DGS应用程序配置
project(vulkan_3dgs_app VERSION 1.0.0 LANGUAGES CXX)

# 创建示例应用程序
add_executable(vulkan_3dgs_app main.cpp)
set_target_properties(vulkan_3dgs_app PROPERTIES OUTPUT_NAME vulkan-3dgs)

find_package(imgui CONFIG REQUIRED)

if(NOT imgui_FOUND)
    FetchContent_Declare(
        imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui.git
        GIT_TAG 1.91.9
    )
    FetchContent_MakeAvailable(imgui)
endif()

# 链接核心库
target_link_libraries(vulkan_3dgs_app PRIVATE vulkan_3dgs_core)
target_link_libraries(vulkan_3dgs_app PRIVATE imgui::imgui)

# 包含目录
target_include_directories(vulkan_3dgs_app PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src/include
)

# 安装配置
install(TARGETS vulkan_3dgs_app
    RUNTIME DESTINATION bin
)
