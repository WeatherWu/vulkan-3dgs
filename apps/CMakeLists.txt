# Vulkan+3DGS应用程序配置
project(vulkan_3dgs_app VERSION 1.0.0 LANGUAGES CXX)

# 创建示例应用程序
add_executable(vulkan_3dgs_app main.cpp)
set_target_properties(vulkan_3dgs_app PROPERTIES OUTPUT_NAME vulkan-3dgs)

find_package(imgui CONFIG REQUIRED)

# 链接核心库
target_link_libraries(vulkan_3dgs_app PRIVATE vulkan_3dgs_core)
target_link_libraries(vulkan_3dgs_app PRIVATE imgui::imgui)

# 安装配置
install(TARGETS vulkan_3dgs_app
    RUNTIME DESTINATION bin
)
