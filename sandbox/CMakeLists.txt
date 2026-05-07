# Vulkan+3DGS应用程序配置
project(vk_gs_sandbox VERSION 1.0.0 LANGUAGES CXX)

# 创建示例应用程序
add_executable(vk_gs_sandbox sandbox.cpp)

# 链接核心库
target_link_libraries(vk_gs_sandbox PRIVATE vk_gs_core)
    
# 包含目录
target_include_directories(vk_gs_sandbox PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src/include
)

# 安装配置
install(TARGETS vk_gs_sandbox
    RUNTIME DESTINATION bin
)