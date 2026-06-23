# Vulkan+3DGS应用程序配置
project(vk_gs_sandbox VERSION 1.0.0 LANGUAGES CXX)

# 创建示例应用程序。默认 ALL_BUILD 不直接包含它，下面用 Debug-only 聚合目标控制默认构建。
add_executable(vk_gs_sandbox EXCLUDE_FROM_ALL sandbox.cpp)
add_executable(vk_gs_dataset_probe EXCLUDE_FROM_ALL dataset_probe.cpp)

# Visual Studio / multi-config 生成器不能按配置改变 EXCLUDE_FROM_ALL。
# 这里让 sandbox 本体始终退出 ALL_BUILD，然后让这个 ALL 目标只在 Debug
# 配置下显式构建 sandbox。不要用 DEPENDS vk_gs_sandbox，否则 VS 会在
# Release 下也生成 ProjectReference 并构建 sandbox。
add_custom_target(vk_gs_sandbox_debug ALL
    COMMAND "$<$<CONFIG:Debug>:${CMAKE_COMMAND};--build;${CMAKE_BINARY_DIR};--config;$<CONFIG>;--target;vk_gs_sandbox>"
    COMMAND_EXPAND_LISTS
)

set_target_properties(vk_gs_sandbox PROPERTIES
    EXCLUDE_FROM_DEFAULT_BUILD_RELEASE TRUE
    EXCLUDE_FROM_DEFAULT_BUILD_RELWITHDEBINFO TRUE
    EXCLUDE_FROM_DEFAULT_BUILD_MINSIZEREL TRUE
)

# 链接核心库
target_link_libraries(vk_gs_sandbox PRIVATE vk_gs_core)
target_link_libraries(vk_gs_dataset_probe PRIVATE vk_gs_core)
    
# 包含目录
target_include_directories(vk_gs_sandbox PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src/include
)

target_include_directories(vk_gs_dataset_probe PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src/include
)

# 安装配置
install(TARGETS vk_gs_sandbox
    CONFIGURATIONS Debug
   RUNTIME DESTINATION bin
)
