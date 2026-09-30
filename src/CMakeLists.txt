# Vulkan+3DGS核心库配置
project(vulkan_3dgs_core VERSION 1.0.0 LANGUAGES CXX)


# 查找Vulkan SDK components from vcpkg explicitly.
find_package(VulkanHeaders CONFIG REQUIRED)
find_package(VulkanLoader CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(Stb REQUIRED)
find_package(Ktx CONFIG REQUIRED)
# find_package(COLMAP REQUIRED)

set(VULKAN_LINK_TARGETS Vulkan::Headers)
if(TARGET Vulkan::Loader)
    list(APPEND VULKAN_LINK_TARGETS Vulkan::Loader)
elseif(TARGET Vulkan::Vulkan)
    list(APPEND VULKAN_LINK_TARGETS Vulkan::Vulkan)
else()
    message(FATAL_ERROR "VulkanLoader package did not provide Vulkan::Loader or Vulkan::Vulkan")
endif()

set(IMGUI_FILE_DIALOG_DIR "${CMAKE_SOURCE_DIR}/third_party/ImGuiFileDialog")

# 创建核心库
add_library(vulkan_3dgs_core STATIC
    # Context 模块（Vulkan 基础环境）
    context/context.cpp
    context/device.cpp
    context/device_selection.cpp
    
    # Vulkan 资源封装（通用模块）
    vulkan/command_pool.cpp
    vulkan/buffer.cpp
    vulkan/shader.cpp
    vulkan/swapchain.cpp
    vulkan/compute_pipeline.cpp
    vulkan/vrdx_impl.cpp
    
    # Gaussian Splatting 渲染模块
    graphics/gaussian_model.cpp
    graphics/gaussian_renderer.cpp
    graphics/graphics_splat_sorter.cpp
    graphics/graphics_splat_resources.cpp
    graphics/graphics_pipeline_set.cpp
    graphics/graphics_frame_runtime.cpp
    graphics/graphics_command_recorder.cpp
    graphics/renderpass.cpp
    graphics/pipeline.cpp

    # Gaussian Training 训练核心
    training/core/gaussian_training.cpp
    training/core/training_validation_service.cpp
    training/core/training_profiling_service.cpp
    training/core/training_frame_scheduler.cpp
    training/core/training_model_io.cpp
    training/core/training_image_runtime.cpp
    training/core/training_step_executor.cpp
    training/core/training_buffers.cpp
    training/core/training_pipeline.cpp
    training/core/gaussian_forward_renderer.cpp
    training/core/gaussian_backward_renderer.cpp
    training/core/backward/backward_command_utils.cpp
    training/core/backward/backward_loss_pass.cpp
    training/core/backward/pixel_to_2dgs_dispatcher.cpp
    training/core/backward/projection_optimizer_pass.cpp
    training/core/backward/backward_validation_pass.cpp
    training/core/gaussian_densification_renderer.cpp
    training/core/training_dataset.cpp
    training/cache/device_image_cache.cpp

    # 通用图像加载与缓存模块
    image/image_decoder.cpp
    image/image_disk_cache.cpp
    image/image_streamer.cpp

    # 查看器相机与通用工具
    viewer/camera.cpp
    viewer/camera_controller.cpp
    utils/system_memory.cpp
    utils/stb_image_impl.cpp
    ${IMGUI_FILE_DIALOG_DIR}/ImGuiFileDialog.cpp
    
    # 训练应用层与 UI
    training/app/training_controller.cpp
    training/app/training_settings.cpp
    training/ui/training_command_panel.cpp
    training/ui/training_diagnostics_panel.cpp
    training/ui/training_settings_panel.cpp
    training/ui/training_file_dialogs.cpp
    training/ui/training_panel.cpp
    viewer/viewer_controller.cpp
    viewer/viewer_panel.cpp
    viewer/viewer_settings.cpp

    # 应用组合根与两条渲染路径的共同接口
    app/window.cpp
    app/application.cpp
    render/renderer.cpp
)

# 包含目录
target_include_directories(vulkan_3dgs_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/third_party/vulkan_radix_sort/include
)

# 包含目录（私有依赖）
target_include_directories(vulkan_3dgs_core PRIVATE
    ${Stb_INCLUDE_DIR}
    ${Stb_SOURCE_DIR}
    ${glm_SOURCE_DIR}
    ${IMGUI_FILE_DIALOG_DIR}
)

# 链接库
target_link_libraries(vulkan_3dgs_core PUBLIC
    ${VULKAN_LINK_TARGETS}
    glfw
)

target_link_libraries(vulkan_3dgs_core PRIVATE
    glm::glm
    imgui::imgui
    KTX::ktx
    # colmap::colmap_image
    # colmap::colmap_scene
    # colmap::colmap_sensor
)

# 编译器选项
if(MSVC)
    add_compile_options(/W4 /WX /utf-8)
    add_compile_options(/Zc:__cplusplus)
else()
    add_compile_options(-Wall -Wextra -Werror)
endif()

# 编译器定义
target_compile_definitions(vulkan_3dgs_core PRIVATE
    GLFW_INCLUDE_VULKAN
    GLM_FORCE_DEPTH_ZERO_TO_ONE
    NOMINMAX
    VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
)

if(MSVC)
    set_source_files_properties(vulkan/vrdx_impl.cpp PROPERTIES COMPILE_OPTIONS "/W0")
    set_source_files_properties(${IMGUI_FILE_DIALOG_DIR}/ImGuiFileDialog.cpp PROPERTIES COMPILE_OPTIONS "/W0")
endif()

# Opt-in quality gates for modules introduced by the staged architecture work.
# These targets are intentionally separate from normal Debug/Release builds.
include("${CMAKE_SOURCE_DIR}/cmake/CodeQuality.cmake")

# 安装配置
install(TARGETS vulkan_3dgs_core
    ARCHIVE DESTINATION lib
)

install(DIRECTORY app context graphics image render training utils viewer vulkan
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)
