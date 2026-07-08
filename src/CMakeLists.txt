# Vulkan+3DGS核心库配置
project(vulkan_3dgs_core VERSION 1.0.0 LANGUAGES CXX)


# 查找Vulkan SDK
find_package(Vulkan REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(Stb QUIET)

set(IMGUI_FILE_DIALOG_DIR "${CMAKE_SOURCE_DIR}/third_party/ImGuiFileDialog")

# 查找GLFW
if(NOT glfw3_FOUND)
    FetchContent_Declare(
        glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG 3.3.8
    )
    FetchContent_MakeAvailable(glfw)
endif()

# 查找glm
if(NOT glm_FOUND)
    FetchContent_Declare(
        glm
        GIT_REPOSITORY https://github.com/g-truc/glm.git
        GIT_TAG 0.9.9.8
    )
    FetchContent_MakeAvailable(glm)
endif()

# 查找stb_image
if(NOT Stb_FOUND)
    FetchContent_Declare(
        stb
        GIT_REPOSITORY https://github.com/nothings/stb.git
        GIT_TAG 5736b15f7ea0ffb08dd38af21067c314d6a3aae9
    )
    FetchContent_MakeAvailable(stb)
endif()

# 创建核心库
add_library(vulkan_3dgs_core STATIC
    # Context 模块（Vulkan 基础环境）
    context/context.cpp
    context/device.cpp
    
    # Vulkan 资源封装（通用模块）
    vulkan/command_pool.cpp
    vulkan/buffer.cpp
    vulkan/shader.cpp
    vulkan/swapchain.cpp
    vulkan/compute_pipeline.cpp
    vulkan/vrdx_impl.cpp
    
    # Gaussian Splatting 渲染模块
    gaussian_renderer/gaussian_model.cpp
    gaussian_renderer/gaussian_renderer.cpp
    gaussian_renderer/renderpass.cpp
    gaussian_renderer/pipeline.cpp

    # Gaussian Training 训练模块
    gaussian_training/gaussian_training.cpp
    gaussian_training/training_buffers.cpp
    gaussian_training/training_pipeline.cpp
    gaussian_training/gaussian_forward_renderer.cpp
    gaussian_training/gaussian_backward_renderer.cpp
    gaussian_training/gaussian_densification_renderer.cpp
    gaussian_training/training_dataset.cpp

    # 工具模块
    utils/camera.cpp
    utils/stb_image_impl.cpp
    ${IMGUI_FILE_DIALOG_DIR}/ImGuiFileDialog.cpp
    
    # 应用框架与渲染管理
    window.cpp
    application.cpp
    renderer.cpp
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
    Vulkan::Vulkan
    glfw
)

target_link_libraries(vulkan_3dgs_core PRIVATE
    glm::glm
    imgui::imgui
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

# 安装配置
install(TARGETS vulkan_3dgs_core
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/ DESTINATION include)
