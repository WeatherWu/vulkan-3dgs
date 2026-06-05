cmake_minimum_required(VERSION 3.26)
project(vk_gs VERSION 1.0.0 LANGUAGES CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 编译器选项
if(MSVC)
    add_compile_options(/W3 /WX /utf-8)
elseif(MINGW)
    add_compile_options(-Wall -Wextra -Werror -static -static-libgcc -static-libstdc++)
else()
    add_compile_options(-Wall -Wextra -Werror)
endif()

# 项目配置
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# 设置不同构建类型的输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Release)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Debug)

# 包含FetchContent
include(FetchContent)

# 依赖配置
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)

# === 着色器编译配置 ===
set(SLANG_HINTS)
if(DEFINED ENV{VCPKG_ROOT})
    list(APPEND SLANG_HINTS "$ENV{VCPKG_ROOT}/installed/x64-windows/tools/shader-slang")
endif()

find_program(SLANGC_EXECUTABLE slangc HINTS ${SLANG_HINTS} REQUIRED)
message(STATUS "Found slangc: ${SLANGC_EXECUTABLE}")

set(SLANG_SHADER_SRC_DIR "${CMAKE_SOURCE_DIR}/shaders/gaussian_compute_shader/slang")
set(GRAPHICS_SHADER_SPV_DIR "${CMAKE_SOURCE_DIR}/shaders/gaussian_compute_shader/spv")
set(TRAINING_SHADER_SRC_DIR "${CMAKE_SOURCE_DIR}/shaders/training_shader/slang")
set(TRAINING_SHADER_SPV_DIR "${CMAKE_SOURCE_DIR}/shaders/training_shader/spv")

function(compile_slang_shader SHADER_SRC SHADER_STAGE OUTPUT_BASENAME)
    set(SHADER_SPV "${GRAPHICS_SHADER_SPV_DIR}/${OUTPUT_BASENAME}.spv")
    set(BUILD_SPV "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders/${OUTPUT_BASENAME}.spv")

    add_custom_command(
        OUTPUT ${SHADER_SPV}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${GRAPHICS_SHADER_SPV_DIR}"
        COMMAND ${SLANGC_EXECUTABLE}
            ${SHADER_SRC}
            -target spirv
            -profile sm_6_5
            -entry main
            -stage ${SHADER_STAGE}
            -matrix-layout-column-major
            -fvk-use-gl-layout
            -o ${SHADER_SPV}
        DEPENDS
            ${SHADER_SRC}
            "${SLANG_SHADER_SRC_DIR}/gaussian_common.slang"
        COMMENT "Compiling Slang shader ${SHADER_SRC} -> ${OUTPUT_BASENAME}.spv"
        VERBATIM
    )

    add_custom_command(
        OUTPUT ${BUILD_SPV}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SHADER_SPV} ${BUILD_SPV}
        DEPENDS ${SHADER_SPV}
        VERBATIM
    )

    set_source_files_properties(${SHADER_SPV} ${BUILD_SPV} PROPERTIES GENERATED TRUE)
    set(ALL_SHADER_FILES ${ALL_SHADER_FILES} ${BUILD_SPV} PARENT_SCOPE)
endfunction()

compile_slang_shader("${SLANG_SHADER_SRC_DIR}/gaussian_compute_shader.vert.slang" vertex "gaussian_compute_shader.vert")
compile_slang_shader("${SLANG_SHADER_SRC_DIR}/gaussian_compute_shader.frag.slang" fragment "gaussian_compute_shader.frag")
compile_slang_shader("${SLANG_SHADER_SRC_DIR}/radix_keygen.comp.slang" compute "radix_keygen.comp")

function(compile_training_shader SHADER_SRC OUTPUT_BASENAME)
    set(SHADER_SPV "${TRAINING_SHADER_SPV_DIR}/${OUTPUT_BASENAME}.spv")
    set(BUILD_SPV "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders/${OUTPUT_BASENAME}.spv")

    add_custom_command(
        OUTPUT ${SHADER_SPV}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${TRAINING_SHADER_SPV_DIR}"
        COMMAND ${SLANGC_EXECUTABLE}
            ${SHADER_SRC}
            -target spirv
            -profile sm_6_5
            -entry main
            -stage compute
            -matrix-layout-column-major
            -fvk-use-gl-layout
            -o ${SHADER_SPV}
        DEPENDS
            ${SHADER_SRC}
            "${TRAINING_SHADER_SRC_DIR}/train_common.slang"
            "${TRAINING_SHADER_SRC_DIR}/train_loss_common.slang"
        COMMENT "Compiling training Slang shader ${SHADER_SRC} -> ${OUTPUT_BASENAME}.spv"
        VERBATIM
    )

    add_custom_command(
        OUTPUT ${BUILD_SPV}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SHADER_SPV} ${BUILD_SPV}
        DEPENDS ${SHADER_SPV}
        VERBATIM
    )

    set_source_files_properties(${SHADER_SPV} ${BUILD_SPV} PROPERTIES GENERATED TRUE)
    set(ALL_SHADER_FILES ${ALL_SHADER_FILES} ${BUILD_SPV} PARENT_SCOPE)
endfunction()

compile_training_shader("${TRAINING_SHADER_SRC_DIR}/train_clear.comp.slang" "train_clear.comp")
compile_training_shader("${TRAINING_SHADER_SRC_DIR}/train_project.comp.slang" "train_project.comp")
compile_training_shader("${TRAINING_SHADER_SRC_DIR}/train_forward.comp.slang" "train_forward.comp")
compile_training_shader("${TRAINING_SHADER_SRC_DIR}/train_loss.comp.slang" "train_loss.comp")
compile_training_shader("${TRAINING_SHADER_SRC_DIR}/train_pack_render_buffer.comp.slang" "train_pack_render_buffer.comp")

add_custom_target(compile_shaders DEPENDS ${ALL_SHADER_FILES})

# 添加子目录
add_subdirectory(src)
add_subdirectory(apps)
if(CMAKE_CONFIGURATION_TYPES OR CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_subdirectory(sandbox)
endif()

# 确保在构建库之前编译着色器
# 注意：这里假设 src/CMakeLists.txt 中定义的目标名为 vk_gs_core
# 如果目标名不同，请相应调整
add_dependencies(vk_gs_core compile_shaders)

# 安装配置
install(DIRECTORY src/ DESTINATION include)

# 安装着色器文件（使用生成器表达式支持多配置）
install(DIRECTORY ${GRAPHICS_SHADER_SPV_DIR}/ 
    DESTINATION bin/$<CONFIG>/shaders
    FILES_MATCHING PATTERN "*.spv"
)

install(DIRECTORY ${TRAINING_SHADER_SPV_DIR}/
    DESTINATION bin/$<CONFIG>/shaders
    FILES_MATCHING PATTERN "*.spv"
)

install(TARGETS vk_gs_core # vk_gs_demo
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
