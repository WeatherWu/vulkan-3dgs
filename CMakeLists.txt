cmake_minimum_required(VERSION 3.26)
project(vulkan_3dgs VERSION 1.0.0 LANGUAGES C CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 编译器选项
if(MSVC)
    add_compile_options(/W3 /WX /utf-8)
elseif(MINGW)
    add_compile_options(-Wall -Wextra -static -static-libgcc -static-libstdc++)
else()
    add_compile_options(-Wall -Wextra)
endif()

# 项目配置
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# 设置不同构建类型的输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Release)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Debug)

# 依赖配置
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)

# === 着色器编译配置 ===
set(SLANG_HINTS)
if(DEFINED ENV{VCPKG_ROOT})
    if(DEFINED VCPKG_TARGET_TRIPLET)
        list(APPEND SLANG_HINTS "$ENV{VCPKG_ROOT}/installed/${VCPKG_TARGET_TRIPLET}/tools/shader-slang")
    endif()
    if(DEFINED ENV{VCPKG_DEFAULT_TRIPLET})
        list(APPEND SLANG_HINTS "$ENV{VCPKG_ROOT}/installed/$ENV{VCPKG_DEFAULT_TRIPLET}/tools/shader-slang")
    endif()
    list(APPEND SLANG_HINTS
        "$ENV{VCPKG_ROOT}/installed/x64-windows/tools/shader-slang"
        "$ENV{VCPKG_ROOT}/installed/x64-linux/tools/shader-slang")
endif()

find_program(SLANGC_EXECUTABLE slangc HINTS ${SLANG_HINTS} REQUIRED)
message(STATUS "Found slangc: ${SLANGC_EXECUTABLE}")

set(SLANG_SHADER_SRC_DIR "${CMAKE_SOURCE_DIR}/shaders/gaussian_compute_shader/slang")
set(GRAPHICS_SHADER_SPV_DIR "${CMAKE_SOURCE_DIR}/shaders/gaussian_compute_shader/spv")
set(TRAINING_SHADER_SRC_DIR "${CMAKE_SOURCE_DIR}/shaders/training_shader/slang")
set(TRAINING_SHADER_COMMON_DIR "${TRAINING_SHADER_SRC_DIR}/common")
set(TRAINING_SHADER_PASSES_DIR "${TRAINING_SHADER_SRC_DIR}/passes")
set(TRAINING_SHADER_UINT_RADIUS_DIR "${TRAINING_SHADER_SRC_DIR}/uint_radius")
set(TRAINING_SHADER_SPV_DIR "${CMAKE_SOURCE_DIR}/shaders/training_shader/spv")
set(TRAINING_SHADER_COMMON_DEPS
    "${TRAINING_SHADER_COMMON_DIR}/train_common.slang"
    "${TRAINING_SHADER_COMMON_DIR}/train_loss_common.slang"
    "${TRAINING_SHADER_COMMON_DIR}/train_forward_common.slang"
    "${TRAINING_SHADER_COMMON_DIR}/train_forward_tile_common.slang"
    "${TRAINING_SHADER_COMMON_DIR}/train_forward_project_common.slang"
    "${TRAINING_SHADER_COMMON_DIR}/train_backward_types.slang"
    "${TRAINING_SHADER_COMMON_DIR}/train_backward_common.slang"
    "${TRAINING_SHADER_COMMON_DIR}/train_backward_project_common.slang"
)

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
            ${TRAINING_SHADER_COMMON_DEPS}
            ${ARGN}
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

compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_clear.comp.slang" "train_clear.comp")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_project.comp.slang" "train_project.comp")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_forward_project.comp.slang" "train_forward_project.comp")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_forward_tile_clear.comp.slang" "train_forward_tile_clear.comp")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_forward_tile_count.comp.slang" "train_forward_tile_count.comp")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_forward_tile_prefix.comp.slang" "train_forward_tile_prefix.comp")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_forward_tile_emit.comp.slang" "train_forward_tile_emit.comp")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_forward_tile_gather_high.comp.slang" "train_forward_tile_gather_high.comp")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_forward_tile_gather_items.comp.slang" "train_forward_tile_gather_items.comp")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_forward_tile_sort.comp.slang" "train_forward_tile_sort.comp")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_forward.comp.slang" "train_forward.comp")
compile_training_shader("${TRAINING_SHADER_UINT_RADIUS_DIR}/train_forward_uint_radius.comp.slang"
                        "train_forward_uint_radius.comp"
                        "${TRAINING_SHADER_PASSES_DIR}/train_forward.comp.slang")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_loss.comp.slang" "train_loss.comp")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_backward_clear.comp.slang" "train_backward_clear.comp")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_backward_loss_to_pixel.comp.slang" "train_backward_loss_to_pixel.comp")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_backward_pixel_to_2dgs.comp.slang" "train_backward_pixel_to_2dgs.comp")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_backward_2dgs_to_3dgs.comp.slang" "train_backward_2dgs_to_3dgs.comp")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_optimizer.comp.slang" "train_optimizer.comp")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_densify_clear.comp.slang" "train_densify_clear.comp")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_densify_prune.comp.slang" "train_densify_prune.comp")
compile_training_shader("${TRAINING_SHADER_UINT_RADIUS_DIR}/train_densify_prune_uint_radius.comp.slang"
                        "train_densify_prune_uint_radius.comp"
                        "${TRAINING_SHADER_PASSES_DIR}/train_densify_prune.comp.slang")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_densify_finalize_prune.comp.slang"
                        "train_densify_finalize_prune.comp")
compile_training_shader("${TRAINING_SHADER_UINT_RADIUS_DIR}/train_densify_finalize_prune_uint_radius.comp.slang"
                        "train_densify_finalize_prune_uint_radius.comp"
                        "${TRAINING_SHADER_PASSES_DIR}/train_densify_finalize_prune.comp.slang")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_opacity_reset.comp.slang" "train_opacity_reset.comp")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_pack_render_buffer.comp.slang" "train_pack_render_buffer.comp")

add_custom_target(compile_shaders DEPENDS ${ALL_SHADER_FILES})

# 添加子目录
add_subdirectory(src)
add_subdirectory(apps)

# 确保在构建库之前编译着色器
# 注意：这里假设 src/CMakeLists.txt 中定义的目标名为 vulkan_3dgs_core
# 如果目标名不同，请相应调整
add_dependencies(vulkan_3dgs_core compile_shaders)

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

install(TARGETS vulkan_3dgs_core # vulkan_3dgs_demo
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
