cmake_minimum_required(VERSION 3.26)
project(vulkan_3dgs VERSION 1.0.0 LANGUAGES C CXX)

include(CTest)

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
execute_process(COMMAND "${SLANGC_EXECUTABLE}" -version
    OUTPUT_VARIABLE OPTIMIZER_SLANG_VERSION ERROR_VARIABLE OPTIMIZER_SLANG_VERSION_ERROR
    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_STRIP_TRAILING_WHITESPACE)
if(NOT OPTIMIZER_SLANG_VERSION)
    set(OPTIMIZER_SLANG_VERSION "${OPTIMIZER_SLANG_VERSION_ERROR}")
endif()
string(REGEX REPLACE "[\r\n]+" " " OPTIMIZER_SLANG_VERSION "${OPTIMIZER_SLANG_VERSION}")

option(VULKAN_3DGS_TRAINING_SHADER_DEBUG_INFO
       "Embed Slang debug information in training SPIR-V for shader profiling" OFF)
set(TRAINING_SHADER_DEBUG_FLAGS)
if(VULKAN_3DGS_TRAINING_SHADER_DEBUG_INFO)
    # Keep the existing optimization settings; debug information is independent
    # of the C++ build configuration and does not request -O0.
    list(APPEND TRAINING_SHADER_DEBUG_FLAGS -g2)
endif()
message(STATUS "Training shader debug info: ${VULKAN_3DGS_TRAINING_SHADER_DEBUG_INFO}")

# Make ON/OFF changes rebuild shaders even with timestamp-based generators.
# file(CONFIGURE) only updates this dependency when its contents change.
set(TRAINING_SHADER_DEBUG_INFO_STAMP "${CMAKE_CURRENT_BINARY_DIR}/training_shader_debug_info.txt")
file(CONFIGURE OUTPUT "${TRAINING_SHADER_DEBUG_INFO_STAMP}"
     CONTENT "debug_info=@VULKAN_3DGS_TRAINING_SHADER_DEBUG_INFO@\n" @ONLY)

set(SLANG_SHADER_SRC_DIR "${CMAKE_SOURCE_DIR}/shaders/gaussian_compute_shader/slang")
set(GRAPHICS_SHADER_SPV_DIR "${CMAKE_SOURCE_DIR}/shaders/gaussian_compute_shader/spv")
set(TRAINING_SHADER_SRC_DIR "${CMAKE_SOURCE_DIR}/shaders/training_shader/slang")
set(TRAINING_SHADER_COMMON_DIR "${TRAINING_SHADER_SRC_DIR}/common")
set(TRAINING_SHADER_PASSES_DIR "${TRAINING_SHADER_SRC_DIR}/passes")
set(TRAINING_SHADER_VALIDATION_DIR "${TRAINING_SHADER_SRC_DIR}/validation")
set(TRAINING_SHADER_UINT_RADIUS_DIR "${TRAINING_SHADER_SRC_DIR}/uint_radius")
set(TRAINING_SHADER_SPV_DIR "${CMAKE_SOURCE_DIR}/shaders/training_shader/spv")
set(TRAINING_SHADER_COMMON_DEPS
    "${TRAINING_SHADER_COMMON_DIR}/train_common.slang"
    "${TRAINING_SHADER_COMMON_DIR}/train_loss_common.slang"
    "${TRAINING_SHADER_COMMON_DIR}/train_ssim_separable.slang"
    "${TRAINING_SHADER_COMMON_DIR}/train_checkpoint_layout.slang"
    "${TRAINING_SHADER_COMMON_DIR}/train_pixel_culling.slang"
    "${TRAINING_SHADER_COMMON_DIR}/train_target_common.slang"
    "${TRAINING_SHADER_COMMON_DIR}/train_forward_common.slang"
    "${TRAINING_SHADER_COMMON_DIR}/train_forward_tile_common.slang"
    "${TRAINING_SHADER_COMMON_DIR}/train_forward_project_common.slang"
    "${TRAINING_SHADER_COMMON_DIR}/train_backward_types.slang"
    "${TRAINING_SHADER_COMMON_DIR}/train_backward_common.slang"
    "${TRAINING_SHADER_COMMON_DIR}/train_backward_project_common.slang"
    "${TRAINING_SHADER_COMMON_DIR}/train_optimizer_common.slang"
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
    # RowEmpty's interval ledger requires NoContraction. Some Slang SPIR-V backends
    # ignore even global precise. Use the GLSL path only for RowEmpty outputs;
    # inspect the emitted SPIR-V at runtime and leave every old shader alone.
    set(TRAINING_SHADER_NUMERIC_FLAGS)
    set(TRAINING_SHADER_NUMERIC_DEPENDENCY)
    if(OUTPUT_BASENAME STREQUAL "train_row_empty_census.comp" OR
       OUTPUT_BASENAME STREQUAL "train_backward_vksplat_tensor_row_empty.comp" OR
       OUTPUT_BASENAME STREQUAL "train_backward_tensor_row_empty_test.comp")
        set(TRAINING_SHADER_NUMERIC_FLAGS -fp-mode precise -emit-spirv-via-glsl)
        set(TRAINING_SHADER_NUMERIC_DEPENDENCY "${CMAKE_CURRENT_BINARY_DIR}/training_row_empty_numeric_mode.txt")
        file(CONFIGURE OUTPUT "${TRAINING_SHADER_NUMERIC_DEPENDENCY}"
             CONTENT "flags=@TRAINING_SHADER_NUMERIC_FLAGS@\n" @ONLY)
    endif()

    add_custom_command(
        OUTPUT ${SHADER_SPV}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${TRAINING_SHADER_SPV_DIR}"
        COMMAND ${SLANGC_EXECUTABLE}
            ${SHADER_SRC}
            -target spirv
            -profile sm_6_5
            -entry main
            -stage compute
            ${TRAINING_SHADER_DEBUG_FLAGS}
            ${TRAINING_SHADER_NUMERIC_FLAGS}
            -matrix-layout-column-major
            -fvk-use-gl-layout
            -o ${SHADER_SPV}
        DEPENDS
            ${SHADER_SRC}
            ${TRAINING_SHADER_COMMON_DEPS}
            "${TRAINING_SHADER_DEBUG_INFO_STAMP}"
            ${TRAINING_SHADER_NUMERIC_DEPENDENCY}
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

function(compile_training_tile_shader SHADER_SRC OUTPUT_BASENAME TILE_SIZE)
    set(SHADER_SPV "${TRAINING_SHADER_SPV_DIR}/${OUTPUT_BASENAME}.spv")
    set(BUILD_SPV "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders/${OUTPUT_BASENAME}.spv")

    add_custom_command(
        OUTPUT ${SHADER_SPV}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${TRAINING_SHADER_SPV_DIR}"
        COMMAND ${SLANGC_EXECUTABLE}
            ${SHADER_SRC}
            -DTRAIN_TILE_SIZE=${TILE_SIZE}
            -target spirv
            -profile sm_6_5
            -entry main
            -stage compute
            ${TRAINING_SHADER_DEBUG_FLAGS}
            -matrix-layout-column-major
            -fvk-use-gl-layout
            -o ${SHADER_SPV}
        DEPENDS
            ${SHADER_SRC}
            ${TRAINING_SHADER_COMMON_DEPS}
            "${TRAINING_SHADER_DEBUG_INFO_STAMP}"
            ${ARGN}
        COMMENT "Compiling ${TILE_SIZE}x${TILE_SIZE} training shader ${SHADER_SRC} -> ${OUTPUT_BASENAME}.spv"
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

function(compile_training_adaptive_tile_shader SHADER_SRC OUTPUT_BASENAME TILE_SIZE)
    cmake_parse_arguments(ADAPTIVE "" "CHILD_TENSOR_BATCH" "" ${ARGN})
    set(ADAPTIVE_DEFINES)
    if(ADAPTIVE_CHILD_TENSOR_BATCH)
        list(APPEND ADAPTIVE_DEFINES "-DTRAIN_TENSOR_CHILD_BATCH=${ADAPTIVE_CHILD_TENSOR_BATCH}")
    endif()
    set(SHADER_SPV "${TRAINING_SHADER_SPV_DIR}/${OUTPUT_BASENAME}.spv")
    set(BUILD_SPV "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders/${OUTPUT_BASENAME}.spv")

    add_custom_command(
        OUTPUT ${SHADER_SPV}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${TRAINING_SHADER_SPV_DIR}"
        COMMAND ${SLANGC_EXECUTABLE}
            ${SHADER_SRC}
            -DTRAIN_TILE_SIZE=${TILE_SIZE}
            -DTRAIN_ADAPTIVE_TILES=1
            ${ADAPTIVE_DEFINES}
            -target spirv
            -profile sm_6_5
            -entry main
            -stage compute
            ${TRAINING_SHADER_DEBUG_FLAGS}
            -matrix-layout-column-major
            -fvk-use-gl-layout
            -o ${SHADER_SPV}
        DEPENDS
            ${SHADER_SRC}
            ${TRAINING_SHADER_COMMON_DEPS}
            "${TRAINING_SHADER_DEBUG_INFO_STAMP}"
            ${ADAPTIVE_UNPARSED_ARGUMENTS}
        COMMENT "Compiling adaptive ${TILE_SIZE}x${TILE_SIZE} training shader ${SHADER_SRC} -> ${OUTPUT_BASENAME}.spv"
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

compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_project.comp.slang" "train_project.comp")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_forward_project.comp.slang" "train_forward_project.comp")
foreach(checkpoint_pass IN ITEMS layout prefix state tasks)
    compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_checkpoint_${checkpoint_pass}.comp.slang"
                            "train_checkpoint_${checkpoint_pass}.comp")
endforeach()
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_backward_checkpoint_warp.comp.slang"
                        "train_backward_checkpoint_warp.comp")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_forward_tile_clear.comp.slang" "train_forward_tile_clear.comp")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_morton_reorder.comp.slang" "train_morton_reorder.comp")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_forward_tile_count.comp.slang" "train_forward_tile_count.comp")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_forward_gaussian_prefix_ranges.comp.slang"
                        "train_forward_gaussian_prefix_ranges.comp")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_forward_gaussian_prefix_scratch.comp.slang"
                        "train_forward_gaussian_prefix_scratch.comp")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_forward_gaussian_prefix_top.comp.slang"
                        "train_forward_gaussian_prefix_top.comp")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_forward_gaussian_prefix_add_scratch.comp.slang"
                        "train_forward_gaussian_prefix_add_scratch.comp")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_forward_gaussian_prefix_add_ranges.comp.slang"
                        "train_forward_gaussian_prefix_add_ranges.comp")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_forward_tile_emit.comp.slang" "train_forward_tile_emit.comp")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_forward_tile_gather_high.comp.slang" "train_forward_tile_gather_high.comp")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_forward_tile_gather_items.comp.slang" "train_forward_tile_gather_items.comp")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_forward_tile_sort.comp.slang" "train_forward_tile_sort.comp")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_forward_tile_range_boundaries.comp.slang"
                        "train_forward_tile_range_boundaries.comp")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_range_boundaries_sorted_keys_test.comp.slang"
                        "train_range_boundaries_sorted_keys_test.comp")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_full_sort_prepare_test.comp.slang"
                        "train_full_sort_prepare_test.comp")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_full_sort_emit_test.comp.slang"
                        "train_full_sort_emit_test.comp")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_forward.comp.slang" "train_forward.comp")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_forward_workgroup.comp.slang"
                        "train_forward_workgroup.comp")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_loss.comp.slang" "train_loss.comp")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_ssim_moments_horizontal.comp.slang"
                        "train_ssim_moments_horizontal.comp")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_ssim_coefficients_horizontal.comp.slang"
                        "train_ssim_coefficients_horizontal.comp")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_loss_separable.comp.slang"
                        "train_loss_separable.comp" "${TRAINING_SHADER_PASSES_DIR}/train_loss.comp.slang")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_backward_loss_to_pixel_separable.comp.slang"
                        "train_backward_loss_to_pixel_separable.comp"
                        "${TRAINING_SHADER_PASSES_DIR}/train_backward_loss_to_pixel.comp.slang")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_backward_clear.comp.slang" "train_backward_clear.comp")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_backward_loss_to_pixel.comp.slang" "train_backward_loss_to_pixel.comp")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_backward_pixel_to_2dgs.comp.slang" "train_backward_pixel_to_2dgs.comp")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_backward_pixel_to_2dgs_workgroup.comp.slang"
                        "train_backward_pixel_to_2dgs_workgroup.comp")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_backward_pixel_to_2dgs_subgroup.comp.slang"
                        "train_backward_pixel_to_2dgs_subgroup.comp")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_backward_pixel_to_2dgs_adaptive.comp.slang"
                        "train_backward_pixel_to_2dgs_adaptive.comp")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_backward_tile_gaussian_atomic.comp.slang"
                        "train_backward_tile_gaussian_atomic.comp"
                        "${TRAINING_SHADER_COMMON_DIR}/train_backward_tile_gaussian_impl.slang")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_backward_vksplat_per_splat.comp.slang"
                        "train_backward_vksplat_per_splat.comp")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_backward_vksplat_tensor.comp.slang"
                        "train_backward_vksplat_tensor.comp")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_backward_vksplat_tensor_subgroup.comp.slang"
                        "train_backward_vksplat_tensor_subgroup.comp"
                        "${TRAINING_SHADER_PASSES_DIR}/train_backward_vksplat_tensor.comp.slang")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_backward_vksplat_tensor_subgroup_global_grad.comp.slang"
                        "train_backward_vksplat_tensor_subgroup_global_grad.comp"
                        "${TRAINING_SHADER_PASSES_DIR}/train_backward_vksplat_tensor.comp.slang")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_backward_vksplat_tensor_subgroup_global_grad_latency.comp.slang"
                        "train_backward_vksplat_tensor_subgroup_global_grad_latency.comp"
                        "${TRAINING_SHADER_PASSES_DIR}/train_backward_vksplat_tensor.comp.slang")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_backward_vksplat_tensor_early_cull.comp.slang"
                        "train_backward_vksplat_tensor_early_cull.comp"
                        "${TRAINING_SHADER_PASSES_DIR}/train_backward_vksplat_tensor.comp.slang")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_pixel_culling_probe.comp.slang"
                        "train_pixel_culling_probe.comp")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_backward_pixel_culling_test.comp.slang"
                        "train_backward_pixel_culling_test.comp"
                        "${TRAINING_SHADER_PASSES_DIR}/train_backward_pixel_to_2dgs.comp.slang")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_scanline_census.comp.slang"
                        "train_scanline_census.comp"
                        "${TRAINING_SHADER_COMMON_DIR}/train_scanline_diagnostic.slang")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_scanline_bounds_cost.comp.slang"
                        "train_scanline_bounds_cost.comp"
                        "${TRAINING_SHADER_PASSES_DIR}/train_scanline_census.comp.slang"
                        "${TRAINING_SHADER_COMMON_DIR}/train_scanline_diagnostic.slang")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_row_empty_census.comp.slang"
                        "train_row_empty_census.comp"
                        "${TRAINING_SHADER_PASSES_DIR}/train_scanline_census.comp.slang"
                        "${TRAINING_SHADER_COMMON_DIR}/train_scanline_diagnostic.slang"
                        "${TRAINING_SHADER_COMMON_DIR}/train_row_empty.slang")
foreach(row_empty_pass IN ITEMS train_backward_vksplat_tensor_row_empty
                                train_backward_tensor_batch4_test train_backward_tensor_row_empty_test)
    compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/${row_empty_pass}.comp.slang"
                            "${row_empty_pass}.comp"
                            "${TRAINING_SHADER_PASSES_DIR}/train_backward_vksplat_tensor.comp.slang"
                            "${TRAINING_SHADER_COMMON_DIR}/train_row_empty.slang")
endforeach()
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_backward_2dgs_to_3dgs.comp.slang" "train_backward_2dgs_to_3dgs.comp")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_backward_2dgs_to_3dgs_optimizer.comp.slang"
                        "train_backward_2dgs_to_3dgs_optimizer.comp")
# Private diagnostic helper; no training pipeline selects it.
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_scheduling_sort_test.comp.slang"
                        "train_scheduling_sort_test.comp")
# Offline-only candidates; no training pipeline selects them.
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_backward_2dgs_to_3dgs_optimizer_stream_adam.comp.slang"
                        "train_backward_2dgs_to_3dgs_optimizer_stream_adam.comp")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_backward_2dgs_to_3dgs_optimizer_sh_color_reuse.comp.slang"
                        "train_backward_2dgs_to_3dgs_optimizer_sh_color_reuse.comp"
                        "${TRAINING_SHADER_PASSES_DIR}/train_backward_2dgs_to_3dgs_optimizer.comp.slang")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_optimizer.comp.slang" "train_optimizer.comp")
compile_training_shader("${TRAINING_SHADER_VALIDATION_DIR}/train_gaussian_validation.comp.slang"
                        "train_gaussian_validation.comp")
compile_training_shader("${TRAINING_SHADER_VALIDATION_DIR}/train_validation_finalize.comp.slang"
                        "train_validation_finalize.comp")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_densify_clear.comp.slang" "train_densify_clear.comp")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_densify_prune.comp.slang" "train_densify_prune.comp")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_densify_dispatch.comp.slang"
                        "train_densify_dispatch.comp")
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

compile_training_tile_shader("${TRAINING_SHADER_PASSES_DIR}/train_forward_tile_clear.comp.slang"
                             "train_forward_tile_clear_tile8.comp" 8)
compile_training_tile_shader("${TRAINING_SHADER_PASSES_DIR}/train_forward_tile_count.comp.slang"
                             "train_forward_tile_count_tile8.comp" 8)
compile_training_tile_shader("${TRAINING_SHADER_PASSES_DIR}/train_forward_tile_emit.comp.slang"
                             "train_forward_tile_emit_tile8.comp" 8)
compile_training_tile_shader("${TRAINING_SHADER_PASSES_DIR}/train_forward_tile_range_boundaries.comp.slang"
                             "train_forward_tile_range_boundaries_tile8.comp" 8)
compile_training_tile_shader("${TRAINING_SHADER_PASSES_DIR}/train_forward_tile_sort.comp.slang"
                             "train_forward_tile_sort_tile8.comp" 8)
compile_training_tile_shader("${TRAINING_SHADER_PASSES_DIR}/train_forward.comp.slang"
                             "train_forward_tile8.comp" 8)
compile_training_tile_shader("${TRAINING_SHADER_PASSES_DIR}/train_forward_workgroup.comp.slang"
                             "train_forward_workgroup_tile8.comp" 8)
compile_training_tile_shader("${TRAINING_SHADER_PASSES_DIR}/train_backward_pixel_to_2dgs.comp.slang"
                             "train_backward_pixel_to_2dgs_tile8.comp" 8)
compile_training_tile_shader("${TRAINING_SHADER_PASSES_DIR}/train_backward_pixel_to_2dgs_workgroup.comp.slang"
                             "train_backward_pixel_to_2dgs_workgroup_tile8.comp" 8)
compile_training_tile_shader("${TRAINING_SHADER_PASSES_DIR}/train_backward_pixel_to_2dgs_subgroup.comp.slang"
                             "train_backward_pixel_to_2dgs_subgroup_tile8.comp" 8)
compile_training_tile_shader("${TRAINING_SHADER_PASSES_DIR}/train_backward_pixel_to_2dgs_adaptive.comp.slang"
                             "train_backward_pixel_to_2dgs_adaptive_tile8.comp" 8)
compile_training_tile_shader("${TRAINING_SHADER_PASSES_DIR}/train_backward_tile_gaussian_atomic.comp.slang"
                             "train_backward_tile_gaussian_atomic_tile8.comp" 8
                             "${TRAINING_SHADER_COMMON_DIR}/train_backward_tile_gaussian_impl.slang")
compile_training_tile_shader("${TRAINING_SHADER_PASSES_DIR}/train_backward_vksplat_per_splat.comp.slang"
                             "train_backward_vksplat_per_splat_tile8.comp" 8)
compile_training_tile_shader("${TRAINING_SHADER_PASSES_DIR}/train_backward_vksplat_tensor.comp.slang"
                             "train_backward_vksplat_tensor_tile8.comp" 8)

compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_forward_adaptive_child_count.comp.slang"
                        "train_forward_adaptive_child_count.comp")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_forward_adaptive_child_select.comp.slang"
                        "train_forward_adaptive_child_select.comp")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_forward_adaptive_child_dispatch.comp.slang"
                        "train_forward_adaptive_child_dispatch.comp")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_forward_adaptive_child_prefix.comp.slang"
                        "train_forward_adaptive_child_prefix.comp")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_forward_adaptive_child_block_prefix.comp.slang"
                        "train_forward_adaptive_child_block_prefix.comp")
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_forward_adaptive_child_scatter.comp.slang"
                        "train_forward_adaptive_child_scatter.comp")
compile_training_adaptive_tile_shader(
    "${TRAINING_SHADER_PASSES_DIR}/train_forward_adaptive_parent_range_boundaries.comp.slang"
    "train_forward_adaptive_parent_range_boundaries.comp" 16)
compile_training_adaptive_tile_shader(
    "${TRAINING_SHADER_PASSES_DIR}/train_forward_tile_clear.comp.slang"
    "train_forward_tile_clear_adaptive.comp" 16)
compile_training_adaptive_tile_shader(
    "${TRAINING_SHADER_PASSES_DIR}/train_forward_tile_sort.comp.slang"
    "train_forward_tile_sort_adaptive.comp" 16)
compile_training_shader("${TRAINING_SHADER_PASSES_DIR}/train_forward_adaptive_child_ranges.comp.slang"
                        "train_forward_adaptive_child_ranges.comp")
compile_training_adaptive_tile_shader(
    "${TRAINING_SHADER_PASSES_DIR}/train_forward.comp.slang"
    "train_forward_adaptive.comp" 16)
foreach(adaptive_tile_size IN ITEMS 8 16)
    compile_training_adaptive_tile_shader(
        "${TRAINING_SHADER_PASSES_DIR}/train_forward_workgroup.comp.slang"
        "train_forward_workgroup_adaptive_tile${adaptive_tile_size}.comp"
        ${adaptive_tile_size})
    compile_training_adaptive_tile_shader(
        "${TRAINING_SHADER_PASSES_DIR}/train_backward_vksplat_per_splat.comp.slang"
        "train_backward_vksplat_per_splat_adaptive_tile${adaptive_tile_size}.comp"
        ${adaptive_tile_size})
    compile_training_adaptive_tile_shader(
        "${TRAINING_SHADER_PASSES_DIR}/train_backward_vksplat_tensor.comp.slang"
        "train_backward_vksplat_tensor_adaptive_tile${adaptive_tile_size}.comp"
        ${adaptive_tile_size})
endforeach()

foreach(child_tensor_batch IN ITEMS 16 32)
    compile_training_adaptive_tile_shader(
        "${TRAINING_SHADER_PASSES_DIR}/train_backward_vksplat_tensor.comp.slang"
        "train_backward_vksplat_tensor_adaptive_tile8_batch${child_tensor_batch}.comp"
        8 CHILD_TENSOR_BATCH ${child_tensor_batch})
endforeach()

add_custom_target(compile_shaders DEPENDS ${ALL_SHADER_FILES})

# 添加子目录
add_subdirectory(src)
add_subdirectory(apps)
if(BUILD_TESTING)
    add_subdirectory(tests test)
endif()

# 确保在构建库之前编译着色器
# 注意：这里假设 src/CMakeLists.txt 中定义的目标名为 vulkan_3dgs_core
# 如果目标名不同，请相应调整
add_dependencies(vulkan_3dgs_core compile_shaders)

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
