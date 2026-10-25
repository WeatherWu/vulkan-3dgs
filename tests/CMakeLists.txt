function(configure_vulkan_3dgs_test target)
    set(test_output_directory "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>")
    set_target_properties(${target} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${test_output_directory}"
        ARCHIVE_OUTPUT_DIRECTORY "${test_output_directory}"
        LIBRARY_OUTPUT_DIRECTORY "${test_output_directory}"
        PDB_OUTPUT_DIRECTORY "${test_output_directory}"
    )

    set(test_configurations ${CMAKE_CONFIGURATION_TYPES})
    if(CMAKE_BUILD_TYPE)
        list(APPEND test_configurations ${CMAKE_BUILD_TYPE})
    endif()
    list(REMOVE_DUPLICATES test_configurations)
    foreach(config IN LISTS test_configurations)
        string(TOUPPER "${config}" config_upper)
        set(config_output_directory "${CMAKE_CURRENT_BINARY_DIR}/${config}")
        set_target_properties(${target} PROPERTIES
            "RUNTIME_OUTPUT_DIRECTORY_${config_upper}" "${config_output_directory}"
            "ARCHIVE_OUTPUT_DIRECTORY_${config_upper}" "${config_output_directory}"
            "LIBRARY_OUTPUT_DIRECTORY_${config_upper}" "${config_output_directory}"
            "PDB_OUTPUT_DIRECTORY_${config_upper}" "${config_output_directory}"
        )
    endforeach()

    add_test(NAME ${target} COMMAND $<TARGET_FILE:${target}>)
endfunction()

add_executable(application_command_line_tests application_command_line_tests.cpp)
target_include_directories(application_command_line_tests PRIVATE ../src)
configure_vulkan_3dgs_test(application_command_line_tests)
add_test(NAME application_validation_help
    COMMAND $<TARGET_FILE:vulkan_3dgs_app> --validation off --help)
set_tests_properties(application_validation_help PROPERTIES
    PASS_REGULAR_EXPRESSION "--validation on[|]off")

add_executable(image_cache_tests image_cache_tests.cpp)
target_link_libraries(image_cache_tests PRIVATE vulkan_3dgs_core)
configure_vulkan_3dgs_test(image_cache_tests)

add_executable(training_ssim_tests training_ssim_tests.cpp)
target_include_directories(training_ssim_tests PRIVATE ../src)
configure_vulkan_3dgs_test(training_ssim_tests)

add_executable(training_pixel_culling_tests training_pixel_culling_tests.cpp)
configure_vulkan_3dgs_test(training_pixel_culling_tests)
add_executable(training_scanline_tests training_scanline_tests.cpp)
target_link_libraries(training_scanline_tests PRIVATE vulkan_3dgs_core)
configure_vulkan_3dgs_test(training_scanline_tests)
add_executable(training_row_empty_tests training_row_empty_tests.cpp)
target_link_libraries(training_row_empty_tests PRIVATE vulkan_3dgs_core)
configure_vulkan_3dgs_test(training_row_empty_tests)

add_executable(training_optimizer_tests training_optimizer_tests.cpp)
target_link_libraries(training_optimizer_tests PRIVATE vulkan_3dgs_core)
configure_vulkan_3dgs_test(training_optimizer_tests)

add_executable(training_scheduling_tests training_scheduling_tests.cpp)
target_link_libraries(training_scheduling_tests PRIVATE vulkan_3dgs_core)
configure_vulkan_3dgs_test(training_scheduling_tests)

add_executable(training_sort_replay_tests training_sort_replay_tests.cpp)
target_link_libraries(training_sort_replay_tests PRIVATE vulkan_3dgs_core)
configure_vulkan_3dgs_test(training_sort_replay_tests)

# Private baseline replay; hardware execution is explicit, not a CTest prerequisite.
file(SHA256 "${CMAKE_SOURCE_DIR}/third_party/vulkan_radix_sort/include/vk_radix_sort.h"
    SORT_REPLAY_SORTER_HASH)
add_executable(training_sort_baseline_gpu_tests training_sort_baseline_gpu_tests.cpp)
target_link_libraries(training_sort_baseline_gpu_tests PRIVATE vulkan_3dgs_core)
target_compile_definitions(training_sort_baseline_gpu_tests PRIVATE
    VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
    SORT_REPLAY_RUNTIME_DIR="${CMAKE_BINARY_DIR}/bin/$<CONFIG>"
    SORT_REPLAY_BUILD_CONFIG="$<CONFIG>"
    SORT_REPLAY_COMPILER="${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
    SORT_REPLAY_SHADER_COMPILER="Slang ${OPTIMIZER_SLANG_VERSION}, debug=${VULKAN_3DGS_TRAINING_SHADER_DEBUG_INFO}"
    SORT_REPLAY_SORTER_HASH="${SORT_REPLAY_SORTER_HASH}")
set_target_properties(training_sort_baseline_gpu_tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>")
foreach(sort_config IN LISTS CMAKE_CONFIGURATION_TYPES)
    string(TOUPPER "${sort_config}" sort_config_upper)
    set_target_properties(training_sort_baseline_gpu_tests PROPERTIES
        "RUNTIME_OUTPUT_DIRECTORY_${sort_config_upper}" "${CMAKE_CURRENT_BINARY_DIR}/${sort_config}")
endforeach()
add_test(NAME training_sort_baseline_help COMMAND $<TARGET_FILE:training_sort_baseline_gpu_tests> --help)

# Explicit S4-3 private diagnostics, not a hardware-dependent CTest.
file(SHA256 "${CMAKE_SOURCE_DIR}/third_party/vulkan_radix_sort/include/vk_radix_sort.h"
    SCHEDULING_SORTER_HASH)
add_executable(training_scheduling_gpu_tests training_scheduling_gpu_tests.cpp)
target_link_libraries(training_scheduling_gpu_tests PRIVATE vulkan_3dgs_core)
target_compile_definitions(training_scheduling_gpu_tests PRIVATE
    VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
    SCHEDULING_RUNTIME_DIR="${CMAKE_BINARY_DIR}/bin/$<CONFIG>"
    SCHEDULING_SORTER_HASH="${SCHEDULING_SORTER_HASH}")
set_target_properties(training_scheduling_gpu_tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>")
foreach(scheduling_config IN LISTS CMAKE_CONFIGURATION_TYPES)
    string(TOUPPER "${scheduling_config}" scheduling_config_upper)
    set_target_properties(training_scheduling_gpu_tests PROPERTIES
        "RUNTIME_OUTPUT_DIRECTORY_${scheduling_config_upper}"
        "${CMAKE_CURRENT_BINARY_DIR}/${scheduling_config}")
endforeach()

# Explicit independent S4 runner: no checkpoint/scanline suite or Tensor capability gate.
add_executable(training_optimizer_gpu_tests training_optimizer_gpu_tests.cpp)
target_link_libraries(training_optimizer_gpu_tests PRIVATE vulkan_3dgs_core)
target_compile_definitions(training_optimizer_gpu_tests PRIVATE
    VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
    OPTIMIZER_RUNTIME_DIR="${CMAKE_BINARY_DIR}/bin/$<CONFIG>")
set_target_properties(training_optimizer_gpu_tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>")
foreach(optimizer_config IN LISTS CMAKE_CONFIGURATION_TYPES)
    string(TOUPPER "${optimizer_config}" optimizer_config_upper)
    set_target_properties(training_optimizer_gpu_tests PROPERTIES
        "RUNTIME_OUTPUT_DIRECTORY_${optimizer_config_upper}" "${CMAKE_CURRENT_BINARY_DIR}/${optimizer_config}")
endforeach()

add_executable(training_checkpoint_tests training_checkpoint_tests.cpp)
target_include_directories(training_checkpoint_tests PRIVATE ../src)
# This suite checks reference/layout contracts and the rejection regression.
# The new reverse-boundary CPU admission gate is registered independently.
# The old --candidate-gate remains an expected failure and is run manually.
configure_vulkan_3dgs_test(training_checkpoint_tests)
add_test(NAME training_checkpoint_reverse_boundary_gate
    COMMAND $<TARGET_FILE:training_checkpoint_tests> --reverse-boundary-gate)

# Explicit GPU diagnostic target: hardware/runtime availability is not a CPU CTest prerequisite.
add_executable(training_checkpoint_gpu_tests training_checkpoint_gpu_tests.cpp)
target_link_libraries(training_checkpoint_gpu_tests PRIVATE vulkan_3dgs_core)
target_compile_definitions(training_checkpoint_gpu_tests PRIVATE
    VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
    CHECKPOINT_RUNTIME_DIR="${CMAKE_BINARY_DIR}/bin/$<CONFIG>")
set_target_properties(training_checkpoint_gpu_tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>")
foreach(checkpoint_config IN LISTS CMAKE_CONFIGURATION_TYPES)
    string(TOUPPER "${checkpoint_config}" checkpoint_config_upper)
    set_target_properties(training_checkpoint_gpu_tests PROPERTIES
        "RUNTIME_OUTPUT_DIRECTORY_${checkpoint_config_upper}"
        "${CMAKE_CURRENT_BINARY_DIR}/${checkpoint_config}")
endforeach()

add_executable(training_validation_tests training_validation_tests.cpp)
target_link_libraries(training_validation_tests PRIVATE vulkan_3dgs_core)
configure_vulkan_3dgs_test(training_validation_tests)

add_executable(training_services_tests training_services_tests.cpp)
target_link_libraries(training_services_tests PRIVATE vulkan_3dgs_core)
configure_vulkan_3dgs_test(training_services_tests)

add_executable(training_frame_scheduler_tests training_frame_scheduler_tests.cpp)
target_link_libraries(training_frame_scheduler_tests PRIVATE vulkan_3dgs_core)
configure_vulkan_3dgs_test(training_frame_scheduler_tests)

add_executable(training_model_io_tests training_model_io_tests.cpp)
target_link_libraries(training_model_io_tests PRIVATE vulkan_3dgs_core)
configure_vulkan_3dgs_test(training_model_io_tests)

add_executable(training_image_runtime_tests training_image_runtime_tests.cpp)
target_link_libraries(training_image_runtime_tests PRIVATE vulkan_3dgs_core)
configure_vulkan_3dgs_test(training_image_runtime_tests)

add_executable(training_math_tests training_math_tests.cpp)
target_include_directories(training_math_tests PRIVATE ../src)
find_package(glm CONFIG REQUIRED)
target_link_libraries(training_math_tests PRIVATE glm::glm)
configure_vulkan_3dgs_test(training_math_tests)

add_executable(backward_dispatch_policy_tests backward_dispatch_policy_tests.cpp)
target_link_libraries(backward_dispatch_policy_tests PRIVATE vulkan_3dgs_core)
configure_vulkan_3dgs_test(backward_dispatch_policy_tests)

add_executable(training_settings_tests training_settings_tests.cpp)
target_link_libraries(training_settings_tests PRIVATE vulkan_3dgs_core)
configure_vulkan_3dgs_test(training_settings_tests)

add_executable(camera_controller_tests camera_controller_tests.cpp)
target_link_libraries(camera_controller_tests PRIVATE vulkan_3dgs_core)
configure_vulkan_3dgs_test(camera_controller_tests)

add_executable(viewer_controller_tests viewer_controller_tests.cpp)
target_link_libraries(viewer_controller_tests PRIVATE vulkan_3dgs_core)
configure_vulkan_3dgs_test(viewer_controller_tests)

add_executable(gaussian_render_profile_tests gaussian_render_profile_tests.cpp)
configure_vulkan_3dgs_test(gaussian_render_profile_tests)

add_executable(gpu_selection_tests
    gpu_selection_tests.cpp
    ../src/context/device_selection.cpp
)
target_include_directories(gpu_selection_tests PRIVATE ../src)
configure_vulkan_3dgs_test(gpu_selection_tests)
