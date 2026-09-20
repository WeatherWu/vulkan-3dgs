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

add_executable(image_cache_tests image_cache_tests.cpp)
target_link_libraries(image_cache_tests PRIVATE vulkan_3dgs_core)
configure_vulkan_3dgs_test(image_cache_tests)

add_executable(training_ssim_tests training_ssim_tests.cpp)
configure_vulkan_3dgs_test(training_ssim_tests)

add_executable(training_validation_tests training_validation_tests.cpp)
target_link_libraries(training_validation_tests PRIVATE vulkan_3dgs_core)
configure_vulkan_3dgs_test(training_validation_tests)

add_executable(training_math_tests training_math_tests.cpp)
configure_vulkan_3dgs_test(training_math_tests)

add_executable(camera_controller_tests camera_controller_tests.cpp)
target_link_libraries(camera_controller_tests PRIVATE vulkan_3dgs_core)
configure_vulkan_3dgs_test(camera_controller_tests)

add_executable(gaussian_render_profile_tests gaussian_render_profile_tests.cpp)
configure_vulkan_3dgs_test(gaussian_render_profile_tests)

add_executable(gpu_selection_tests
    gpu_selection_tests.cpp
    ../src/context/device_selection.cpp
)
target_include_directories(gpu_selection_tests PRIVATE ../src)
configure_vulkan_3dgs_test(gpu_selection_tests)
