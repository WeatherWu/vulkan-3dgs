set(VULKAN_3DGS_QUALITY_RELATIVE_FILES
    app/application.cpp
    app/application.hpp

    viewer/viewer_controller.cpp
    viewer/viewer_controller.hpp
    viewer/viewer_panel.cpp
    viewer/viewer_panel.hpp
    viewer/viewer_settings.cpp
    viewer/viewer_settings.hpp

    training/app/training_controller.cpp
    training/app/training_controller.hpp
    training/app/training_settings.cpp
    training/app/training_settings.hpp

    training/ui/training_command_panel.cpp
    training/ui/training_command_panel.hpp
    training/ui/training_diagnostics_panel.cpp
    training/ui/training_diagnostics_panel.hpp
    training/ui/training_file_dialogs.cpp
    training/ui/training_file_dialogs.hpp
    training/ui/training_panel.cpp
    training/ui/training_panel.hpp
    training/ui/training_settings_panel.cpp
    training/ui/training_settings_panel.hpp

    training/core/training_frame_scheduler.cpp
    training/core/training_frame_scheduler.hpp
    training/core/training_image_runtime.cpp
    training/core/training_image_runtime.hpp
    training/core/training_model_io.cpp
    training/core/training_model_io.hpp
    training/core/training_profiling_service.cpp
    training/core/training_profiling_service.hpp
    training/core/training_step_executor.cpp
    training/core/training_step_executor.hpp
    training/core/training_validation_service.cpp
    training/core/training_validation_service.hpp
    training/core/backward/backward_pass_context.hpp
    training/core/backward/backward_command_utils.cpp
    training/core/backward/backward_command_utils.hpp
    training/core/backward/backward_loss_pass.cpp
    training/core/backward/backward_loss_pass.hpp
    training/core/backward/pixel_to_2dgs_dispatcher.cpp
    training/core/backward/pixel_to_2dgs_dispatcher.hpp
    training/core/backward/projection_optimizer_pass.cpp
    training/core/backward/projection_optimizer_pass.hpp
    training/core/backward/backward_validation_pass.cpp
    training/core/backward/backward_validation_pass.hpp
    training/core/gaussian_backward_renderer.cpp
    training/core/gaussian_backward_renderer.hpp

    graphics/gaussian_renderer.cpp
    graphics/gaussian_renderer.hpp
    graphics/graphics_command_recorder.cpp
    graphics/graphics_command_recorder.hpp
    graphics/graphics_frame_runtime.cpp
    graphics/graphics_frame_runtime.hpp
    graphics/graphics_pipeline_set.cpp
    graphics/graphics_pipeline_set.hpp
    graphics/graphics_splat_resources.cpp
    graphics/graphics_splat_resources.hpp
    graphics/graphics_splat_sorter.cpp
    graphics/graphics_splat_sorter.hpp

    ../tests/backward_dispatch_policy_tests.cpp
)

set(VULKAN_3DGS_QUALITY_FILES)
set(VULKAN_3DGS_QUALITY_SOURCES)
foreach(relative_file IN LISTS VULKAN_3DGS_QUALITY_RELATIVE_FILES)
    set(absolute_file "${CMAKE_CURRENT_SOURCE_DIR}/${relative_file}")
    list(APPEND VULKAN_3DGS_QUALITY_FILES "${absolute_file}")
    if(relative_file MATCHES "\\.cpp$")
        list(APPEND VULKAN_3DGS_QUALITY_SOURCES "${absolute_file}")
    endif()
endforeach()

find_program(VULKAN_3DGS_CLANG_FORMAT_EXECUTABLE NAMES clang-format)
if(VULKAN_3DGS_CLANG_FORMAT_EXECUTABLE)
    add_custom_target(format-new
        COMMAND "${VULKAN_3DGS_CLANG_FORMAT_EXECUTABLE}"
                -i --style=file ${VULKAN_3DGS_QUALITY_FILES}
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        COMMENT "Formatting refactored C++ modules"
        VERBATIM
    )
    add_custom_target(check-format-new
        COMMAND "${VULKAN_3DGS_CLANG_FORMAT_EXECUTABLE}"
                --dry-run --Werror --style=file ${VULKAN_3DGS_QUALITY_FILES}
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        COMMENT "Checking formatting of refactored C++ modules"
        VERBATIM
    )
else()
    message(STATUS "clang-format not found; format-new targets are unavailable")
endif()

find_program(VULKAN_3DGS_CLANG_TIDY_EXECUTABLE NAMES clang-tidy)
if(VULKAN_3DGS_CLANG_TIDY_EXECUTABLE)
    set(VULKAN_3DGS_REQUIRED_TIDY_COMMANDS)
    set(VULKAN_3DGS_ADVISORY_TIDY_COMMANDS)
    foreach(source_file IN LISTS VULKAN_3DGS_QUALITY_SOURCES)
        set(common_arguments
            "-DCLANG_TIDY_EXECUTABLE=${VULKAN_3DGS_CLANG_TIDY_EXECUTABLE}"
            "-DSOURCE_FILE=${source_file}"
            "-DSOURCE_ROOT=${CMAKE_SOURCE_DIR}"
            "-DSOURCE_INCLUDE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
            "-DRADIX_INCLUDE_DIR=${CMAKE_SOURCE_DIR}/third_party/vulkan_radix_sort/include"
            "-DIMGUI_INCLUDE_DIR=${IMGUI_FILE_DIALOG_DIR}"
            "-DDEPENDENCY_INCLUDE_DIR=${Stb_INCLUDE_DIR}"
        )
        list(APPEND VULKAN_3DGS_REQUIRED_TIDY_COMMANDS
            COMMAND "${CMAKE_COMMAND}" ${common_arguments}
                    -DMODE=required
                    -P "${CMAKE_SOURCE_DIR}/cmake/RunClangTidy.cmake"
        )
        list(APPEND VULKAN_3DGS_ADVISORY_TIDY_COMMANDS
            COMMAND "${CMAKE_COMMAND}" ${common_arguments}
                    -DMODE=advisory
                    -P "${CMAKE_SOURCE_DIR}/cmake/RunClangTidy.cmake"
        )
    endforeach()
    add_custom_target(clang-tidy-new
        ${VULKAN_3DGS_REQUIRED_TIDY_COMMANDS}
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        COMMENT "Running required bugprone checks on refactored C++ modules"
        VERBATIM
    )
    add_custom_target(clang-tidy-advisory-new
        ${VULKAN_3DGS_ADVISORY_TIDY_COMMANDS}
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        COMMENT "Reporting Core Guidelines and modernization advice for refactored modules"
        VERBATIM
    )
else()
    message(STATUS "clang-tidy not found; clang-tidy-new target is unavailable")
endif()

if(TARGET check-format-new AND TARGET clang-tidy-new)
    add_custom_target(quality-new)
    add_dependencies(quality-new check-format-new clang-tidy-new)
endif()
