foreach(required_variable
        CLANG_TIDY_EXECUTABLE
        SOURCE_FILE
        SOURCE_ROOT
        SOURCE_INCLUDE_DIR
        RADIX_INCLUDE_DIR
        IMGUI_INCLUDE_DIR
        DEPENDENCY_INCLUDE_DIR
        MODE)
    if(NOT DEFINED ${required_variable})
        message(FATAL_ERROR "RunClangTidy.cmake requires ${required_variable}")
    endif()
endforeach()

file(RELATIVE_PATH source_label "${SOURCE_ROOT}" "${SOURCE_FILE}")
message(STATUS "clang-tidy ${MODE}: ${source_label}")

set(command
    "${CLANG_TIDY_EXECUTABLE}"
    "${SOURCE_FILE}"
    "--config-file=${SOURCE_ROOT}/.clang-tidy"
)
if(MODE STREQUAL "required")
    list(APPEND command "--checks=-*,bugprone-*" --quiet)
elseif(MODE STREQUAL "advisory")
    list(APPEND command "--warnings-as-errors=-*")
else()
    message(FATAL_ERROR "Unknown clang-tidy mode: ${MODE}")
endif()

list(APPEND command
    "--extra-arg=-std=c++20"
    "--extra-arg=-DGLFW_INCLUDE_VULKAN"
    "--extra-arg=-DGLM_FORCE_DEPTH_ZERO_TO_ONE"
    "--extra-arg=-DNOMINMAX"
    "--extra-arg=-DVULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1"
    "--extra-arg=-I${SOURCE_INCLUDE_DIR}"
    "--extra-arg=-I${RADIX_INCLUDE_DIR}"
    "--extra-arg=-I${IMGUI_INCLUDE_DIR}"
    "--extra-arg=-I${DEPENDENCY_INCLUDE_DIR}"
    --
)

execute_process(
    COMMAND ${command}
    WORKING_DIRECTORY "${SOURCE_ROOT}"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE standard_output
    ERROR_VARIABLE standard_error
    ENCODING UTF-8
)

set(output "${standard_output}${standard_error}")
string(REPLACE "\r\n" "\n" output "${output}")
string(REGEX REPLACE "(^|\n)[ \t]*[0-9]+ warnings generated\\.[ \t]*(\n|$)" "\\1" output "${output}")
string(REGEX REPLACE "(^|\n)[ \t]*Suppressed [^\n]*(\n|$)" "\\1" output "${output}")
string(REGEX REPLACE "(^|\n)[ \t]*Use -header-filter=[^\n]*(\n|$)" "\\1" output "${output}")
string(STRIP "${output}" output)
if(output)
    message("${output}")
endif()

if(NOT result EQUAL 0)
    message(FATAL_ERROR "clang-tidy ${MODE} failed for ${source_label} (exit ${result})")
endif()
