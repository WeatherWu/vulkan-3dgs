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
# 查找Vulkan GLSL编译器 (glslangValidator)
find_program(GLSLANG_VALIDATOR glslangValidator REQUIRED)

# 定义着色器编译函数（无论是否找到glslangValidator都定义）
function(compile_shader SHADER_SRC SHADER_DST)
    get_filename_component(SHADER_NAME ${SHADER_SRC} NAME_WE)
    get_filename_component(SHADER_EXT ${SHADER_SRC} EXT)
    
    # 根据文件扩展名确定着色器阶段
    if(${SHADER_EXT} STREQUAL ".vert")
        set(SHADER_STAGE "vert")
    elseif(${SHADER_EXT} STREQUAL ".frag")
        set(SHADER_STAGE "frag")
    elseif(${SHADER_EXT} STREQUAL ".comp")
        set(SHADER_STAGE "comp")
    elseif(${SHADER_EXT} STREQUAL ".geom")
        set(SHADER_STAGE "geom")
    else()
        message(WARNING "Unknown shader type: ${SHADER_EXT}")
        return()
    endif()
    
    # 创建输出目录
    get_filename_component(OUTPUT_DIR ${SHADER_DST} DIRECTORY)
    file(MAKE_DIRECTORY ${OUTPUT_DIR})
    
    # 添加自定义命令编译着色器
    add_custom_command(
        OUTPUT ${SHADER_DST}
        COMMAND ${GLSLANG_VALIDATOR} -V ${SHADER_SRC} -o ${SHADER_DST} --target-env vulkan1.2
        DEPENDS ${SHADER_SRC}
        COMMENT "Compiling shader ${SHADER_NAME}${SHADER_EXT} -> ${SHADER_NAME}.${SHADER_STAGE}.spv"
        VERBATIM
    )
    
    # 将生成的SPV文件标记为 GENERATED
    set_source_files_properties(${SHADER_DST} PROPERTIES GENERATED TRUE)
endfunction()

if(NOT GLSLANG_VALIDATOR)
    message(WARNING "glslangValidator not found! Shaders will not be compiled automatically.")
else()
    message(STATUS "Found glslangValidator: ${GLSLANG_VALIDATOR}")
    
    # 编译图形着色器
    set(GRAPHICS_SHADER_SRC_DIR "${CMAKE_SOURCE_DIR}/shaders/gaussian_compute_shader/src")
    set(GRAPHICS_SHADER_SPV_DIR "${CMAKE_SOURCE_DIR}/shaders/gaussian_compute_shader/spv")
    
    # 定义着色器文件列表（包含 Compute Shader）
    set(GRAPHICS_SHADERS
        "${GRAPHICS_SHADER_SRC_DIR}/gaussian_compute_shader.vert"
        "${GRAPHICS_SHADER_SRC_DIR}/gaussian_compute_shader.frag"
        "${GRAPHICS_SHADER_SRC_DIR}/gaussian_compute_shader.comp"
    )
    
    # 编译每个着色器并复制到构建目录
    foreach(SHADER_SRC ${GRAPHICS_SHADERS})
        get_filename_component(SHADER_NAME ${SHADER_SRC} NAME_WE)
        get_filename_component(SHADER_EXT ${SHADER_SRC} EXT)
        
        # 确定SPIR-V输出路径（源码目录）
        if(${SHADER_EXT} STREQUAL ".vert")
            set(SHADER_SPV "${GRAPHICS_SHADER_SPV_DIR}/${SHADER_NAME}.vert.spv")
        elseif(${SHADER_EXT} STREQUAL ".frag")
            set(SHADER_SPV "${GRAPHICS_SHADER_SPV_DIR}/${SHADER_NAME}.frag.spv")
        elseif(${SHADER_EXT} STREQUAL ".comp")
            set(SHADER_SPV "${GRAPHICS_SHADER_SPV_DIR}/${SHADER_NAME}.comp.spv")
        else()
            continue()
        endif()

        # 编译着色器到源码目录的spv文件夹
        compile_shader(${SHADER_SRC} ${SHADER_SPV})
        
        # 使用简化的生成器表达式动态确定目标目录（支持多配置生成器）
        set(BUILD_SPV "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders/${SHADER_NAME}${SHADER_EXT}.spv")
        
        # 添加自定义命令，在构建后强制复制SPV文件到构建目录（确保使用最新版本）
        add_custom_command(
            OUTPUT ${BUILD_SPV}
            COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SHADER_SPV} ${BUILD_SPV}
            DEPENDS ${SHADER_SPV}
            VERBATIM
        )
        
        set_source_files_properties(${BUILD_SPV} PROPERTIES GENERATED TRUE)
        
        # 添加到全局着色器文件列表
        list(APPEND ALL_SHADER_FILES ${BUILD_SPV})
    endforeach()
    
    # 创建自定义目标用于着色器编译
    add_custom_target(compile_shaders DEPENDS ${ALL_SHADER_FILES})
endif()

# 添加子目录
add_subdirectory(src)
add_subdirectory(apps)
if(CMAKE_CONFIGURATION_TYPES OR CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_subdirectory(sandbox)
endif()

# 确保在构建库之前编译着色器
if(GLSLANG_VALIDATOR)
    # 注意：这里假设 src/CMakeLists.txt 中定义的目标名为 vk_gs_core
    # 如果目标名不同，请相应调整
    add_dependencies(vk_gs_core compile_shaders)
endif()

# 安装配置
install(DIRECTORY src/ DESTINATION include)

# 安装着色器文件（使用生成器表达式支持多配置）
install(DIRECTORY ${GRAPHICS_SHADER_SPV_DIR}/ 
    DESTINATION bin/$<CONFIG>/shaders
    FILES_MATCHING PATTERN "*.spv"
)

install(TARGETS vk_gs_core # vk_gs_demo
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
