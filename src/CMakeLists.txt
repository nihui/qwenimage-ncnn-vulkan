cmake_policy(SET CMP0091 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0091 NEW)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

cmake_minimum_required(VERSION 3.18)
project(qwenimage-ncnn-vulkan LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE release CACHE STRING "Choose the type of build" FORCE)
endif()

add_executable(qwenimage-ncnn-vulkan
    main.cpp
    image_io.cpp
    runtime.cpp
    dupup3d.cpp
    avgdown3d.cpp
    tokenizer.cpp
    scheduler.cpp
    models.cpp
    lora.cpp
    vision.cpp
    text_encoder.cpp
    vae.cpp
    transformer.cpp
    pipeline.cpp
    edit_pipeline.cpp
    controlnet.cpp
    edit_input.cpp
)

include(${CMAKE_CURRENT_SOURCE_DIR}/deps_ncnn.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/deps_codec.cmake)

set(QWENIMAGE_LINK_LIBRARIES ncnn ${DEPS_CODEC_LIBRARIES})

target_include_directories(qwenimage-ncnn-vulkan PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/ncnn/src
)

target_link_libraries(qwenimage-ncnn-vulkan ${QWENIMAGE_LINK_LIBRARIES})

include(GNUInstallDirs)
install(TARGETS qwenimage-ncnn-vulkan RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
