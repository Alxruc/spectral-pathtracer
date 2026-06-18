# Spectral path tracer — builds on AMD (HIP/ROCm) by default. Switch hardware
# entirely from the command line, no edits needed:
#   AMD:    -DGPU_RUNTIME=HIP   (GPU_TARGETS auto-detected, or pass -DGPU_TARGETS=gfx{number})
#   NVIDIA: -DGPU_RUNTIME=CUDA -DGPU_TARGETS=120  (compute capability, e.g. 120=Consumer Blackwell)
cmake_minimum_required(VERSION 3.21)

set(GPU_RUNTIME "HIP" CACHE STRING "GPU backend: HIP (AMD) or CUDA (NVIDIA)")
set_property(CACHE GPU_RUNTIME PROPERTY STRINGS HIP CUDA)

if(GPU_RUNTIME STREQUAL "CUDA")
    set(CMAKE_HIP_PLATFORM "nvidia")
else()
    set(CMAKE_HIP_PLATFORM "amd")
endif()

project(spectral_pathtracer
    VERSION 0.1.0
    DESCRIPTION "Spectral path tracer on HIP (AMD by default, CUDA-capable)"
    LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 20)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# The HIP toolchain links executables as position-independent (PIE) by default,
# so every object linked in — including the plain C++ ones — must be built with
# -fPIC, otherwise ld.lld rejects R_X86_64_32 relocations against local symbols.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(GPU_TARGETS "" CACHE STRING "GPU architecture(s) to target; auto-detected for AMD if empty")
if(NOT GPU_TARGETS)
    if(GPU_RUNTIME STREQUAL "CUDA")
        message(FATAL_ERROR "Set the NVIDIA arch explicitly, e.g. -DGPU_TARGETS=86")
    endif()
    get_filename_component(_hip_bin "${CMAKE_HIP_COMPILER}" DIRECTORY)
    find_program(AMDGPU_ARCH_EXE amdgpu-arch HINTS "${_hip_bin}")
    if(NOT AMDGPU_ARCH_EXE)
        message(FATAL_ERROR "Could not auto-detect a GPU. Pass -DGPU_TARGETS=gfx... (see: rocminfo | grep gfx)")
    endif()
    execute_process(COMMAND "${AMDGPU_ARCH_EXE}"
        OUTPUT_VARIABLE _detected_archs OUTPUT_STRIP_TRAILING_WHITESPACE)
    string(REPLACE "\n" ";" _detected_archs "${_detected_archs}")
    list(GET _detected_archs 0 GPU_TARGETS)
    message(STATUS "Auto-detected GPU_TARGETS=${GPU_TARGETS} (override with -DGPU_TARGETS=)")
endif()
set(CMAKE_HIP_ARCHITECTURES "${GPU_TARGETS}")

option(SPT_WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)

add_library(spectral_core
    src/renderer.hip
)
target_include_directories(spectral_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_source_files_properties(src/renderer.hip PROPERTIES LANGUAGE HIP)
target_compile_options(spectral_core PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-fms-extensions>)

if(SPT_WARNINGS_AS_ERRORS)
    target_compile_options(spectral_core PRIVATE -Wall -Wextra -Werror)
else()
    target_compile_options(spectral_core PRIVATE -Wall -Wextra)
endif()

add_library(image STATIC src/image/image_io.cpp)
target_include_directories(image PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/stb
    ${CMAKE_CURRENT_SOURCE_DIR}/src/image)

add_executable(spectral_pathtracer src/main.cpp)
target_link_libraries(spectral_pathtracer PRIVATE spectral_core image)
