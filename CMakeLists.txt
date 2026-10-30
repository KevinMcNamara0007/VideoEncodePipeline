# ------------------------------------------------------------------------
# Copyright (c) 2026 Mjolnir Photonics Labs. All rights reserved.
# Architect: Kevin McNamara, CTO and Chief Science Officer.
# PROPRIETARY AND CONFIDENTIAL. Protected by patents and pending patent
# applications. No permission to copy, modify or reuse; use is strictly
# governed by the EULA (EULA.md at the repository root).
# ------------------------------------------------------------------------
# EWORLD aerial encode, native. Links EHC (libehc_core.a) for the streaming
# video encoder and the LSH index, OpenCV for YOLO (DNN) and drawing,
# libtesseract for stencil reads. The HaloCore Florence door is opened at
# run time (dlopen); only its C ABI header is read at build time.
#
#   cmake -S . -B build -DEHC_ROOT=/path/to/EHC -DHALOCORE_ROOT=/path/to/HaloCore
#   cmake --build build -j
cmake_minimum_required(VERSION 3.20)
project(eworld LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)   # EHC headers use std::span
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(EHC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../EHC"
    CACHE PATH "EHC repository root (include/ and a built build/libehc_core.a)")
set(HALOCORE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../HaloCore"
    CACHE PATH "HaloCore repository root (cpp/src/doc/door_abi.h)")
if(NOT EXISTS "${EHC_ROOT}/include/ehc/media/streaming_video_encoder.hpp")
    message(FATAL_ERROR "EHC headers not found under ${EHC_ROOT}. Pass -DEHC_ROOT=/path/to/EHC.")
endif()
# build/ only: a stale build-*/ archive links fine and fails at run time.
set(EHC_CORE_LIB "${EHC_ROOT}/build/libehc_core.a")
if(NOT EXISTS "${EHC_CORE_LIB}")
    message(FATAL_ERROR "${EHC_CORE_LIB} missing. Build EHC first: cmake --build ${EHC_ROOT}/build --target ehc_core")
endif()
if(NOT EXISTS "${HALOCORE_ROOT}/cpp/src/doc/door_abi.h")
    message(FATAL_ERROR "HaloCore door ABI not found under ${HALOCORE_ROOT}. Pass -DHALOCORE_ROOT=/path/to/HaloCore.")
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs dnn)
find_package(nlohmann_json 3 REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(TESSERACT REQUIRED IMPORTED_TARGET tesseract lept)

add_library(eworld_core STATIC
    src/rules.cpp
    src/yolo.cpp
    src/ocr.cpp
    src/florence.cpp
    src/aerial_detector.cpp
)
target_include_directories(eworld_core
    PUBLIC  src "${EHC_ROOT}/include"
    PRIVATE "${HALOCORE_ROOT}/cpp/src")
target_link_libraries(eworld_core
    PUBLIC  "${EHC_CORE_LIB}" ${OpenCV_LIBS} nlohmann_json::nlohmann_json PkgConfig::TESSERACT ${CMAKE_DL_LIBS})
target_compile_options(eworld_core PRIVATE -Wall -Wextra)

add_executable(aerial_encode src/aerial_encode.cpp)
target_link_libraries(aerial_encode PRIVATE eworld_core)
target_compile_definitions(aerial_encode PRIVATE EWORLD_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_options(aerial_encode PRIVATE -Wall -Wextra)

add_executable(aerial_paint src/aerial_paint.cpp)
target_link_libraries(aerial_paint PRIVATE "${EHC_CORE_LIB}" ${OpenCV_LIBS} nlohmann_json::nlohmann_json)
target_include_directories(aerial_paint PRIVATE "${EHC_ROOT}/include")
target_compile_options(aerial_paint PRIVATE -Wall -Wextra)

# Dev checks against the Python-era references: the boxes one head returns
# (vs Ultralytics predict) and the words Tesseract keeps (vs the CLI).
add_executable(yolo_dump tests/yolo_dump.cpp)
target_link_libraries(yolo_dump PRIVATE eworld_core)
add_executable(ocr_dump tests/ocr_dump.cpp)
target_link_libraries(ocr_dump PRIVATE eworld_core)
# The reader bench: the same crops through Tesseract or the door, timed.
add_executable(ocr_bench tests/ocr_bench.cpp)
target_link_libraries(ocr_bench PRIVATE eworld_core)

# A conjunction asked of the frame vectors by superposition (tests/scene_probe.cpp).
add_executable(scene_probe tests/scene_probe.cpp)
target_link_libraries(scene_probe PRIVATE eworld_core)
