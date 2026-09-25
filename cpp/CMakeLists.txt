# CDM on TensorRT: C++ runtime and tools.
#
#   cdm_preprocessing  infer_depth's pre/post-processing (OpenCV only)
#   cdm_tensorrt       the TensorRT runtime (DepthCompleter)
#   cdm_dataset        run over a recorded sequence
#   cdm_realsense      run live on a RealSense (built when librealsense2 is found)
#   cdm_probe          check an engine against the Python reference
#   test_preprocessing unit test, no GPU (ctest)
cmake_minimum_required(VERSION 3.18)
project(cdm_tensorrt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs highgui)
find_package(CUDAToolkit REQUIRED)
find_package(yaml-cpp REQUIRED)

set(TENSORRT_ROOT "" CACHE PATH "TensorRT install root (default: system paths)")
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS "${TENSORRT_ROOT}/include" /usr/include/x86_64-linux-gnu /usr/include/aarch64-linux-gnu
    PATH_SUFFIXES tensorrt)
find_library(TENSORRT_LIBRARY nvinfer
    HINTS "${TENSORRT_ROOT}/lib" "${TENSORRT_ROOT}/lib64"
          /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu)
if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
    message(FATAL_ERROR "TensorRT not found; set -DTENSORRT_ROOT=<install>")
endif()
message(STATUS "TensorRT: ${TENSORRT_LIBRARY}")

add_library(cdm_preprocessing STATIC src/preprocessing.cpp src/io.cpp)
target_include_directories(cdm_preprocessing PUBLIC include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(cdm_preprocessing PUBLIC ${OpenCV_LIBS})

add_library(cdm_tensorrt STATIC src/depth_completer.cpp)
target_include_directories(cdm_tensorrt PRIVATE ${TENSORRT_INCLUDE_DIR})
target_link_libraries(cdm_tensorrt
    PUBLIC cdm_preprocessing CUDA::cudart
    PRIVATE ${TENSORRT_LIBRARY})

add_executable(cdm_dataset apps/cdm_dataset.cpp)
target_link_libraries(cdm_dataset PRIVATE cdm_tensorrt yaml-cpp::yaml-cpp)

add_executable(cdm_probe apps/cdm_probe.cpp)
target_link_libraries(cdm_probe PRIVATE cdm_tensorrt yaml-cpp::yaml-cpp)

find_package(realsense2 QUIET)
if(realsense2_FOUND)
    add_executable(cdm_realsense apps/cdm_realsense.cpp)
    target_link_libraries(cdm_realsense PRIVATE cdm_tensorrt realsense2::realsense2 yaml-cpp::yaml-cpp)
    message(STATUS "librealsense2 ${realsense2_VERSION}: building cdm_realsense")
else()
    message(STATUS "librealsense2 not found: cdm_realsense is not built")
endif()

enable_testing()
add_executable(test_preprocessing tests/test_preprocessing.cpp)
target_link_libraries(test_preprocessing PRIVATE cdm_preprocessing)
add_test(NAME preprocessing COMMAND test_preprocessing)

install(TARGETS cdm_dataset cdm_probe RUNTIME DESTINATION bin)
if(TARGET cdm_realsense)
    install(TARGETS cdm_realsense RUNTIME DESTINATION bin)
endif()
