cmake_minimum_required(VERSION 3.24)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)  # the GPU in this machine; pass e.g. -DCMAKE_CUDA_ARCHITECTURES=120 to cross-build
endif()
if(NOT DEFINED CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX})
  # nvcc is often not on PATH: Arch puts CUDA in /opt/cuda, NVIDIA's packages in /usr/local/cuda
  find_program(NVCC nvcc HINTS /opt/cuda/bin /usr/local/cuda/bin /usr/local/cuda-13.4/bin)
  if(NVCC)
    set(CMAKE_CUDA_COMPILER ${NVCC})
  endif()
endif()
project(vsrplayer LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(Vulkan REQUIRED)
find_package(SDL3 REQUIRED CONFIG)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET libavformat libavcodec libavutil libswresample libswscale)

set(TENSORRT_ROOT "/opt/tensorrt" CACHE PATH "TensorRT folder when it is not in the system paths (Fedora tarball)")
find_path(TENSORRT_INCLUDE NvInfer.h HINTS ${TENSORRT_ROOT}/include REQUIRED)
find_library(NVINFER nvinfer HINTS ${TENSORRT_ROOT}/lib REQUIRED)  # rpath to it is set automatically
find_library(NVONNXPARSER nvonnxparser HINTS ${TENSORRT_ROOT}/lib REQUIRED)

add_executable(vsrplayer main.cpp media.cpp vsr.cpp display.cpp kernels.cu)
target_include_directories(vsrplayer SYSTEM PRIVATE ${TENSORRT_INCLUDE})  # SYSTEM: hide NVIDIA's own deprecation warnings
# DT_RPATH instead of RUNPATH: glibc also searches the program's RPATH when libnvinfer dlopen()s its builder
# resources, which a tarball install in /opt/tensorrt needs (their odd sonames keep ldconfig from finding them)
target_link_options(vsrplayer PRIVATE -Wl,--disable-new-dtags)
target_link_libraries(vsrplayer PRIVATE ${CMAKE_DL_LIBS} CUDA::cudart Vulkan::Vulkan SDL3::SDL3 PkgConfig::FFMPEG ${NVINFER} ${NVONNXPARSER})
