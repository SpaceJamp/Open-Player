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
project(open-player LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(Vulkan REQUIRED)
find_package(SDL3 REQUIRED CONFIG)
add_library(ffmpeg INTERFACE)
if(WIN32)  # from vcpkg (player/vcpkg.json); Linux uses the system's FFmpeg
  find_package(FFMPEG REQUIRED)
  target_include_directories(ffmpeg INTERFACE ${FFMPEG_INCLUDE_DIRS})
  target_link_directories(ffmpeg INTERFACE ${FFMPEG_LIBRARY_DIRS})
  target_link_libraries(ffmpeg INTERFACE ${FFMPEG_LIBRARIES})
  set(TENSORRT_DEFAULT "C:/TensorRT/TensorRT-11.5.0.63")
else()
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET libavformat libavcodec libavutil libswresample libswscale)
  target_link_libraries(ffmpeg INTERFACE PkgConfig::FFMPEG)
  set(TENSORRT_DEFAULT "/opt/tensorrt")
endif()

set(TENSORRT_ROOT ${TENSORRT_DEFAULT} CACHE PATH "TensorRT folder when it is not in the system paths (Fedora tarball, Windows zip)")
find_path(TENSORRT_INCLUDE NvInfer.h HINTS ${TENSORRT_ROOT}/include REQUIRED)
# the Windows zip names its libraries after the major version (nvinfer_11.lib)
find_library(NVINFER NAMES nvinfer nvinfer_11 HINTS ${TENSORRT_ROOT}/lib REQUIRED)  # rpath to it is set automatically
find_library(NVONNXPARSER NAMES nvonnxparser nvonnxparser_11 HINTS ${TENSORRT_ROOT}/lib REQUIRED)

set(IMGUI third_party/imgui)
add_executable(open-player main.cpp media.cpp vsr.cpp display.cpp ui.cpp theme.cpp collect.cpp settings.cpp logfile.cpp update.cpp trt_loader.cpp addon.cpp training_addon.cpp training_run.cpp kernels.cu
  ${IMGUI}/imgui.cpp ${IMGUI}/imgui_draw.cpp ${IMGUI}/imgui_tables.cpp ${IMGUI}/imgui_widgets.cpp
  ${IMGUI}/backends/imgui_impl_sdl3.cpp ${IMGUI}/backends/imgui_impl_vulkan.cpp)
target_include_directories(open-player SYSTEM PRIVATE ${IMGUI} ${IMGUI}/backends)  # SYSTEM: no warnings from it
target_include_directories(open-player SYSTEM PRIVATE ${TENSORRT_INCLUDE})  # SYSTEM: hide NVIDIA's own deprecation warnings
# build number (About, --version, --check): refreshed at every build, rewritten only when it changes
add_custom_target(build_info ALL
  COMMAND ${CMAKE_COMMAND} -DSRC=${CMAKE_CURRENT_SOURCE_DIR}/.. -DOUT=${CMAKE_CURRENT_BINARY_DIR}/build_info.h
          -P ${CMAKE_CURRENT_SOURCE_DIR}/build_info.cmake
  BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/build_info.h)
add_dependencies(open-player build_info)
target_include_directories(open-player PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(open-player PRIVATE ${CMAKE_DL_LIBS} CUDA::cuda_driver Vulkan::Vulkan SDL3::SDL3 ffmpeg ${NVINFER} ${NVONNXPARSER})
if(WIN32)
  target_link_libraries(open-player PRIVATE CUDA::cudart_static)  # one DLL less to ship
  target_link_libraries(open-player PRIVATE bcrypt crypt32 winhttp)  # update checks and downloads
  target_link_libraries(open-player PRIVATE delayimp version shell32 ole32)
  # TensorRT is loaded only when needed and from known folders (trt_loader.cpp), so the player starts without it
  target_link_options(open-player PRIVATE /DELAYLOAD:nvinfer_11.dll /DELAYLOAD:nvonnxparser_11.dll)
  target_compile_definitions(open-player PRIVATE _CRT_SECURE_NO_WARNINGS)  # getenv, strerror: fine here
  target_sources(open-player PRIVATE open-player.manifest)  # UTF-8 code page for file names
  target_sources(open-player PRIVATE open-player.rc)  # the program's icon
  # a windowed program: no console window on double-click (main() still the entry; see the top of main)
  set_target_properties(open-player PROPERTIES WIN32_EXECUTABLE TRUE)
  target_link_options(open-player PRIVATE /ENTRY:mainCRTStartup)
  # TensorRT's DLLs (incl. the builder resources it loads when making an engine) next to the exe; vcpkg copies its own
  file(GLOB TENSORRT_DLLS ${TENSORRT_ROOT}/bin/nvinfer*.dll ${TENSORRT_ROOT}/bin/nvonnxparser*.dll)
  add_custom_command(TARGET open-player POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TENSORRT_DLLS} $<TARGET_FILE_DIR:open-player>)
  # NVIDIA RTX Video SDK (optional, never in this repo: its license forbids that): RTX VSR as the first upscaler
  set(RTX_VIDEO_SDK "C:/rtx_video_sdk" CACHE PATH "NVIDIA RTX Video SDK folder")
  if(EXISTS ${RTX_VIDEO_SDK}/include/nvsdk_ngx.h)
    target_sources(open-player PRIVATE rtx_vsr.cpp)
    target_compile_definitions(open-player PRIVATE HAVE_RTX_VSR)
    target_include_directories(open-player SYSTEM PRIVATE ${RTX_VIDEO_SDK}/include)
    # _d: the build against the DLL C runtime (/MD), like the rest of the player
    target_link_libraries(open-player PRIVATE ${RTX_VIDEO_SDK}/lib/Windows/x64/nvsdk_ngx_d.lib)
    target_link_options(open-player PRIVATE /FORCE:MULTIPLE)  # our cudaGetDeviceProperties for NGX, see rtx_vsr.cpp
    add_custom_command(TARGET open-player POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       ${RTX_VIDEO_SDK}/bin/Windows/x64/rel/nvngx_vsr.dll $<TARGET_FILE_DIR:open-player>)
  else()
    message(STATUS "RTX Video SDK not found in ${RTX_VIDEO_SDK}: building without NVIDIA RTX VSR")
  endif()
else()
  target_link_libraries(open-player PRIVATE CUDA::cudart)
  # DT_RPATH instead of RUNPATH: glibc also searches the program's RPATH when libnvinfer dlopen()s its builder
  # resources, which a tarball install in /opt/tensorrt needs (their odd sonames keep ldconfig from finding them)
  target_link_options(open-player PRIVATE -Wl,--disable-new-dtags)
endif()
