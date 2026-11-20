@echo off
rem Build Open Player natively on Windows into build-win\open-player.exe.
rem Needs: Visual Studio 2022+ with C++ (its CMake and Ninja are used), CUDA, the Vulkan SDK, vcpkg in C:\vcpkg
rem (or VCPKG_ROOT), and TensorRT unzipped (default C:\TensorRT\TensorRT-11.5.0.63, or pass -DTENSORRT_ROOT=...).
rem The first run builds SDL3 and FFmpeg with vcpkg, which takes a while.
setlocal
for /f "usebackq delims=" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -property installationPath`) do set "VS=%%i"
if not defined VS (echo Visual Studio not found & exit /b 1)
call "%VS%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
if not defined VCPKG_ROOT set "VCPKG_ROOT=C:\vcpkg"
cd /d "%~dp0"
rem vcpkg's FFmpeg build breaks on folders with spaces, so its packages go to one without
cmake -B build-win -S player -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_TOOLCHAIN_FILE="%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake" ^
  -DVCPKG_TARGET_TRIPLET=x64-windows -DVCPKG_INSTALLED_DIR=C:/open-player-deps %* || exit /b 1
cmake --build build-win || exit /b 1
echo Built build-win\open-player.exe
