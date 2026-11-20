@echo off
rem Build Open Player natively on Windows into build-win\open-player.exe.
rem Needs: Visual Studio 2022+ with C++ (its CMake and Ninja are used), CUDA, the Vulkan SDK, vcpkg in C:\vcpkg
rem (or VCPKG_ROOT), and TensorRT unzipped (default C:\TensorRT\TensorRT-11.5.0.63, or pass -DTENSORRT_ROOT=...).
rem The first run builds SDL3 and FFmpeg with vcpkg, which takes a while.
setlocal
for /f "usebackq delims=" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VS=%%i"
if not defined VS (echo No Visual Studio with the C++ tools found & exit /b 1)
call "%VS%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
rem not VCPKG_ROOT: vcvars points that at Visual Studio's own vcpkg
if not defined OPEN_PLAYER_VCPKG set "OPEN_PLAYER_VCPKG=C:\vcpkg"
cd /d "%~dp0"
rem vcpkg's FFmpeg build breaks on folders with spaces, so its packages go to one without
cmake -B build-win -S player -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_MAKE_PROGRAM="%VS%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe" ^
  -DCMAKE_TOOLCHAIN_FILE="%OPEN_PLAYER_VCPKG%\scripts\buildsystems\vcpkg.cmake" ^
  -DVCPKG_TARGET_TRIPLET=x64-windows -DVCPKG_INSTALLED_DIR=C:/open-player-deps %* || exit /b 1
cmake --build build-win || exit /b 1
echo Built build-win\open-player.exe
