@echo off
rem Builds the release player (build-windows.bat) and its installer: installer\out\OpenPlayer-Setup-<build>.exe.
rem Needs Inno Setup 6 (setup-windows.ps1 installs it).
setlocal
pushd "%~dp0.." || exit /b 1
rem by full path: this folder isn't searched for programs when NoDefaultCurrentDirectoryInExePath is set
call "%~dp0..\build-windows.bat" || exit /b 1
for /f "tokens=4" %%b in ('build-win\open-player.exe --version') do set "BUILD=%%b"
if "%BUILD%"=="unknown" (echo No build number: build from a git checkout & exit /b 1)
for /f "tokens=*" %%t in ('build-win\open-player.exe --version ^| findstr /c:"TEST BUILD"') do (echo Refusing a TEST build & exit /b 1)
for /f "usebackq delims=" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VS=%%i"
for /d %%d in ("%VS%\VC\Redist\MSVC\*") do if exist "%%d\x64\Microsoft.VC143.CRT\vcruntime140.dll" set "VCREDIST=%%d\x64\Microsoft.VC143.CRT"
rem no %VS% inside the parentheses: the ")" in "Program Files (x86)" would end the block
if not defined VCREDIST (echo VC++ runtime DLLs not found in Visual Studio's Redist folder & exit /b 1)
set "ISCC=%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"
if not exist "%ISCC%" set "ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if not exist "%ISCC%" (echo Inno Setup 6 not found: winget install JRSoftware.InnoSetup & exit /b 1)
"%ISCC%" /Q /DBuild=%BUILD% "/DVCRedist=%VCREDIST%" installer\open-player.iss || exit /b 1
echo Built installer\out\OpenPlayer-Setup-%BUILD%.exe
