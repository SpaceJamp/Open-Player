@echo off
rem Play a video with Open Player from Windows: drag a video file onto this, or run  open-player.bat "D:\movie.mkv"
rem The player runs in the Arch WSL distro and uses vsr.onnx next to this file once you've trained and exported one.
setlocal
if "%~1"=="" (echo Drag a video file onto open-player.bat & pause & exit /b 1)
set "VIDEO=%~f1"
if "%VIDEO:~0,2%"=="\\" goto wsl_file
for /f "delims=" %%p in ('wsl -d Arch wslpath -a "%VIDEO%"') do set "VIDEO=%%p"
goto play
:wsl_file
rem a file inside WSL (\\wsl.localhost\Arch\... or \\wsl$\Arch\...): wslpath can't map those, the rest is the Linux path
set "VIDEO=%VIDEO:\\wsl.localhost\Arch=%"
set "VIDEO=%VIDEO:\\wsl$\Arch=%"
set "VIDEO=%VIDEO:\=/%"
:play
wsl -d Arch --cd "%~dp0." ./build/open-player "%VIDEO%"
if errorlevel 1 pause
