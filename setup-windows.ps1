# One-time setup on Windows: what build-windows.bat and train.py need, like setup.sh on Linux.
# Installs with winget what it can (Build Tools 2022 with C++, CUDA, Vulkan SDK, Git, Python, FFmpeg), clones vcpkg
# to C:\vcpkg and pip-installs PyTorch. TensorRT and the NVIDIA RTX Video SDK need an NVIDIA login, so it only checks
# them and says where to get them. Skips anything already there; safe to run again. Install the NVIDIA driver
# first (NVIDIA's CUDA installer may also update it).
#
#   powershell -ExecutionPolicy Bypass -File setup-windows.ps1            # install what's missing
#   powershell -ExecutionPolicy Bypass -File setup-windows.ps1 -Check     # only report
#   ... -NoTraining                                                       # skip Python, PyTorch and FFmpeg
param([switch]$Check, [switch]$NoTraining)
$ErrorActionPreference = "Continue"  # "Stop" makes Windows PowerShell 5.1 fail on any native tool's stderr
$TensorRT = "C:\TensorRT\TensorRT-11.5.0.63"  # build-windows.bat's default; other versions: -DTENSORRT_ROOT=...
$RtxSdk = "C:\rtx_video_sdk"
$Vcpkg = if ($env:OPEN_PLAYER_VCPKG) { $env:OPEN_PLAYER_VCPKG } else { "C:\vcpkg" }
if ($Vcpkg -match ' ') { throw "OPEN_PLAYER_VCPKG ($Vcpkg) must have no spaces: vcpkg's FFmpeg build breaks on them" }
$missing = @()

function Have($cmd) { [bool](Get-Command $cmd -ErrorAction SilentlyContinue) }

function Update-Path {  # winget installs change PATH in the registry, not in this process
    $m = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $u = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = "$m;$u;$env:Path"  # keeps what this shell had (a portable Git, say)
    foreach ($v in "CUDA_PATH", "VULKAN_SDK") {
        $x = [Environment]::GetEnvironmentVariable($v, "Machine")
        if ($x) { Set-Item "env:$v" $x }
    }
}

# Reports one item. $ok: a script block that says whether it is installed; $fix: one that installs it.
function Step($name, $ok, $fix, $hint) {
    if (& $ok) { Write-Host "  ok       $name"; return }
    if ($Check -or -not $fix) {
        Write-Host "  MISSING  $name$(if ($hint) { " - $hint" })" -ForegroundColor Yellow
        $script:missing += $name
        return
    }
    Write-Host "  install  $name" -ForegroundColor Cyan
    & $fix
    Update-Path
    if (& $ok) { Write-Host "  ok       $name" }
    else {
        Write-Host "  FAILED   $name$(if ($hint) { " - $hint" })" -ForegroundColor Red
        $script:missing += $name
    }
}

function Winget($id, [string[]]$more = @()) {
    winget install --id $id --exact --silent --accept-package-agreements --accept-source-agreements @more
    # 0: installed; -1978335189: already installed / no newer version. Anything else shows up in the re-check.
}

function VS-Path {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) { return $null }
    # the same query as build-windows.bat: C++ and the CMake tools, whose Ninja it uses
    & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        Microsoft.VisualStudio.Component.VC.CMake.Project -property installationPath
}

function Install-VS {
    # --passive alone may reboot the PC when it finishes
    $vcTools = "--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --passive --wait --norestart"
    $inst = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\setup.exe"
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    $bt = if (Test-Path $vswhere) { & $vswhere -products Microsoft.VisualStudio.Product.BuildTools -property installationPath | Select-Object -First 1 }
    if ($bt) {  # Build Tools without the C++ part: winget would say "already installed", so add it directly
        Start-Process $inst -ArgumentList "modify --installPath `"$bt`" $vcTools" -Wait -Verb RunAs
    } else {
        Winget Microsoft.VisualStudio.2022.BuildTools @("--override", $vcTools)
    }
}

function Torch-Ok {
    if (-not (Have py)) { return $false }
    py -c "import torch, numpy, onnx, onnxscript; assert torch.cuda.is_available()" 2>$null
    $LASTEXITCODE -eq 0
}

Write-Host "Open Player setup (Windows)"
if (-not (Have nvidia-smi)) {
    Write-Host "  MISSING  NVIDIA driver - install it from https://www.nvidia.com/drivers first" -ForegroundColor Yellow
    $missing += "NVIDIA driver"
}
if (-not $Check -and -not (Have winget)) {
    throw "winget not found: install 'App Installer' from the Microsoft Store, then run this again"
}

Step "Visual Studio Build Tools with C++" { [bool](VS-Path) } { Install-VS }
Step "CUDA Toolkit" { $env:CUDA_PATH -and (Test-Path "$env:CUDA_PATH\bin\nvcc.exe") } { Winget Nvidia.CUDA }
Step "Vulkan SDK" { $env:VULKAN_SDK -and (Test-Path "$env:VULKAN_SDK\Include\vulkan\vulkan.h") } { Winget KhronosGroup.VulkanSDK }
Step "Git" { Have git } { Winget Git.Git }
Step "Inno Setup 6 (for installer\build-installer.bat)" {
    (Test-Path "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe") -or (Test-Path "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe")
} { Winget JRSoftware.InnoSetup }
Step "vcpkg in $Vcpkg" { Test-Path "$Vcpkg\vcpkg.exe" } {
    if (-not (Test-Path "$Vcpkg\bootstrap-vcpkg.bat")) {
        if (Test-Path $Vcpkg) { Write-Host "  $Vcpkg is incomplete (an interrupted clone?): delete it and run this again"; return }
        git clone https://github.com/microsoft/vcpkg $Vcpkg
    }
    & "$Vcpkg\bootstrap-vcpkg.bat" -disableMetrics
}
Step "TensorRT in $TensorRT" { (Test-Path "$TensorRT\include\NvInfer.h") -and (Test-Path "$TensorRT\lib\nvinfer_11.lib") } $null `
    "download TensorRT 11.5 for Windows (zip) from https://developer.nvidia.com/tensorrt and unzip it to C:\TensorRT"
$rtx = (Test-Path "$RtxSdk\include\nvsdk_ngx.h") -and (Test-Path "$RtxSdk\lib\Windows\x64\nvsdk_ngx_d.lib") -and
    (Test-Path "$RtxSdk\bin\Windows\x64\rel\nvngx_vsr.dll")  # what player/CMakeLists.txt looks for
if ($rtx) { Write-Host "  ok       NVIDIA RTX Video SDK in $RtxSdk" }
else {
    Write-Host "  optional NVIDIA RTX Video SDK - for RTX VSR, unzip it from https://developer.nvidia.com/rtx-video-sdk to $RtxSdk"
}

if (-not $NoTraining) {
    # per user, with the py launcher, no admin needed (the default asks for admin for the launcher)
    Step "Python" { Have py } { Winget Python.Python.3.13 @("--override",
        "/quiet InstallAllUsers=0 InstallLauncherAllUsers=0 Include_launcher=1 PrependPath=1") }
    Step "PyTorch with CUDA, numpy, onnx" { Torch-Ok } {
        # torch alone from NVIDIA's CUDA index: with PyPI as an extra index pip takes PyPI's newer, CPU-only build
        py -m pip install torch --index-url https://download.pytorch.org/whl/cu130
        py -m pip install numpy onnx onnxscript
    } ("needs a driver new enough for CUDA 13; a CPU-only torch already installed stays: " +
       "run 'py -m pip uninstall torch' and then this again")
    Step "FFmpeg on the PATH" { Have ffmpeg } { Winget Gyan.FFmpeg }
}

if ($missing.Count) {
    Write-Host "`nStill missing: $($missing -join ', ')" -ForegroundColor Yellow
    exit 1
}
Write-Host "`nAll set. Open a new terminal (for the new PATH), then run build-windows.bat"
