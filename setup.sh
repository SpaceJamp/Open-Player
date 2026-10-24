#!/usr/bin/env bash
# One-time setup: CUDA 13.4, TensorRT 11, and the FFmpeg/SDL3/Vulkan dev files the player builds against.
# Arch, Fedora 44, Ubuntu 26.04 (also WSL2). Installs no GPU driver: the NVIDIA driver must already be there.
set -euo pipefail
. /etc/os-release
TRT_VER=11.4.0.106
TRT_URL="https://developer.download.nvidia.com/compute/machine-learning/tensorrt/${TRT_VER%.*}/tars/TensorRT-Enterprise-${TRT_VER}-Linux-x86_64-cuda-13.4-Release-external.tar.zst"

# NVIDIA's generic TensorRT tarball (3.7 GB) into /opt/tensorrt, where the player's CMake looks.
# Used on Arch too: the AUR package pulls nvidia-utils, which breaks CUDA under WSL, and builds a lot we don't use.
tensorrt_tarball() {
  [ -e /opt/tensorrt/include/NvInfer.h ] && return
  curl -L --fail "$TRT_URL" -o /tmp/tensorrt.tar.zst
  sudo mkdir -p /opt/tensorrt
  sudo tar --zstd -xf /tmp/tensorrt.tar.zst -C /opt/tensorrt --strip-components=1
  rm /tmp/tensorrt.tar.zst
}

case " $ID $ID_LIKE " in
*" arch "*)
  extra=()
  grep -qi microsoft /proc/version && extra+=(vulkan-dzn)  # WSL: Vulkan goes through Mesa's D3D12 driver
  sudo pacman -S --needed base-devel cmake cuda sdl3 ffmpeg vulkan-headers vulkan-icd-loader python mpv "${extra[@]}"
  tensorrt_tarball
  ;;
*" fedora "*)
  sudo dnf config-manager addrepo --overwrite \
    --from-repofile="https://developer.download.nvidia.com/compute/cuda/repos/fedora$VERSION_ID/x86_64/cuda-fedora$VERSION_ID.repo"
  # Fedora's own ffmpeg-free has no H.264/HEVC decoding; RPM Fusion's full build has those and NVDEC
  sudo dnf install -y "https://mirrors.rpmfusion.org/free/fedora/rpmfusion-free-release-$VERSION_ID.noarch.rpm"
  sudo dnf swap -y ffmpeg-free ffmpeg --allowerasing
  sudo dnf install -y cuda-toolkit-13-4 gcc-c++ cmake pkgconf SDL3-devel vulkan-loader-devel vulkan-headers \
    ffmpeg-devel python3 mpv zstd
  tensorrt_tarball  # NVIDIA packages TensorRT for RHEL, not Fedora
  ;;
*" ubuntu "*)
  cd "$(mktemp -d)"
  wget -q "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${VERSION_ID//./}/x86_64/cuda-keyring_1.1-1_all.deb"
  sudo dpkg -i cuda-keyring_1.1-1_all.deb
  sudo apt-get update
  sudo apt-get install -y cuda-toolkit-13-4 tensorrt-dev build-essential cmake pkg-config libsdl3-dev libvulkan-dev \
    mesa-vulkan-drivers libavformat-dev libavcodec-dev libavutil-dev libswresample-dev libswscale-dev ffmpeg mpv python3-venv
  ;;
*)
  echo "unsupported distro: $ID. Needed: CUDA 13.4+, TensorRT 10+, SDL3, FFmpeg dev libs, Vulkan headers, CMake"
  exit 1
  ;;
esac
echo "setup done"
