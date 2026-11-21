#pragma once
// The TensorRT add-on: NVIDIA's TensorRT 11.5 runtime for Windows, downloaded from NVIDIA's own package index
// (pypi.nvidia.com, no login), checked against a SHA-256 fixed in this source, and unpacked (DLLs and NVIDIA's
// licence only) into addon_dir(). Only people who train their own Pixel Flow need it.
#include <atomic>
#include <cstdint>
#include <functional>
#include <string>

inline constexpr const char* TRT_WHEEL_URL =
    "https://pypi.nvidia.com/tensorrt-cu13-libs/tensorrt_cu13_libs-11.5.0.63-py3-none-win_amd64.whl";
inline constexpr const char* TRT_WHEEL_SHA256 = "d656f2e523e822993915ef8d7b70557afab69182f19315ddc8c326eacef2d645";
inline constexpr uint64_t TRT_WHEEL_SIZE = 1958038086;

// Unpacks tensorrt_libs/*.dll and the licence from a wheel (a zip) into dir, through dir + ".new" so a failure
// leaves no half add-on. Uses Windows' own tar.exe; refuses anything that would land outside dir.
bool extract_tensorrt(const std::string& wheel, const std::string& dir, std::string& error);
// Download (progress: bytes so far, total), hash check, unpack, clean up. false + error on any failure or cancel.
bool install_tensorrt_addon(const std::function<void(uint64_t, uint64_t)>& progress, const std::atomic<bool>& cancel,
                            std::string& error);
// Removes the add-on folder; in_use (its DLLs are loaded): marks it for removal at the next start instead.
void remove_tensorrt_addon(bool in_use);
void finish_pending_addon_removal();  // at start, before TensorRT loads
int addon_selftest();                 // part of --uitest; 0 on pass
