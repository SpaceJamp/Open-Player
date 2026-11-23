#pragma once
// Finding TensorRT at run time. On Windows the player starts without it (its DLLs are delay-loaded) and loads it
// only from known folders: the user's own TensorRT 11.x folder (Settings), the add-on folder, or next to the
// program (developer builds). Plus the version lines shown in Settings > About, the log and --check.
#include <string>
#include <vector>

enum class TrtFrom { None, UserFolder, AddOn, NextToExe };

// Folders to look in, in order; empty ones skipped.
std::vector<std::string> tensorrt_candidates(const std::string& user_folder, const std::string& addon,
                                             const std::string& exe_dir);
// Loads TensorRT 11 from the first candidate that has it. version: "11.5.0"; why_not: the reason when None.
TrtFrom load_tensorrt(const std::string& user_folder, std::string& version, std::string& why_not);
std::string trt_line(TrtFrom from, const std::string& version, const std::string& why_not);  // "TensorRT: ..."

std::string user_data_dir();  // %LOCALAPPDATA%\Open Player (Linux: ~/.local/share/open-player): the user's model
std::string training_addon_dir();  // user_data_dir()\training: the Training add-on (Python, PyTorch, FFmpeg)
std::string training_work_dir();   // user_data_dir()\training-work: the player's own training run
std::string program_dir();         // the folder open-player.exe is in
std::string addon_dir();     // user_data_dir()\tensorrt
bool installed_copy();        // installed by the installer (installed.txt next to the program)

// "NVIDIA RTX VSR: <nvngx_vsr.dll version> (SDK API 1.5.0)"; ", not in use" when not opened; why not when it failed
std::string rtx_vsr_line(bool tried, bool available);
std::string driver_line();       // "NVIDIA driver: CUDA 13.5"
int trt_loader_selftest();       // part of --uitest; 0 on pass
