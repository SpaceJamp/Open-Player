#pragma once
// The Training add-on: what train.py needs on a PC without Python. python.org's embeddable Python, the wheels of
// PyTorch (CUDA) and its dependencies, and FFmpeg's ffmpeg.exe/ffprobe.exe, each downloaded from its own server,
// checked against a SHA-256 fixed in this source and unpacked (wheels are zips: no pip, nothing runs) into
// training_addon_dir(). Only people who train their own Pixel Flow need it.
#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

struct TrainingPin {
    const char* url;
    const char* sha256;
    uint64_t size;
    const char* kind;  // "python": the embeddable zip -> python\; "wheel" -> python\Lib\site-packages\;
                       // "ffmpeg": a release zip -> ffmpeg\ (ffmpeg.exe, ffprobe.exe and the licence only)
};
// Raise whenever any pin changes: an installed add-on with another number is offered as an update.
inline constexpr int TRAINING_ADDON_VERSION = 1;
extern const std::vector<TrainingPin> TRAINING_PINS;

uint64_t training_addon_size();  // all downloads together, for the "3.5 GB" text
int training_addon_installed();  // 0: none, 1: current, 2: outdated (another TRAINING_ADDON_VERSION)
std::string training_python();      // training_addon_dir()\python\python.exe
std::string training_ffmpeg_dir();  // training_addon_dir()\ffmpeg
// Where train.py and the other scripts are: <program>\training (installed), else the repo (a developer build).
std::string training_scripts_dir();
// Points the embeddable Python at its packages and at the scripts folder (its python313._pth), in python_dir.
bool write_python_pth(const std::string& python_dir, const std::string& scripts_dir);

// Unpacks one downloaded, verified file of the given kind into an add-on folder being built (into).
bool unpack_training_file(const TrainingPin& pin, const std::string& file, const std::string& into,
                          std::string& error);
// Download (progress: bytes so far, total), hash check, unpack each file; build in training.new, then move it into
// place. false + error on any failure or cancel, leaving no training.new and the old add-on as it was.
bool install_training_addon(const std::function<void(uint64_t, uint64_t)>& progress, const std::atomic<bool>& cancel,
                            std::string& error);
// in_use (its python.exe is running): marks it for removal at the next start instead.
void remove_training_addon(bool in_use);
void finish_pending_training_removal();  // at start
int training_addon_selftest();           // part of --uitest; 0 on pass
