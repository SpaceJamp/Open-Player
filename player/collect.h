#pragma once
// --collect: while NVIDIA RTX VSR plays a video, saves short clips of the decoded frames next to RTX VSR's exact 2x
// upscale of them, for train.py --pairs. Our model then learns to imitate RTX VSR, which runs only on Windows, so the
// result also upscales that way on Linux.
//
// Pair file: "OPPAIR1\0", then int32 T, lr_w, lr_h, hr_w, hr_h (little-endian), then the low-res frames and then the
// RTX upscales, each T x 3 x h x w uint8 (planar RGB).
#include <cstdint>
#include <string>
#include <vector>

#include "kernels.h"

bool write_pair_file(const std::string& path, int T, int lw, int lh, int hw, int hh, const uint8_t* lr,
                     const uint8_t* hr);
bool read_pair_header(const std::string& path, int& T, int& lw, int& lh, int& hw, int& hh);

#ifdef HAVE_RTX_VSR
// Crops come from playback's own RTX VSR output (a second RTX VSR instance can't be created), so it only collects
// when that output is exactly scale x the frame, not when the video is shrunk to fit the screen first.
class Collector {
public:
    ~Collector();
    // false: playback's upscale isn't exactly 2x, the video is smaller than a crop, or the folder is unusable or full
    bool open(const std::string& dir, int w, int h, int out_w, int out_h, int scale, double fps, cudaStream_t s);
    // the frame just decoded and RTX VSR's upscale of it (planar fp16 RGB), at pts seconds; every frame while playing
    void add(const __half* frame, const __half* up, double pts);
    // a seek or pause: drop the clip in progress (its frames would not be consecutive)
    void restart() { got_ = 0; }

private:
    std::string dir_;
    int w_ = 0, h_ = 0, s_ = 0, got_ = 0, x_ = 0, y_ = 0;
    double last_ = -1e9, prev_ = 0, gap_ = 0.5;  // gap_: longer than this between frames means one was dropped
    uint64_t bytes_ = 0;  // in the folder so far
    bool off_ = false;
    cudaStream_t st_ = nullptr;
    __half* crop_ = nullptr;  // low-res crop then its upscale, on the GPU
    std::vector<__half> tmp_;
    std::vector<uint8_t> lr_, hr_;
};
#endif
