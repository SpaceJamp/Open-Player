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
#include "rtx_vsr.h"

class Collector {
public:
    ~Collector();
    // false: no RTX VSR for the crop, the video is smaller than a crop, or the folder is already at its cap
    bool open(const std::string& dir, int w, int h, int scale, cudaStream_t s);
    // the frame just decoded (planar fp16 RGB, w x h) at pts seconds; call for every frame while playing
    void add(const __half* frame, double pts);

private:
    std::string dir_;
    int w_ = 0, h_ = 0, s_ = 0, got_ = 0, x_ = 0, y_ = 0;
    double last_ = -1e9, prev_ = 0;
    uint64_t bytes_ = 0;  // in the folder so far
    bool off_ = false;
    cudaStream_t st_ = nullptr;
    RtxVsr rtx_;
    __half* crop_ = nullptr;
    std::vector<__half> tmp_;
    std::vector<uint8_t> lr_, hr_;
};
#endif
