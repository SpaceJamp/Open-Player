#include "collect.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>

static const char MAGIC[8] = {'O', 'P', 'P', 'A', 'I', 'R', '1', '\0'};

bool write_pair_file(const std::string& path, int T, int lw, int lh, int hw, int hh, const uint8_t* lr,
                     const uint8_t* hr) {
    std::string tmp = path + ".tmp";  // train.py may be reading the folder: only whole files get the real name
    {
        std::ofstream f(tmp, std::ios::binary);
        int32_t head[5] = {T, lw, lh, hw, hh};
        f.write(MAGIC, sizeof MAGIC).write((const char*)head, sizeof head);
        f.write((const char*)lr, (size_t)T * 3 * lw * lh).write((const char*)hr, (size_t)T * 3 * hw * hh);
        if (!f) return false;
    }
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    return !ec;
}

bool read_pair_header(const std::string& path, int& T, int& lw, int& lh, int& hw, int& hh) {
    std::ifstream f(path, std::ios::binary);
    char magic[8];
    int32_t head[5];
    if (!f.read(magic, sizeof magic).read((char*)head, sizeof head) || memcmp(magic, MAGIC, sizeof MAGIC)) return false;
    T = head[0], lw = head[1], lh = head[2], hw = head[3], hh = head[4];
    return true;
}

#ifdef HAVE_RTX_VSR
static constexpr int CLIP = 8;            // consecutive frames per clip, like train.py --frames
static constexpr int CROP = 128;          // low-res crop; RTX VSR makes it CROP * scale
static constexpr double EVERY = 10;       // seconds of playback between clips
static constexpr uint64_t CAP = 2ull << 30;  // the folder stops growing at 2 GB

bool Collector::open(const std::string& dir, int w, int h, int out_w, int out_h, int scale, cudaStream_t s) {
    dir_ = dir, w_ = w, h_ = h, s_ = scale, st_ = s;
    if (w < CROP || h < CROP || scale < 2 || out_w != w * scale || out_h != h * scale) return false;
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    for (auto& e : std::filesystem::directory_iterator(dir, ec))
        if (e.is_regular_file(ec)) bytes_ += e.file_size(ec);
    if (bytes_ >= CAP) return false;
    lr_.resize((size_t)CLIP * 3 * CROP * CROP), hr_.resize(lr_.size() * scale * scale);
    tmp_.resize(3 * CROP * CROP * (1 + scale * scale));
    return cudaMalloc(&crop_, 2ull * tmp_.size()) == cudaSuccess;
}

void Collector::add(const __half* frame, const __half* up, double pts) {
    if (off_) return;
    if (got_ && (pts <= prev_ || pts - prev_ > 0.5)) got_ = 0;  // a seek or a gap: no longer consecutive frames
    if (!got_) {
        if (pts - last_ < EVERY) return;
        static std::mt19937 rng(std::random_device{}());
        x_ = std::uniform_int_distribution<int>(0, w_ - CROP)(rng);
        y_ = std::uniform_int_distribution<int>(0, h_ - CROP)(rng);
    }
    prev_ = pts;
    const int H = CROP * s_, n = 3 * CROP * CROP, W = w_ * s_, Hh = h_ * s_;
    for (int c = 0; c < 3; c++) {  // the same region of the frame and of its upscale
        cudaMemcpy2DAsync(crop_ + c * CROP * CROP, CROP * 2, frame + (size_t)c * w_ * h_ + (size_t)y_ * w_ + x_, w_ * 2,
                          CROP * 2, CROP, cudaMemcpyDeviceToDevice, st_);
        cudaMemcpy2DAsync(crop_ + n + c * H * H, H * 2, up + (size_t)c * W * Hh + (size_t)y_ * s_ * W + x_ * s_, W * 2,
                          H * 2, H, cudaMemcpyDeviceToDevice, st_);
    }
    cudaMemcpyAsync(tmp_.data(), crop_, tmp_.size() * 2, cudaMemcpyDeviceToHost, st_);
    // shortcut: waits for the GPU, ~1 ms for 8 frames every 10 s; use pinned memory and events if it ever shows
    cudaStreamSynchronize(st_);
    auto to8 = [](__half v) { return (uint8_t)std::clamp(__half2float(v) * 255.f + .5f, 0.f, 255.f); };
    std::transform(tmp_.begin(), tmp_.begin() + n, lr_.begin() + (size_t)got_ * n, to8);
    std::transform(tmp_.begin() + n, tmp_.end(), hr_.begin() + (size_t)got_ * 3 * H * H, to8);
    if (++got_ < CLIP) return;
    got_ = 0, last_ = pts;
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    std::string path = (std::filesystem::path(dir_) / ("pair-" + std::to_string(ms) + ".bin")).string();
    if (!write_pair_file(path, CLIP, CROP, CROP, H, H, lr_.data(), hr_.data()))
        return (void)(off_ = true, fprintf(stderr, "collect: cannot write %s, stopping\n", path.c_str()));
    if ((bytes_ += 28 + lr_.size() + hr_.size()) >= CAP)
        off_ = true, fprintf(stderr, "collect: %s reached 2 GB, stopping\n", dir_.c_str());
}

Collector::~Collector() { cudaFree(crop_); }
#endif
