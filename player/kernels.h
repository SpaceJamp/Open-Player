#pragma once
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// A decoded frame already in GPU memory: NV12 (8-bit) or P010/P016 (16-bit, value in the high bits).
struct YuvFrame {
    const uint8_t *y, *uv;
    int pitch_y, pitch_uv, w, h;
    bool sixteen_bit, bt709, full_range;
};

// -> planar fp16 RGB (3 x h x w) in [0,1], the layout the network takes
void yuv_to_rgb(const YuvFrame& f, __half* rgb, cudaStream_t s);

// Starts the VSR history over from `cur`: prev = cur, prev_hr = bilinear x`scale` of cur. With diff_sum set,
// only when *diff_sum > max_sum (a scene cut), decided on the GPU so the CPU never waits for it.
void reset_state(const __half* cur, __half* prev, __half* prev_hr, int w, int h, int scale, const float* diff_sum,
                 float max_sum, cudaStream_t s);

// Planar RGB w x h -> W x H, smaller by up to 2x (2x2 bilinear samples per output pixel)
void shrink_rgb(const __half* src, int w, int h, __half* dst, int W, int H, cudaStream_t s);

// *sum = sum of |a-b| over n values (sum: one device float), without waiting
void abs_diff_sum(const __half* a, const __half* b, int n, float* sum, cudaStream_t s);

// Planar fp16 RGB -> packed RGBA8 for display, with a seek bar along the bottom when progress >= 0
void rgb_to_rgba8(const __half* rgb, int w, int h, uint8_t* rgba, float progress, cudaStream_t s);
