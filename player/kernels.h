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

// Bilinear x`scale` of planar RGB, same sampling as torch F.interpolate(align_corners=False)
void upscale_bilinear(const __half* src, int w, int h, int scale, __half* dst, cudaStream_t s);

// Mean |a-b| over n values; blocks until done. scratch: one device float.
float mean_abs_diff(const __half* a, const __half* b, int n, float* scratch, cudaStream_t s);

// Planar fp16 RGB -> packed RGBA8 for display, with a seek bar along the bottom when progress >= 0
void rgb_to_rgba8(const __half* rgb, int w, int h, uint8_t* rgba, float progress, cudaStream_t s);
