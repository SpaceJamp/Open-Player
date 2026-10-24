#include "kernels.h"

static dim3 grid2d(int w, int h) { return dim3((w + 15) / 16, (h + 15) / 16); }

template <typename T>
__global__ void k_yuv_to_rgb(YuvFrame f, __half* rgb) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= f.w || y >= f.h) return;
    const float norm = sizeof(T) == 2 ? 1.f / 256.f : 1.f;  // bring 16-bit samples to the 8-bit scale
    float Y = *(const T*)(f.y + y * f.pitch_y + x * sizeof(T)) * norm;
    const T* c = (const T*)(f.uv + (y / 2) * f.pitch_uv) + (x / 2) * 2;  // shortcut: nearest chroma, fine at 4:2:0
    float U = c[0] * norm - 128.f, V = c[1] * norm - 128.f;
    if (f.full_range) Y /= 255.f, U /= 255.f, V /= 255.f;
    else Y = (Y - 16.f) / 219.f, U /= 224.f, V /= 224.f;
    // shortcut: SDR only; HDR (PQ/HLG) sources play with washed-out colors until tone mapping is added
    float r, g, b;
    if (f.bt709) r = Y + 1.5748f * V, g = Y - 0.1873f * U - 0.4681f * V, b = Y + 1.8556f * U;
    else r = Y + 1.402f * V, g = Y - 0.344136f * U - 0.714136f * V, b = Y + 1.772f * U;
    int n = f.w * f.h, i = y * f.w + x;
    rgb[i] = __float2half(__saturatef(r));
    rgb[n + i] = __float2half(__saturatef(g));
    rgb[2 * n + i] = __float2half(__saturatef(b));
}

void yuv_to_rgb(const YuvFrame& f, __half* rgb, cudaStream_t s) {
    if (f.sixteen_bit) k_yuv_to_rgb<uint16_t><<<grid2d(f.w, f.h), dim3(16, 16), 0, s>>>(f, rgb);
    else k_yuv_to_rgb<uint8_t><<<grid2d(f.w, f.h), dim3(16, 16), 0, s>>>(f, rgb);
}

__global__ void k_reset(const __half* cur, __half* prev, __half* prev_hr, int w, int h, int scale,
                        const float* diff_sum, float max_sum) {
    if (diff_sum && *diff_sum <= max_sum) return;  // no scene cut: keep the history
    int W = w * scale, H = h * scale;
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    // prev_hr = bilinear upscale of cur, same sampling as torch F.interpolate(align_corners=False)
    float sx = fmaxf((x + 0.5f) / scale - 0.5f, 0.f), sy = fmaxf((y + 0.5f) / scale - 0.5f, 0.f);
    int x0 = (int)sx, y0 = (int)sy, x1 = min(x0 + 1, w - 1), y1 = min(y0 + 1, h - 1);
    float fx = sx - x0, fy = sy - y0;
    for (int c = 0; c < 3; c++) {
        const __half* p = cur + c * w * h;
        float top = __half2float(p[y0 * w + x0]) * (1 - fx) + __half2float(p[y0 * w + x1]) * fx;
        float bot = __half2float(p[y1 * w + x0]) * (1 - fx) + __half2float(p[y1 * w + x1]) * fx;
        prev_hr[c * W * H + y * W + x] = __float2half(top * (1 - fy) + bot * fy);
        if (x < w && y < h) prev[c * w * h + y * w + x] = p[y * w + x];  // prev = cur
    }
}

void reset_state(const __half* cur, __half* prev, __half* prev_hr, int w, int h, int scale, const float* diff_sum,
                 float max_sum, cudaStream_t s) {
    k_reset<<<grid2d(w * scale, h * scale), dim3(16, 16), 0, s>>>(cur, prev, prev_hr, w, h, scale, diff_sum, max_sum);
}

__global__ void k_absdiff(const __half* a, const __half* b, int n, float* sum) {
    float acc = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        acc += fabsf(__half2float(a[i]) - __half2float(b[i]));
    for (int o = 16; o; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
    if ((threadIdx.x & 31) == 0) atomicAdd(sum, acc);
}

void abs_diff_sum(const __half* a, const __half* b, int n, float* sum, cudaStream_t s) {
    cudaMemsetAsync(sum, 0, sizeof(float), s);
    k_absdiff<<<256, 256, 0, s>>>(a, b, n, sum);
}

__global__ void k_rgba(const __half* rgb, int w, int h, uint8_t* out, float progress) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int n = w * h, i = y * w + x;
    float r = __half2float(rgb[i]), g = __half2float(rgb[n + i]), b = __half2float(rgb[2 * n + i]);
    int bar = max(4, h / 120);
    if (progress >= 0 && y >= h - bar) {
        if (x < progress * w) r = 0.95f, g = 0.8f, b = 0.3f;  // played part: amber, visible on white video too
        else r *= 0.35f, g *= 0.35f, b *= 0.35f;              // rest: dimmed video
    }
    uchar4 px = make_uchar4(__saturatef(r) * 255.f + .5f, __saturatef(g) * 255.f + .5f, __saturatef(b) * 255.f + .5f, 255);
    ((uchar4*)out)[i] = px;
}

void rgb_to_rgba8(const __half* rgb, int w, int h, uint8_t* rgba, float progress, cudaStream_t s) {
    k_rgba<<<grid2d(w, h), dim3(16, 16), 0, s>>>(rgb, w, h, rgba, progress);
}
