#pragma once
// NVIDIA RTX Video Super Resolution through the RTX Video SDK (Windows only): the driver's own upscaler. The player
// uses it before our network when the GPU and driver support it.
#include "kernels.h"

struct NVSDK_NGX_Handle;
struct NVSDK_NGX_Parameter;

class RtxVsr {
public:
    ~RtxVsr();
    // false when this GPU or driver has no RTX VSR (it then stays unused)
    bool open(int w, int h, int out_w, int out_h, cudaStream_t s);
    // Upscales planar fp16 RGB (3 x h x w) to 3 x out_h x out_w; valid until the next run(). nullptr if NVIDIA's
    // upscaler failed.
    const __half* run(const __half* in);
    // --check: upscales a test pattern; 0 when the output matches it
    static int selftest();

private:
    int w_ = 0, h_ = 0, ow_ = 0, oh_ = 0, i_ = 0;
    cudaStream_t s_ = nullptr;
    NVSDK_NGX_Parameter* params_ = nullptr;
    NVSDK_NGX_Handle* feature_ = nullptr;
    bool inited_ = false;
    uint8_t *rgba_in_ = nullptr, *rgba_out_ = nullptr;
    cudaArray_t src_ = nullptr, dst_ = nullptr;
    cudaTextureObject_t tex_ = 0;
    cudaSurfaceObject_t surf_ = 0;
    __half* out_[2] = {};
};
