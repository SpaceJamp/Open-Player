#include "rtx_vsr.h"
#include "theme.h"  // exe_path

#include <cuda.h>
#include <nvsdk_ngx.h>
#include <nvsdk_ngx_helpers_vsr.h>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <vector>

// NGX was built against an older CUDA and calls cudaGetDeviceProperties with its older, smaller cudaDeviceProp:
// CUDA 13's real one writes past NGX's stack buffer and Windows ends the process. NGX only reads the leading fields
// (name, uuid, luid), laid out the same in every CUDA version, so this fills just those. The player itself never calls
// cudaGetDeviceProperties (it asks the driver API instead), so this replacement only ever serves NGX. Linked with
// /FORCE:MULTIPLE so it wins over the CUDA runtime's own.
extern "C" cudaError_t CUDARTAPI cudaGetDeviceProperties(cudaDeviceProp* p, int device) {
    CUdevice d;
    if (cuDeviceGet(&d, device) != CUDA_SUCCESS) return cudaErrorInvalidDevice;
    cuDeviceGetName(p->name, sizeof p->name, d);
    cuDeviceGetUuid((CUuuid*)&p->uuid, d);
    cuDeviceGetLuid(p->luid, &p->luidDeviceNodeMask, d);
    return cudaSuccess;
}

bool RtxVsr::open(int w, int h, int out_w, int out_h, cudaStream_t s) {
    w_ = w, h_ = h, ow_ = out_w, oh_ = out_h, s_ = s;
    cudaFree(nullptr);  // make sure the runtime's context exists, NGX wants it
    CUcontext ctx = nullptr;
    cuCtxGetCurrent(&ctx);
    // NGX keeps its logs in this folder
    if (NVSDK_NGX_FAILED(NVSDK_NGX_CUDA_Init(0, std::filesystem::path(exe_path()).parent_path().wstring().c_str())))
        return false;
    inited_ = true;
    int available = 0;
    if (NVSDK_NGX_FAILED(NVSDK_NGX_CUDA_GetCapabilityParameters(&params_)) ||
        NVSDK_NGX_FAILED(params_->Get(NVSDK_NGX_Parameter_VSR_Available, &available)) || !available)
        return false;
    NVSDK_NGX_CUDA_VSR_Create_Params cp{};
    cp.InCUContext = ctx, cp.InCUStream = s;
    if (NVSDK_NGX_FAILED(NGX_CUDA_CREATE_VSR(&feature_, params_, &cp))) return false;

    // NGX reads a texture and writes a surface: RGBA8 arrays, filled from and copied back to linear buffers
    cudaChannelFormatDesc rgba = cudaCreateChannelDesc<uchar4>();
    if (cudaMallocArray(&src_, &rgba, w, h) || cudaMallocArray(&dst_, &rgba, out_w, out_h, cudaArraySurfaceLoadStore) ||
        cudaMalloc(&rgba_in_, 4ull * w * h) || cudaMalloc(&rgba_out_, 4ull * out_w * out_h) ||
        cudaMalloc(&out_[0], 6ull * out_w * out_h) || cudaMalloc(&out_[1], 6ull * out_w * out_h))
        return false;
    cudaResourceDesc rd{};
    rd.resType = cudaResourceTypeArray, rd.res.array.array = src_;
    cudaTextureDesc td{};  // as NVIDIA's sample: clamped, filtered, normalised coordinates
    td.addressMode[0] = td.addressMode[1] = cudaAddressModeClamp;
    td.filterMode = cudaFilterModeLinear, td.readMode = cudaReadModeNormalizedFloat, td.normalizedCoords = 1;
    if (cudaCreateTextureObject(&tex_, &rd, &td, nullptr)) return false;
    rd.res.array.array = dst_;
    return cudaCreateSurfaceObject(&surf_, &rd) == cudaSuccess;
}

const __half* RtxVsr::run(const __half* in) {
    rgb_to_rgba8(in, w_, h_, rgba_in_, s_);
    cudaMemcpy2DToArrayAsync(src_, 0, 0, rgba_in_, 4ull * w_, 4ull * w_, h_, cudaMemcpyDeviceToDevice, s_);
    NVSDK_NGX_CUDA_VSR_Eval_Params ep{};
    ep.pInput = &tex_, ep.pOutput = &surf_;
    ep.InputSubrectSize = {(unsigned)w_, (unsigned)h_}, ep.OutputSubrectSize = {(unsigned)ow_, (unsigned)oh_};
    ep.QualityLevel = NVSDK_NGX_VSR_Quality_High;
    NGX_CUDA_EVALUATE_VSR(feature_, params_, &ep);  // on the stream given at open()
    cudaMemcpy2DFromArrayAsync(rgba_out_, 4ull * ow_, dst_, 0, 0, 4ull * ow_, oh_, cudaMemcpyDeviceToDevice, s_);
    i_ ^= 1;  // two outputs: the shown frame stays intact while the next one is prepared
    rgba8_to_rgb(rgba_out_, ow_, oh_, out_[i_], s_);
    return out_[i_];
}

RtxVsr::~RtxVsr() {
    if (s_) cudaStreamSynchronize(s_);
    if (surf_) cudaDestroySurfaceObject(surf_);
    if (tex_) cudaDestroyTextureObject(tex_);
    cudaFreeArray(src_), cudaFreeArray(dst_);
    cudaFree(rgba_in_), cudaFree(rgba_out_), cudaFree(out_[0]), cudaFree(out_[1]);
    if (feature_) NVSDK_NGX_CUDA_ReleaseFeature(feature_);
    if (params_) NVSDK_NGX_CUDA_DestroyParameters(params_);
    if (inited_) NVSDK_NGX_CUDA_Shutdown();
}

int RtxVsr::selftest() {
    const int w = 64, h = 64, W = 128, H = 128;
    std::vector<__half> img(3 * w * h);
    for (int c = 0; c < 3; c++)  // smooth gradients: any sane 2x upscale keeps their mean
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++) img[(c * h + y) * w + x] = __float2half(0.2f + 0.6f * (x + y * (c + 1) % h) / (2 * w));
    cudaStream_t s;
    cudaStreamCreate(&s);
    int rc = 1;
    {
        RtxVsr r;
        __half* in = nullptr;
        if (r.open(w, h, W, H, s) && cudaMalloc(&in, img.size() * 2) == cudaSuccess) {
            cudaMemcpy(in, img.data(), img.size() * 2, cudaMemcpyHostToDevice);
            std::vector<__half> out(3 * W * H);
            cudaMemcpyAsync(out.data(), r.run(in), out.size() * 2, cudaMemcpyDeviceToHost, s);
            cudaStreamSynchronize(s);
            double a = 0, b = 0;
            for (auto v : img) a += __half2float(v);
            for (auto v : out) b += __half2float(v);
            a /= img.size(), b /= out.size();
            printf("rtx vsr selftest: mean in %.3f, out %.3f\n", a, b);
            rc = std::fabs(a - b) < 0.02 ? 0 : 1;
        }
        cudaFree(in);
    }
    cudaStreamDestroy(s);
    return rc;
}
