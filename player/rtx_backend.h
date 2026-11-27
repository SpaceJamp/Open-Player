#pragma once
// TensorRT for RTX, seen from the rest of the player. Its SDK's headers share names and the nvinfer1 namespace with
// the full TensorRT, so only trt_rtx.cpp includes them, once per supported version (HAVE_TRT_RTX_1_<minor>), each
// copy under its own renamed namespace; everything here is plain C++ the player can include anywhere.
#include <memory>
#include <string>

// Pixel Flow's network on TensorRT for RTX, built (or loaded from its plan) for one input size.
struct RtxEngine {
    virtual ~RtxEngine() = default;
    // planar fp16 RGB on the GPU: cur/prev 3 x h x w, prev_hr and out 3 x h*scale x w*scale; stream: a cudaStream_t
    virtual bool infer(const void* cur, const void* prev, const void* prev_hr, void* out, void* stream) = 0;
    virtual void save_cache() = 0;  // keeps the kernels compiled for this GPU for the next start (<plan>.jit)
};

#ifdef HAVE_TRT_RTX_1_8
int trt_rtx_header_version_rtx8();  // NV_TENSORRT_VERSION of the headers that copy was built with
std::unique_ptr<RtxEngine> make_rtx_engine_rtx8(const std::string& onnx, const std::string& plan_path, int w, int h,
                                                int scale);
#endif
#ifdef HAVE_TRT_RTX_1_7
int trt_rtx_header_version_rtx7();
std::unique_ptr<RtxEngine> make_rtx_engine_rtx7(const std::string& onnx, const std::string& plan_path, int w, int h,
                                                int scale);
#endif

// The header version a build's copy for this minor version was compiled with; 0: not compiled in.
inline int trt_rtx_header_version(int minor) {
#ifdef HAVE_TRT_RTX_1_8
    if (minor == 8) return trt_rtx_header_version_rtx8();
#endif
#ifdef HAVE_TRT_RTX_1_7
    if (minor == 7) return trt_rtx_header_version_rtx7();
#endif
    (void)minor;
    return 0;
}

// Builds or loads (plan_path) Pixel Flow on the TensorRT for RTX 1.<minor> that trt_loader loaded; nullptr: failed.
inline std::unique_ptr<RtxEngine> make_rtx_engine(int minor, const std::string& onnx, const std::string& plan_path,
                                                  int w, int h, int scale) {
#ifdef HAVE_TRT_RTX_1_8
    if (minor == 8) return make_rtx_engine_rtx8(onnx, plan_path, w, h, scale);
#endif
#ifdef HAVE_TRT_RTX_1_7
    if (minor == 7) return make_rtx_engine_rtx7(onnx, plan_path, w, h, scale);
#endif
    (void)minor, (void)onnx, (void)plan_path, (void)w, (void)h, (void)scale;
    return nullptr;
}
