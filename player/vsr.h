#pragma once
#include <NvInfer.h>
#include <memory>
#include <string>

#include "kernels.h"
#include "rtx_backend.h"
#include "trt_loader.h"

// Runs the exported VSR network with TensorRT and keeps its recurrent state (previous frame + previous output).
class Vsr {
public:
    ~Vsr();
    // Loads the engine for a w x h input, building and caching it as <onnx>.<w>x<h>.engine the first time.
    // engine: the full TensorRT or TensorRT for RTX 1.<rtx_minor> (whichever trt_loader loaded).
    bool open(const std::string& onnx, int w, int h, cudaStream_t s, Engine engine = Engine::Full, int rtx_minor = -1);
    int scale() const { return scale_; }
    static int model_scale(const std::string& onnx);  // the model's upscale factor, 0 if it isn't exported
    __half* input() { return lr_[i_]; }  // write the next frame here: planar fp16 RGB, 3 x h x w
    // Upscales input(); returns 3 x (h*scale) x (w*scale), valid until the next run(). reset = new shot / seek.
    const __half* run(bool reset);
    // Compares the engine against the .ref file export.py wrote. Returns 0 on pass.
    static int selftest(const std::string& onnx, Engine engine = Engine::Full, int rtx_minor = -1);
    void save_cache();  // TensorRT for RTX: keeps its kernels for this GPU for the next start

private:
    bool load_or_build(const std::string& onnx, bool allow_cache);
    Engine engine_kind_ = Engine::Full;
    int rtx_minor_ = -1;
    std::unique_ptr<RtxEngine> rtx_;
    void infer(const __half* cur, const __half* prev, const __half* prev_hr, __half* out);

    int w_ = 0, h_ = 0, scale_ = 0, i_ = 0;
    cudaStream_t s_ = nullptr;
    __half *lr_[2] = {}, *hr_[2] = {};
    float* scratch_ = nullptr;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> ctx_;
};
