#pragma once
#include <NvInfer.h>
#include <memory>
#include <string>

#include "kernels.h"

// Runs the exported VSR network with TensorRT and keeps its recurrent state (previous frame + previous output).
class Vsr {
public:
    ~Vsr();
    // Loads the engine for a w x h input, building and caching it as <onnx>.<w>x<h>.engine the first time.
    bool open(const std::string& onnx, int w, int h, cudaStream_t s);
    int scale() const { return scale_; }
    __half* input() { return lr_[i_]; }  // write the next frame here: planar fp16 RGB, 3 x h x w
    // Upscales input(); returns 3 x (h*scale) x (w*scale), valid until the next run(). reset = new shot / seek.
    const __half* run(bool reset);
    // Compares the engine against the .ref file export.py wrote. Returns 0 on pass.
    static int selftest(const std::string& onnx);

private:
    bool load_or_build(const std::string& onnx, bool allow_cache);
    void infer(const __half* cur, const __half* prev, const __half* prev_hr, __half* out);

    int w_ = 0, h_ = 0, scale_ = 0, i_ = 0;
    cudaStream_t s_ = nullptr;
    __half *lr_[2] = {}, *hr_[2] = {};
    float* scratch_ = nullptr;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> ctx_;
};
