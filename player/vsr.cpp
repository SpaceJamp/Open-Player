#include "vsr.h"

#include <NvOnnxParser.h>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <vector>

namespace fs = std::filesystem;
using namespace nvinfer1;

static constexpr float SCENE_CUT = 0.12f;  // mean |pixel change| that counts as a cut (same as upscale.py)

static struct Logger : ILogger {
    void log(Severity s, const char* msg) noexcept override {
        if (s <= Severity::kWARNING) fprintf(stderr, "[tensorrt] %s\n", msg);
    }
} logger;

static std::vector<char> read_file(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    return {std::istreambuf_iterator<char>(f), {}};
}

Vsr::~Vsr() {
    for (int i = 0; i < 2; i++) cudaFree(lr_[i]), cudaFree(hr_[i]);
    cudaFree(scratch_);
}

bool Vsr::open(const std::string& onnx, int w, int h, cudaStream_t s) {
    // export.py writes the scale into the .ref header; the ONNX itself only has symbolic sizes
    auto ref = read_file(fs::path(onnx).replace_extension(".ref"));
    if (ref.size() < 12) {
        fprintf(stderr, "vsr: missing %s (export.py writes it next to the .onnx)\n",
                fs::path(onnx).replace_extension(".ref").c_str());
        return false;
    }
    w_ = w, h_ = h, s_ = s, scale_ = ((int*)ref.data())[0];
    if (!load_or_build(onnx, true)) return false;

    size_t n = 3ull * w * h, N = n * scale_ * scale_;
    for (int i = 0; i < 2; i++)
        if (cudaMalloc(&lr_[i], n * 2) || cudaMalloc(&hr_[i], N * 2)) return false;
    return cudaMalloc(&scratch_, sizeof(float)) == cudaSuccess;
}

bool Vsr::load_or_build(const std::string& onnx, bool allow_cache) {
    int s = scale_;
    fs::path cache = onnx + "." + std::to_string(w_) + "x" + std::to_string(h_) + ".engine";
    std::error_code ec;
    bool cached = allow_cache && fs::exists(cache) && fs::last_write_time(cache, ec) >= fs::last_write_time(onnx, ec);
    std::vector<char> plan;
    if (cached) {
        plan = read_file(cache);
    } else {
        fprintf(stderr, "vsr: building TensorRT engine for %dx%d (first time only, can take a minute)\n", w_, h_);
        std::unique_ptr<IBuilder> b(createInferBuilder(logger));
        // strongly typed: the fp16 ONNX decides precision, no builder precision flags needed
#if NV_TENSORRT_MAJOR >= 11
        uint32_t flags = 0;  // strongly typed is the only mode in 11
#else
        uint32_t flags = 1U << (int)NetworkDefinitionCreationFlag::kSTRONGLY_TYPED;
#endif
        std::unique_ptr<INetworkDefinition> net(b->createNetworkV2(flags));
        std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*net, logger));
        if (!parser->parseFromFile(onnx.c_str(), (int)ILogger::Severity::kWARNING)) return false;
        std::unique_ptr<IBuilderConfig> cfg(b->createBuilderConfig());
        IOptimizationProfile* prof = b->createOptimizationProfile();
        auto fix = [&](const char* name, int H, int W) {
            for (auto k : {OptProfileSelector::kMIN, OptProfileSelector::kOPT, OptProfileSelector::kMAX})
                prof->setDimensions(name, k, Dims4{1, 3, H, W});
        };
        fix("cur", h_, w_), fix("prev", h_, w_), fix("prev_hr", h_ * s, w_ * s);
        cfg->addOptimizationProfile(prof);
        std::unique_ptr<IHostMemory> m(b->buildSerializedNetwork(*net, *cfg));
        if (!m) return false;
        plan.assign((char*)m->data(), (char*)m->data() + m->size());
        std::ofstream(cache, std::ios::binary).write(plan.data(), plan.size());
    }

    runtime_.reset(createInferRuntime(logger));
    engine_.reset(runtime_->deserializeCudaEngine(plan.data(), plan.size()));
    if (!engine_) {
        if (!cached) return false;
        fprintf(stderr, "vsr: cached engine is stale (TensorRT or driver changed), rebuilding\n");
        return load_or_build(onnx, false);
    }
    ctx_.reset(engine_->createExecutionContext());
    return ctx_ && ctx_->setInputShape("cur", Dims4{1, 3, h_, w_}) && ctx_->setInputShape("prev", Dims4{1, 3, h_, w_}) &&
           ctx_->setInputShape("prev_hr", Dims4{1, 3, h_ * s, w_ * s});
}

void Vsr::infer(const __half* cur, const __half* prev, const __half* prev_hr, __half* out) {
    ctx_->setTensorAddress("cur", (void*)cur);
    ctx_->setTensorAddress("prev", (void*)prev);
    ctx_->setTensorAddress("prev_hr", (void*)prev_hr);
    ctx_->setTensorAddress("hr", out);
    ctx_->enqueueV3(s_);
}

const __half* Vsr::run(bool reset) {
    __half *cur = lr_[i_], *prev = lr_[i_ ^ 1], *prev_hr = hr_[i_ ^ 1], *out = hr_[i_];
    size_t n = 3ull * w_ * h_;
    if (!reset && mean_abs_diff(cur, prev, n, scratch_, s_) > SCENE_CUT) reset = true;
    if (reset) {  // no usable history: start from this frame alone
        cudaMemcpyAsync(prev, cur, n * 2, cudaMemcpyDeviceToDevice, s_);
        upscale_bilinear(cur, w_, h_, scale_, prev_hr, s_);
    }
    infer(cur, prev, prev_hr, out);
    i_ ^= 1;
    return out;
}

int Vsr::selftest(const std::string& onnx) {
    auto ref = read_file(fs::path(onnx).replace_extension(".ref"));
    if (ref.size() < 12) return fprintf(stderr, "selftest: no .ref next to %s\n", onnx.c_str()), 1;
    const int* hdr = (const int*)ref.data();
    int s = hdr[0], h = hdr[1], w = hdr[2];
    size_t n = 3ull * w * h, N = n * s * s;
    if (ref.size() != 12 + 4 * (2 * n + 2 * N)) return fprintf(stderr, "selftest: .ref has the wrong size\n"), 1;
    const float* cur = (const float*)(ref.data() + 12);
    const float *prev = cur + n, *prev_hr = prev + n, *want = prev_hr + N;

    cudaStream_t st;
    cudaStreamCreate(&st);
    Vsr v;
    if (!v.open(onnx, w, h, st)) return fprintf(stderr, "selftest: could not open engine\n"), 1;
    auto upload = [](__half* dst, const float* src, size_t count) {
        std::vector<__half> t(count);
        for (size_t i = 0; i < count; i++) t[i] = __float2half(src[i]);
        cudaMemcpy(dst, t.data(), count * 2, cudaMemcpyHostToDevice);
    };
    upload(v.lr_[0], cur, n), upload(v.lr_[1], prev, n), upload(v.hr_[1], prev_hr, N);
    v.infer(v.lr_[0], v.lr_[1], v.hr_[1], v.hr_[0]);  // not run(): its scene-cut check would reset on random input
    std::vector<__half> got(N);
    cudaMemcpyAsync(got.data(), v.hr_[0], N * 2, cudaMemcpyDeviceToHost, st);
    if (cudaStreamSynchronize(st) != cudaSuccess) return fprintf(stderr, "selftest: CUDA error\n"), 1;

    double se = 0, worst = 0;
    for (size_t i = 0; i < N; i++) {
        double d = __half2float(got[i]) - want[i];
        se += d * d, worst = std::fmax(worst, std::fabs(d));
    }
    double psnr = -10 * std::log10(se / N + 1e-12);
    bool ok = psnr > 45;  // fp16 vs fp32 reference; a wiring bug lands far below this
    printf("selftest %s: %.1f dB vs PyTorch, max error %.4f\n", ok ? "PASS" : "FAIL", psnr, worst);
    return ok ? 0 : 1;
}
