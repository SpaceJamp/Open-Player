#pragma once
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswresample/swresample.h>
#include <libswscale/swscale.h>
}
#include <SDL3/SDL.h>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

#include "kernels.h"

// Demuxes and decodes on a background thread. Video frames queue for the main thread (in GPU memory when
// NVDEC handles the codec); audio goes straight into an SDL audio stream, which also serves as the clock.
// All times are seconds from the start of the file.
class Media {
public:
    ~Media();
    bool open(const char* path, int cuda_device);
    int width() const { return vctx_->width; }
    int height() const { return vctx_->height; }
    double duration() const { return duration_; }
    bool has_audio() const { return audio_ != nullptr; }

    // Takes the newest frame that is due at `clock`, freeing older due ones (they are late).
    // force: take the next frame even if it is not due (showing a frame after seeking while paused).
    bool take_due(double clock, bool force, AVFrame*& frame, double& pts);
    bool finished();             // end of file reached and every frame taken
    double audio_clock() const;  // position of the audio being heard now
    void seek(double seconds);   // asynchronous; serial() increments once it has happened
    int serial() const { return serial_; }
    void set_paused(bool p);
    void set_volume(float v);
    // Points the CUDA kernels at a decoded frame, uploading it first if it was decoded in software.
    YuvFrame to_gpu(AVFrame* f, cudaStream_t s);

private:
    struct Queued { AVFrame* f; double pts; };
    void run();
    void decode(AVCodecContext* c, AVPacket* pkt);
    void clear_queue();

    AVFormatContext* fmt_ = nullptr;
    AVCodecContext *vctx_ = nullptr, *actx_ = nullptr;
    AVBufferRef* hw_ = nullptr;
    SwrContext* swr_ = nullptr;
    SwsContext* sws_ = nullptr;
    SDL_AudioStream* audio_ = nullptr;
    int vidx_ = -1, aidx_ = -1;
    double duration_ = 0, start_ = 0;

    std::thread thread_;
    std::mutex m_;
    std::condition_variable cv_;
    std::deque<Queued> queue_;
    bool quit_ = false, seek_req_ = false;
    std::atomic<bool> eof_{false};  // also read by audio_clock() on the main thread
    double seek_to_ = 0, skip_until_ = -1;
    std::atomic<int> serial_{0};
    std::atomic<double> audio_end_{0};  // timestamp just past the last audio sample handed to SDL

    AVFrame* frame_ = nullptr;
    std::vector<float> pcm_;
    std::vector<uint8_t> nv12_host_;
    uint8_t* nv12_dev_ = nullptr;
    size_t nv12_dev_size_ = 0;
};
