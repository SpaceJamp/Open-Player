#include "media.h"

extern "C" {
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
}
#include <chrono>
#include <cmath>
#include <cstdio>

static constexpr size_t QUEUE_MAX = 8;    // decoded video frames buffered ahead
static constexpr double AUDIO_AHEAD = 1;  // seconds of audio buffered ahead

static AVPixelFormat pick_cuda(AVCodecContext*, const AVPixelFormat* fmts) {
    for (const AVPixelFormat* p = fmts; *p != AV_PIX_FMT_NONE; p++)
        if (*p == AV_PIX_FMT_CUDA) return *p;
    return fmts[0];  // codec NVDEC can't do: decode in software
}

static AVCodecContext* open_decoder(AVStream* st, AVBufferRef* hw) {
    const AVCodec* dec = avcodec_find_decoder(st->codecpar->codec_id);
    if (!dec) return nullptr;
    AVCodecContext* c = avcodec_alloc_context3(dec);
    avcodec_parameters_to_context(c, st->codecpar);
    c->pkt_timebase = st->time_base;
    c->thread_count = 0;  // auto, for software decoding
    if (hw) {
        c->hw_device_ctx = av_buffer_ref(hw);
        c->get_format = pick_cuda;
        c->extra_hw_frames = QUEUE_MAX + 4;  // we hold up to QUEUE_MAX decoder surfaces in the queue
    }
    if (avcodec_open2(c, dec, nullptr) < 0) avcodec_free_context(&c);
    return c;
}

bool Media::open(const char* path, int cuda_device) {
    if (avformat_open_input(&fmt_, path, nullptr, nullptr) < 0 || avformat_find_stream_info(fmt_, nullptr) < 0) {
        fprintf(stderr, "cannot open %s\n", path);
        return false;
    }
    if (fmt_->duration != AV_NOPTS_VALUE) duration_ = fmt_->duration / (double)AV_TIME_BASE;
    if (fmt_->start_time != AV_NOPTS_VALUE) start_ = fmt_->start_time / (double)AV_TIME_BASE;

    vidx_ = av_find_best_stream(fmt_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (vidx_ < 0) return fprintf(stderr, "no video stream\n"), false;
    // share the CUDA primary context so our kernels can read NVDEC's surfaces directly
    char dev[16];
    snprintf(dev, sizeof dev, "%d", cuda_device);
    if (av_hwdevice_ctx_create(&hw_, AV_HWDEVICE_TYPE_CUDA, dev, nullptr, AV_CUDA_USE_PRIMARY_CONTEXT) < 0)
        fprintf(stderr, "NVDEC unavailable, decoding on the CPU\n");
    if (!(vctx_ = open_decoder(fmt_->streams[vidx_], hw_))) return fprintf(stderr, "cannot decode video\n"), false;

    aidx_ = av_find_best_stream(fmt_, AVMEDIA_TYPE_AUDIO, -1, vidx_, nullptr, 0);
    if (aidx_ >= 0 && (actx_ = open_decoder(fmt_->streams[aidx_], nullptr))) {
        AVChannelLayout stereo = AV_CHANNEL_LAYOUT_STEREO;
        SDL_AudioSpec spec{SDL_AUDIO_F32, 2, actx_->sample_rate};
        if (swr_alloc_set_opts2(&swr_, &stereo, AV_SAMPLE_FMT_FLT, actx_->sample_rate, &actx_->ch_layout,
                                actx_->sample_fmt, actx_->sample_rate, 0, nullptr) == 0 && swr_init(swr_) == 0 &&
            (audio_ = SDL_OpenAudioDeviceStream(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &spec, nullptr, nullptr)))
            SDL_ResumeAudioStreamDevice(audio_);
        else
            fprintf(stderr, "no audio output: %s\n", SDL_GetError());
    }
    if (!audio_) aidx_ = -1, avcodec_free_context(&actx_);

    frame_ = av_frame_alloc();
    thread_ = std::thread(&Media::run, this);
    return true;
}

Media::~Media() {
    {
        std::lock_guard l(m_);
        quit_ = true;
    }
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
    clear_queue();
    if (audio_) SDL_DestroyAudioStream(audio_);
    av_frame_free(&frame_);
    avcodec_free_context(&vctx_);
    avcodec_free_context(&actx_);
    av_buffer_unref(&hw_);
    swr_free(&swr_);
    sws_freeContext(sws_);
    avformat_close_input(&fmt_);
    cudaFree(nv12_dev_);
}

void Media::clear_queue() {
    for (auto& q : queue_) av_frame_free(&q.f);
    queue_.clear();
}

double Media::audio_clock() const {
    double queued = SDL_GetAudioStreamQueued(audio_) / (8.0 * actx_->sample_rate);  // 2 ch x 4 bytes
    if (eof_ && queued == 0) return INFINITY;  // audio ran out: let any remaining video frames through
    return audio_end_ - queued;
}

void Media::run() {
    AVPacket* pkt = av_packet_alloc();
    auto audio_ahead = [&] { return audio_ ? SDL_GetAudioStreamQueued(audio_) / (8.0 * actx_->sample_rate) : 0; };
    while (true) {
        {
            std::unique_lock l(m_);
            bool go = cv_.wait_for(l, std::chrono::milliseconds(10), [&] {  // audio drains without notifying: poll
                return quit_ || seek_req_ ||
                       (!eof_ && queue_.size() < QUEUE_MAX && (audio_ahead() < AUDIO_AHEAD || queue_.size() < 2));
            });
            if (quit_) break;
            if (seek_req_) {
                seek_req_ = false;
                double t = seek_to_;
                int64_t ts = (int64_t)((t + start_) * AV_TIME_BASE);
                avformat_seek_file(fmt_, -1, INT64_MIN, ts, ts, 0);  // lands on the keyframe at or before t
                avcodec_flush_buffers(vctx_);
                if (actx_) avcodec_flush_buffers(actx_);
                clear_queue();
                if (audio_) SDL_ClearAudioStream(audio_);
                audio_end_ = t, skip_until_ = t, eof_ = false;  // decode from the keyframe, show from t
                serial_++;
                continue;
            }
            if (!go) continue;
        }
        if (av_read_frame(fmt_, pkt) < 0) {
            decode(vctx_, nullptr);  // drain frames still inside the decoders
            if (actx_) decode(actx_, nullptr);
            std::lock_guard l(m_);
            eof_ = true;
            continue;
        }
        if (pkt->stream_index == vidx_) decode(vctx_, pkt);
        else if (pkt->stream_index == aidx_) decode(actx_, pkt);
        av_packet_unref(pkt);
    }
    av_packet_free(&pkt);
}

void Media::decode(AVCodecContext* c, AVPacket* pkt) {
    if (avcodec_send_packet(c, pkt) < 0) return;  // corrupt packet: skip it
    AVRational tb = fmt_->streams[c == vctx_ ? vidx_ : aidx_]->time_base;
    while (avcodec_receive_frame(c, frame_) == 0) {
        bool has_pts = frame_->best_effort_timestamp != AV_NOPTS_VALUE;
        double pts = has_pts ? frame_->best_effort_timestamp * av_q2d(tb) - start_ : NAN;
        if (c == vctx_) {
            if (!has_pts || pts < skip_until_ - 0.001) {  // before the seek target
                av_frame_unref(frame_);
                continue;
            }
            AVFrame* f = av_frame_alloc();
            av_frame_move_ref(f, frame_);
            std::lock_guard l(m_);
            queue_.push_back({f, pts});
            continue;
        }
        double dur = frame_->nb_samples / (double)c->sample_rate;
        if (!has_pts) pts = audio_end_;
        if (pts + dur > skip_until_) {
            int n = swr_get_out_samples(swr_, frame_->nb_samples);
            pcm_.resize(n * 2);
            uint8_t* out = (uint8_t*)pcm_.data();
            n = swr_convert(swr_, &out, n, (const uint8_t**)frame_->extended_data, frame_->nb_samples);
            if (n > 0) SDL_PutAudioStreamData(audio_, pcm_.data(), n * 8);
            audio_end_ = pts + dur;
        }
        av_frame_unref(frame_);
    }
}

bool Media::take_due(double clock, bool force, AVFrame*& frame, double& pts) {
    std::lock_guard l(m_);
    if (queue_.empty() || (!force && queue_.front().pts > clock)) return false;
    Queued q = queue_.front();
    queue_.pop_front();
    while (!force && !queue_.empty() && queue_.front().pts <= clock) {  // we're behind: skip to the newest due frame
        av_frame_free(&q.f);
        q = queue_.front();
        queue_.pop_front();
    }
    cv_.notify_all();
    frame = q.f, pts = q.pts;
    return true;
}

bool Media::finished() {
    std::lock_guard l(m_);
    return eof_ && queue_.empty() && !seek_req_;
}

void Media::seek(double seconds) {
    {
        std::lock_guard l(m_);
        seek_to_ = std::fmax(0, duration_ > 0 ? std::fmin(seconds, duration_ - 0.5) : seconds);
        seek_req_ = true;
    }
    cv_.notify_all();
}

void Media::set_paused(bool p) {
    if (audio_) p ? SDL_PauseAudioStreamDevice(audio_) : SDL_ResumeAudioStreamDevice(audio_);
}

void Media::set_volume(float v) {
    if (audio_) SDL_SetAudioStreamGain(audio_, v);
}

YuvFrame Media::to_gpu(AVFrame* f, cudaStream_t s) {
    bool bt709 = f->colorspace == AVCOL_SPC_BT709 || (f->colorspace == AVCOL_SPC_UNSPECIFIED && f->height >= 720);
    bool full = f->color_range == AVCOL_RANGE_JPEG;
    AVFrame* sw = nullptr;
    if (f->format == AV_PIX_FMT_CUDA) {
        auto sw_fmt = ((AVHWFramesContext*)f->hw_frames_ctx->data)->sw_format;
        if (sw_fmt == AV_PIX_FMT_NV12 || sw_fmt == AV_PIX_FMT_P010 || sw_fmt == AV_PIX_FMT_P016)
            return {f->data[0], f->data[1], f->linesize[0], f->linesize[1], f->width, f->height,
                    sw_fmt != AV_PIX_FMT_NV12, bt709, full};
        sw = av_frame_alloc();  // other layouts (e.g. 4:4:4): download, then convert like a software frame
        av_hwframe_transfer_data(sw, f, 0);
        f = sw;
    }
    // shortcut: software frames are squeezed to 8-bit NV12 on the CPU; fine as a fallback, slow for 4K
    int w = f->width, h = f->height, pitch = (w + 1) & ~1;
    size_t size = (size_t)pitch * h + (size_t)pitch * ((h + 1) / 2);
    nv12_host_.resize(size);
    sws_ = sws_getCachedContext(sws_, w, h, (AVPixelFormat)f->format, w, h, AV_PIX_FMT_NV12, SWS_BILINEAR, nullptr,
                                nullptr, nullptr);
    uint8_t* dst[4] = {nv12_host_.data(), nv12_host_.data() + (size_t)pitch * h};
    int lines[4] = {pitch, pitch};
    sws_scale(sws_, f->data, f->linesize, 0, h, dst, lines);
    av_frame_free(&sw);
    if (nv12_dev_size_ < size) {
        cudaFree(nv12_dev_);
        cudaMalloc(&nv12_dev_, size);
        nv12_dev_size_ = size;
    }
    cudaMemcpyAsync(nv12_dev_, nv12_host_.data(), size, cudaMemcpyHostToDevice, s);
    return {nv12_dev_, nv12_dev_ + (size_t)pitch * h, pitch, pitch, w, h, false, bt709, full};
}
