#include "media.h"

#include <algorithm>

extern "C" {
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/pixdesc.h>
}
#include <chrono>
#include <cmath>
#include <cstdio>

static constexpr size_t QUEUE_MAX = 8;      // decoded video frames buffered ahead
static constexpr double AUDIO_AHEAD = 1;    // seconds of audio buffered ahead
static constexpr size_t BACKLOG_MAX = 240;  // compressed video packets read ahead to reach audio (seconds' worth)

static AVPixelFormat pick_cuda(AVCodecContext*, const AVPixelFormat* fmts) {
    for (const AVPixelFormat* p = fmts; *p != AV_PIX_FMT_NONE; p++)
        if (*p == AV_PIX_FMT_CUDA) return *p;
    return fmts[0];  // codec NVDEC can't do: decode in software
}

// With NVDEC available, prefer a decoder that can use it: FFmpeg's default isn't always one (for AV1 it is
// dav1d, CPU only; 4K60 10-bit AV1 then dropped a third of its frames). *_cuvid wrappers are skipped.
static const AVCodec* pick_decoder(AVCodecID id, bool hw) {
    void* it = nullptr;
    while (hw)
        if (const AVCodec* c = av_codec_iterate(&it)) {
            if (c->id != id || !av_codec_is_decoder(c) || (c->capabilities & AV_CODEC_CAP_HARDWARE)) continue;
            for (int i = 0; const AVCodecHWConfig* h = avcodec_get_hw_config(c, i); i++)
                if (h->device_type == AV_HWDEVICE_TYPE_CUDA) return c;
        } else break;
    return avcodec_find_decoder(id);
}

static AVCodecContext* open_decoder(AVStream* st, AVBufferRef* hw) {
    const AVCodec* dec = pick_decoder(st->codecpar->codec_id, hw);
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

bool Media::has_video(const char* path) {
    AVFormatContext* f = nullptr;
    bool ok = avformat_open_input(&f, path, nullptr, nullptr) >= 0 && avformat_find_stream_info(f, nullptr) >= 0 &&
              av_find_best_stream(f, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0) >= 0;
    avformat_close_input(&f);
    return ok;
}

bool Media::open(const char* path, int cuda_device, bool nvdec) {
    if (avformat_open_input(&fmt_, path, nullptr, nullptr) < 0 || avformat_find_stream_info(fmt_, nullptr) < 0) {
        fprintf(stderr, "cannot open %s\n", path);
        return false;
    }
    if (fmt_->duration != AV_NOPTS_VALUE) duration_ = fmt_->duration / (double)AV_TIME_BASE;
    if (fmt_->start_time != AV_NOPTS_VALUE) start_ = fmt_->start_time / (double)AV_TIME_BASE;

    vidx_ = av_find_best_stream(fmt_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (vidx_ < 0) return fprintf(stderr, "no video stream\n"), false;
    if (fmt_->streams[vidx_]->avg_frame_rate.den) fps_ = av_q2d(fmt_->streams[vidx_]->avg_frame_rate);
    // adopt our already-active CUDA context so our kernels can read NVDEC's surfaces directly
    // (USE_PRIMARY_CONTEXT refuses a primary context someone else already started with other flags)
    char dev[16];
    snprintf(dev, sizeof dev, "%d", cuda_device);
    if (!nvdec) fprintf(stderr, "decoding on the CPU (setting)\n");
    else if (av_hwdevice_ctx_create(&hw_, AV_HWDEVICE_TYPE_CUDA, dev, nullptr, AV_CUDA_USE_CURRENT_CONTEXT) < 0)
        fprintf(stderr, "NVDEC unavailable, decoding on the CPU\n"), hw_ = nullptr;
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

    tracks_ = list_tracks(fmt_, path);
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
    clear_backlog();
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
    int queued = SDL_GetAudioStreamQueued(audio_);
    if (queued <= 0) return NAN;  // nothing playing (audio ended, has a gap, or not started): no audio clock
    return audio_end_ - queued / (8.0 * actx_->sample_rate);  // 2 ch x 4 bytes per sample frame
}

void Media::run() {
    AVPacket* pkt = av_packet_alloc();
    auto audio_low = [&] { return audio_ && SDL_GetAudioStreamQueued(audio_) / (8.0 * actx_->sample_rate) < AUDIO_AHEAD; };
    bool read_all = false;  // every packet read; what's left is the backlog and the decoders' insides
    while (true) {
        bool room = false;
        {
            std::unique_lock l(m_);
            bool go = cv_.wait_for(l, std::chrono::milliseconds(10), [&] {  // audio drains without notifying: poll
                room = queue_.size() < QUEUE_MAX;
                // Files store audio and video in alternating chunks, often further apart than QUEUE_MAX frames.
                // So when audio runs low, keep reading past a full video queue and park the video packets.
                bool read_for_audio = !read_all && audio_low() && backlog_.size() < BACKLOG_MAX;
                return quit_ || seek_req_ || (!eof_ && (room || read_for_audio));
            });
            if (quit_) break;
            if (seek_req_) {
                seek_req_ = false;
                if (cap_change_) {
                    l.unlock();
                    apply_captions();  // opens decoders, maybe reads a .srt: not under the lock
                    l.lock();
                }
                cues_.clear(), cue_ver_++;  // embedded cues come back as the demuxer reaches them again
                for (auto& d : cap_dec_)
                    if (d) d->flush();
                double t = seek_to_;
                int64_t ts = (int64_t)((t + start_) * AV_TIME_BASE);
                avformat_seek_file(fmt_, -1, INT64_MIN, ts, ts, 0);  // lands on the keyframe at or before t
                avcodec_flush_buffers(vctx_);
                if (actx_) avcodec_flush_buffers(actx_);
                clear_queue();
                clear_backlog();
                if (audio_) SDL_ClearAudioStream(audio_);
                audio_end_ = t, skip_until_ = t, eof_ = false, read_all = false;  // decode from the keyframe, show from t
                last_vpts_ = t - 1 / (fps_ > 0 ? fps_ : 25);
                serial_++;
                continue;
            }
            if (!go) continue;
        }
        if (room && !backlog_.empty()) {  // video read ahead for the audio's sake, now there's room for it
            AVPacket* p = backlog_.front();
            backlog_.pop_front();
            decode(vctx_, p);
            av_packet_free(&p);
            continue;
        }
        if (read_all) {
            if (!room) continue;
            decode(vctx_, nullptr);  // drain frames still inside the decoders
            if (actx_) decode(actx_, nullptr);
            std::lock_guard l(m_);
            eof_ = true;
            continue;
        }
        if (av_read_frame(fmt_, pkt) < 0) {
            read_all = true;
            continue;
        }
        if (pkt->stream_index == vidx_ && (room && backlog_.empty())) decode(vctx_, pkt);
        else if (pkt->stream_index == vidx_) backlog_.push_back(av_packet_clone(pkt));
        else if (pkt->stream_index == aidx_) decode(actx_, pkt);
        else
            for (int k = 0; k < 2; k++)
                if (pkt->stream_index == cap_stream_[k]) {
                    std::vector<Cue> got;
                    cap_dec_[k]->feed(pkt, got);
                    if (got.empty()) continue;
                    std::lock_guard l(m_);
                    // a film's worth of pictures would pile up: forget cues that began over a minute before now
                    const double old = shown_pts_ - 60;
                    cues_.erase(std::remove_if(cues_.begin(), cues_.end(), [&](const Cue& c) { return c.start < old; }),
                                cues_.end());
                    for (auto& c : got) {  // playback seconds start at the file's start time
                        c.start -= start_;
                        if (c.end >= 0) c.end -= start_;
                        if (cues_.size() >= MAX_CUES) cues_.erase(cues_.begin(), cues_.begin() + MAX_CUES / 2);
                        cues_.push_back(std::move(c));
                    }
                    cue_ver_++;
                }
        av_packet_unref(pkt);
    }
    av_packet_free(&pkt);
}

void Media::clear_backlog() {
    for (AVPacket*& p : backlog_) av_packet_free(&p);
    backlog_.clear();
}

bool Media::software_fallback() {
    const AVCodec* sw = avcodec_find_decoder(vctx_->codec_id);
    if (!sw || sw == vctx_->codec) return false;  // already the default decoder: nothing to fall back to
    fprintf(stderr, "%s can't decode this stream on the GPU, using %s\n", vctx_->codec->name, sw->name);
    avcodec_free_context(&vctx_);
    vctx_ = open_decoder(fmt_->streams[vidx_], nullptr);
    // packets the GPU decoder already took are gone: re-read from the current spot like a seek, so the new
    // decoder starts on a keyframe instead of mid-GOP (the run loop also clears queued audio and frames)
    std::lock_guard l(m_);
    seek_to_ = std::fmax(skip_until_, 0), seek_req_ = true;
    return vctx_ != nullptr;
}

void Media::decode(AVCodecContext* c, AVPacket* pkt) {
    int r = avcodec_send_packet(c, pkt);
    // a GPU-only decoder (FFmpeg's AV1) fails on streams NVDEC can't do, e.g. AV1 4:4:4: switch before any frame
    if (r < 0 && c == vctx_ && !video_frames_ && software_fallback()) return;
    if (r < 0) return;  // corrupt packet: skip it
    AVRational tb = fmt_->streams[c == vctx_ ? vidx_ : aidx_]->time_base;
    while ((r = avcodec_receive_frame(c, frame_)) == 0) {
        if (c == vctx_) video_frames_++;
        bool has_pts = frame_->best_effort_timestamp != AV_NOPTS_VALUE;
        double pts = has_pts ? frame_->best_effort_timestamp * av_q2d(tb) - start_ : NAN;
        if (c == vctx_) {
            // raw .h264/.hevc and some AVIs carry no timestamps: assume one frame after the previous
            if (!has_pts) pts = last_vpts_ + 1 / (fps_ > 0 ? fps_ : 25);
            last_vpts_ = pts;
            if (pts < skip_until_ - 0.001) {  // before the seek target
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
    bool failed = r != AVERROR(EAGAIN) && r != AVERROR_EOF;
    if (failed && c == vctx_ && !video_frames_ && pkt) software_fallback();
}

bool Media::take_due(double clock, double lead, bool force, AVFrame*& frame, double& pts) {
    std::lock_guard l(m_);
    if (queue_.empty() || (!force && queue_.front().pts > clock + lead)) return false;
    Queued q = queue_.front();
    queue_.pop_front();
    // behind: skip to the newest frame due now. Only against `clock`: frames inside the lead window aren't late.
    while (!force && !queue_.empty() && queue_.front().pts <= clock) {
        av_frame_free(&q.f);
        dropped_++;
        q = queue_.front();
        queue_.pop_front();
    }
    cv_.notify_all();
    frame = q.f, pts = q.pts;
    shown_pts_ = pts;
    return true;
}

int Media::queued_frames() {
    std::lock_guard l(m_);
    return (int)queue_.size();
}

double Media::audio_buffered() const {
    return audio_ ? SDL_GetAudioStreamQueued(audio_) / (8.0 * actx_->sample_rate) : 0;
}

bool Media::finished() {
    std::lock_guard l(m_);
    return eof_ && queue_.empty() && !seek_req_;
}

void Media::set_captions(int track, int forced) {
    double at;
    {
        std::lock_guard l(m_);
        if (cap_want_[0] == track && cap_want_[1] == forced) return;
        cap_want_[0] = track, cap_want_[1] = forced, cap_change_ = true;
        if (seek_req_) return;  // a seek is on its way: it applies the change, and lands where it was asked to
        at = reseek_target(serial_ < seek_serial_, last_seek_, shown_pts_);
    }
    seek(at);  // re-read from here, so the new track's cues around now arrive
}

void Media::apply_captions() {
    int want[2];
    {
        std::lock_guard l(m_);
        want[0] = cap_want_[0], want[1] = cap_want_[1], cap_change_ = false;
    }
    std::vector<Cue> side;
    for (int k = 0; k < 2; k++) {
        cap_stream_[k] = -1;
        cap_dec_[k] = std::make_unique<CueDecoder>();
        if (want[k] < 0 || want[k] >= (int)tracks_.size()) continue;
        const CaptionTrack& t = tracks_[want[k]];
        if (t.stream >= 0) {
            if (cap_dec_[k]->open(fmt_->streams[t.stream])) cap_stream_[k] = t.stream;
        } else
            for (auto& c : read_sidecar(t.sidecar)) side.push_back(std::move(c));
    }
    std::lock_guard l(m_);
    sidecar_cues_ = std::move(side);
    cue_ver_++;
}

std::vector<Cue> Media::cues() {
    std::lock_guard l(m_);
    std::vector<Cue> all = cues_;
    all.insert(all.end(), sidecar_cues_.begin(), sidecar_cues_.end());
    return all;
}

void Media::seek(double seconds) {
    {
        std::lock_guard l(m_);
        seek_to_ = std::fmax(0, duration_ > 0 ? std::fmin(seconds, duration_ - 0.5) : seconds);
        seek_req_ = true, last_seek_ = seek_to_, seek_serial_ = serial_ + 1;
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
    if (decode_desc_.empty()) {
        bool hw = f->format == AV_PIX_FMT_CUDA;
        auto fmt = hw ? ((AVHWFramesContext*)f->hw_frames_ctx->data)->sw_format : (AVPixelFormat)f->format;
        decode_desc_ = std::string(hw ? "NVDEC " : "CPU ") + vctx_->codec->name + " " + av_get_pix_fmt_name(fmt);
    }
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
