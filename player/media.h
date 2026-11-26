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
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "captions.h"
#include "kernels.h"

// Demuxes and decodes on a background thread. Video frames queue for the main thread (in GPU memory when
// NVDEC handles the codec); audio goes straight into an SDL audio stream, which also serves as the clock.
// All times are seconds from the start of the file.
class Media {
public:
    ~Media();
    bool open(const char* path, int cuda_device, bool nvdec = true);  // nvdec false: decode on the CPU
    bool nvdec() const { return hw_ != nullptr; }  // NVDEC was opened (codecs it lacks still decode on the CPU)
    static bool has_video(const char* path);  // FFmpeg can open it and it has a video stream
    int width() const { return vctx_->width; }
    int height() const { return vctx_->height; }
    double duration() const { return duration_; }
    double fps() const { return fps_; }  // 0 if the file doesn't say
    bool has_audio() const { return audio_ != nullptr; }

    // diagnostics
    int queued_frames();
    double audio_buffered() const;                       // seconds of audio waiting to play
    int take_dropped() { return dropped_.exchange(0); }  // late frames skipped since the last call
    std::string decode_desc() const { return decode_desc_; }  // e.g. "NVDEC h264 nv12"; set by to_gpu()

    // Takes the newest frame that is due at `clock`, freeing older due ones (they are late).
    // force: take the next frame even if it is not due (showing a frame after seeking while paused).
    // lead: also take a frame due within the next `lead` seconds (render-ahead); skipping uses `clock` alone.
    bool take_due(double clock, double lead, bool force, AVFrame*& frame, double& pts);
    bool finished();             // end of file reached and every frame taken
    double audio_clock() const;  // position of the audio being heard now; NaN while no audio is queued
    void seek(double seconds);   // asynchronous; serial() increments once it has happened
    int serial() const { return serial_; }
    void set_paused(bool p);
    void set_volume(float v);
    // Captions: the file's English tracks; which to show (indexes into them, -1: none; forced: shown alongside, for
    // a forced track with captions off). An embedded track's cues are read as the file plays, so a change re-reads
    // from the playback position (never undoing a seek still under way). cues(): what's known now, in playback
    // seconds; cheap to copy (pictures are shared).
    const std::vector<CaptionTrack>& caption_tracks() const { return tracks_; }
    void set_captions(int track, int forced);
    std::vector<Cue> cues();
    int cues_version() const { return cue_ver_; }  // changes whenever cues() would
    // Points the CUDA kernels at a decoded frame, uploading it first if it was decoded in software.
    YuvFrame to_gpu(AVFrame* f, cudaStream_t s);

private:
    struct Queued { AVFrame* f; double pts; };
    void run();
    void decode(AVCodecContext* c, AVPacket* pkt);
    void clear_queue();
    void clear_backlog();
    bool software_fallback();  // swap a GPU-only video decoder for FFmpeg's default; decode thread only
    void apply_captions();     // decode thread: open the chosen tracks' decoders, read a chosen .srt

    AVFormatContext* fmt_ = nullptr;
    AVCodecContext *vctx_ = nullptr, *actx_ = nullptr;
    AVBufferRef* hw_ = nullptr;
    SwrContext* swr_ = nullptr;
    SwsContext* sws_ = nullptr;
    SDL_AudioStream* audio_ = nullptr;
    int vidx_ = -1, aidx_ = -1;
    double duration_ = 0, start_ = 0, fps_ = 0;
    std::atomic<int> dropped_{0};
    long video_frames_ = 0;  // decoded so far; the software fallback is only allowed before the first one
    std::string decode_desc_;

    std::thread thread_;
    std::mutex m_;
    std::condition_variable cv_;
    std::deque<Queued> queue_;
    std::deque<AVPacket*> backlog_;  // compressed video packets read ahead to reach audio; decode thread only
    bool quit_ = false, seek_req_ = false;
    std::atomic<bool> eof_{false};  // also read by finished() callers on the main thread
    double seek_to_ = 0, skip_until_ = -1, last_vpts_ = -0.04;  // last_vpts_: for frames without timestamps
    std::atomic<int> serial_{0};
    std::atomic<double> audio_end_{0};  // timestamp just past the last audio sample handed to SDL

    std::vector<CaptionTrack> tracks_;
    int cap_want_[2] = {-1, -1};     // under m_: the chosen and the forced track
    bool cap_change_ = false;        // under m_
    int cap_stream_[2] = {-1, -1};   // decode thread: their stream indexes (-1: none or a .srt)
    std::unique_ptr<CueDecoder> cap_dec_[2];  // decode thread
    std::vector<Cue> cues_, sidecar_cues_;  // under m_
    std::atomic<int> cue_ver_{0};
    // where a caption change re-reads from: the last frame handed out, or a seek not shown yet
    std::atomic<double> shown_pts_{0};
    double last_seek_ = 0;      // under m_
    int seek_serial_ = 0;       // under m_: the serial the latest seek() will have once done

    AVFrame* frame_ = nullptr;
    std::vector<float> pcm_;
    std::vector<uint8_t> nv12_host_;
    uint8_t* nv12_dev_ = nullptr;
    size_t nv12_dev_size_ = 0;
};
