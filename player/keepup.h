#pragma once
// Whether Pixel Flow keeps up with the video, from what --stats already measures. Over a window of at least 10 s
// of playback (not counting the first 5 s after a start, seek or engine build): more than 5% of frames dropped, or
// Pixel Flow's own time per frame above the video's frame interval, means it doesn't, and the player falls back to
// NVIDIA RTX VSR for this video.
struct KeepUp {
    double since_start = 0;  // seconds of playback since the start, the last seek or the engine build
    double window_s = 0;     // seconds this window covers
    int frames = 0, dropped = 0;
    double pxf_ms = 0;       // Pixel Flow's mean time per frame in the window
    double frame_ms = 0;     // the video's frame interval (0: unknown)
};

inline bool should_fall_back(const KeepUp& k) {
    const int all = k.frames + k.dropped;
    if (k.since_start < 5 || k.window_s < 10 || all <= 0) return false;
    return k.dropped > 0.05 * all || (k.frame_ms > 0 && k.pxf_ms > k.frame_ms);
}
