#pragma once
// Playback diagnostics: per-stage timings and counters, summed over one second, then reported as one JSON line
// on stdout (--stats) with a verdict naming the stage that makes playback miss frames.
#include <algorithm>
#include <cstdio>
#include <string>

struct Diag {
    struct Avg {
        double sum = 0, max = 0;
        int n = 0;
        void add(double v) { sum += v, max = std::max(max, v), n++; }
        double mean() const { return n ? sum / n : 0; }
    };
    // per new frame: upload (CPU), convert + vsr (GPU); per redraw: rgba (GPU), copy (GPU -> CPU),
    // map (CPU -> Vulkan memory; only without zero-copy), present (incl. vsync)
    Avg upload, convert, vsr, rgba, copy, map, present, late, queue;
    int shown = 0, redraws = 0;

    // Which stage ate the most of this second, when frames were dropped. decode: the decoder can't keep the
    // queue filled. present: mostly waiting for vsync on redraws, i.e. the loop redraws more than it needs to.
    std::string verdict(int dropped) const {
        if (!dropped) return "ok";
        if (queue.n && queue.mean() < 1) return "decode (frame queue runs empty)";
        struct { const char* name; double total; } stage[] = {
            {"upload (software decode path)", upload.sum},
            {"convert", convert.sum},
            {"vsr (network too slow for this resolution)", vsr.sum},
            {"rgba", rgba.sum},
            {"display copy GPU->CPU", copy.sum},
            {"display copy CPU->Vulkan", map.sum},
            {"present/vsync (redraws)", present.sum},
        };
        auto worst = *std::max_element(std::begin(stage), std::end(stage), [](auto& a, auto& b) { return a.total < b.total; });
        char buf[96];
        snprintf(buf, sizeof buf, "%s, %.0f ms of this second", worst.name, worst.total);
        return buf;
    }

    // One report line. Values are per-event means in ms unless the name says otherwise.
    void report(double t, double src_fps, int dropped, double audio_s, double gpu_mem_mb, bool vsr_on, bool paused) {
        printf("stats {\"t\":%.1f,\"fps\":%d,\"src_fps\":%.2f,\"dropped\":%d,\"redraws\":%d,\"upload_ms\":%.2f,"
               "\"convert_ms\":%.2f,\"vsr_ms\":%.2f,\"vsr_max_ms\":%.2f,\"rgba_ms\":%.2f,\"copy_ms\":%.2f,"
               "\"map_ms\":%.2f,\"present_ms\":%.2f,\"late_ms\":%.1f,\"queue\":%.1f,\"audio_s\":%.2f,\"gpu_mem_mb\":%.0f,"
               "\"vsr_on\":%s,\"paused\":%s,\"verdict\":\"%s\"}\n",
               t, shown, src_fps, dropped, redraws, upload.mean(), convert.mean(), vsr.mean(), vsr.max, rgba.mean(),
               copy.mean(), map.mean(), present.mean(), late.mean(), queue.mean(), audio_s, gpu_mem_mb, vsr_on ? "true" : "false",
               paused ? "true" : "false", verdict(dropped).c_str());
        fflush(stdout);
        *this = Diag();
    }
};
