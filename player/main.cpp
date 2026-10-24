// Open Player: video player that upscales with our VSR network on the GPU.
// decode (NVDEC) -> YUV->RGB (CUDA) -> VSR (TensorRT) -> RGBA (CUDA) -> Vulkan blit to the window
#include <SDL3/SDL.h>
#include <dlfcn.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>

#include "diag.h"
#include "display.h"
#include "kernels.h"
#include "media.h"
#include "vsr.h"

static const char* USAGE =
    "usage: open-player <video> [--model vsr.onnx] [--no-vsr] [--stats] [--screen N]\n"
    "--screen N: play on monitor N (1 = first; --check lists them); upscaling is planned for that monitor\n"
    "       open-player --selftest [vsr.onnx]   compare the TensorRT engine against PyTorch\n"
    "       open-player --check [vsr.onnx]      test each part of the setup on its own\n"
    "--stats: print a JSON line of timings every second (the control panel's Diagnostics reads these)\n"
    "keys:  space pause | left/right seek 5s | up/down volume | v VSR on/off | f or double-click fullscreen\n"
    "       click the bar at the bottom to seek | q quit\n";

static std::string clock_text(double t) {
    int s = (int)std::fmax(t, 0);
    char buf[32];
    if (s >= 3600) snprintf(buf, sizeof buf, "%d:%02d:%02d", s / 3600, s / 60 % 60, s % 60);
    else snprintf(buf, sizeof buf, "%02d:%02d", s / 60, s % 60);
    return buf;
}

static std::string trt_version() {
    int v = getInferLibVersion();  // major*10000 + minor*100 + patch
    return std::to_string(v / 10000) + "." + std::to_string(v / 100 % 100) + "." + std::to_string(v % 100);
}

// --check: tries each part of the pipeline on its own and says which one is broken.
static int check(const char* model) {
    int fails = 0;
    auto line = [&](const char* verdict, const char* what, const std::string& detail) {
        printf("%-4s  %-16s %s\n", verdict, what, detail.c_str());
        fails += !strcmp(verdict, "FAIL");
    };
    int n = 0, drv = 0;
    cudaGetDeviceCount(&n);
    cudaDriverGetVersion(&drv);
    cudaDeviceProp p{};
    if (n > 0 && cudaGetDeviceProperties(&p, 0) == cudaSuccess)
        line("PASS", "CUDA", std::string(p.name) + ", sm_" + std::to_string(p.major) + std::to_string(p.minor) +
                                 ", driver CUDA " + std::to_string(drv / 1000) + "." + std::to_string(drv % 1000 / 10));
    else line("FAIL", "CUDA", "no CUDA GPU visible (driver missing, or on WSL: Windows driver too old)");

    void* nvcuvid = dlopen("libnvcuvid.so.1", RTLD_NOW);  // what FFmpeg's NVDEC loads
    line(nvcuvid ? "PASS" : "WARN", "NVDEC", nvcuvid ? "libnvcuvid found" : "libnvcuvid missing: videos decode on the CPU");
    if (nvcuvid) dlclose(nvcuvid);

    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO)) {
        if (SDL_Window* win = SDL_CreateWindow("Open Player check", 64, 64, SDL_WINDOW_VULKAN | SDL_WINDOW_HIDDEN)) {
            Display d;
            if (d.init(win)) {
                cudaSetDevice(d.cuda_device());
                bool ok = d.alloc(64, 64);
                line(ok ? "PASS" : "FAIL", "Vulkan", std::string(d.gpu_name()) + (!ok ? ", cannot allocate frames"
                                                     : d.zero_copy() ? ", zero-copy from CUDA"
                                                                     : ", frames copied through the CPU (slower)"));
            } else line("FAIL", "Vulkan", "no Vulkan GPU for the window (on WSL install vulkan-dzn)");
        } else line("FAIL", "Window", SDL_GetError());
        int nd = 0;
        SDL_DisplayID* ids = SDL_GetDisplays(&nd);
        for (int i = 0; i < nd; i++)
            if (const SDL_DisplayMode* m = SDL_GetDesktopDisplayMode(ids[i])) {
                const char* name = SDL_GetDisplayName(ids[i]);
                char d[160];
                snprintf(d, sizeof d, "%s, %ldx%ld @ %.0f Hz%s (--screen %d)", name ? name : "?",
                         std::lround(m->w * m->pixel_density), std::lround(m->h * m->pixel_density), m->refresh_rate,
                         ids[i] == SDL_GetPrimaryDisplay() ? ", primary" : "", i + 1);
                line("INFO", ("Screen " + std::to_string(i + 1)).c_str(), d);
            }
        SDL_free(ids);
        SDL_AudioDeviceID dev = SDL_OpenAudioDevice(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, nullptr);
        line(dev ? "PASS" : "WARN", "Audio", dev ? SDL_GetAudioDeviceName(dev) : "no audio output: video plays silent");
        if (dev) SDL_CloseAudioDevice(dev);
        SDL_Quit();
    } else line("FAIL", "SDL", SDL_GetError());

    line("PASS", "TensorRT", trt_version());
    if (!std::filesystem::exists(model)) line("FAIL", "Model", std::string(model) + " not found: train, then export");
    else line(Vsr::selftest(model) == 0 ? "PASS" : "FAIL", "Model", model);
    printf(fails ? "%d problem(s) found\n" : "all good\n", fails);
    return fails ? 1 : 0;
}

int main(int argc, char** argv) {
    const char *path = nullptr, *model = "vsr.onnx";
    bool vsr_on = true, selftest = false, checking = false, stats = false;
    int screen = 0;  // --screen N, counted from 1; 0 = the primary display
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "--screen") && i + 1 < argc) screen = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--no-vsr")) vsr_on = false;
        else if (!strcmp(argv[i], "--stats")) stats = true;
        else if (!strcmp(argv[i], "--selftest")) selftest = true;
        else if (!strcmp(argv[i], "--check")) checking = true;
        else if (argv[i][0] != '-') (selftest || checking ? model : path) = argv[i];
        else return fputs(USAGE, stderr), 2;
    }
    if (selftest) return Vsr::selftest(model);
    if (checking) return check(model);
    if (!path) return fputs(USAGE, stderr), 2;

    if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO)) return fprintf(stderr, "SDL: %s\n", SDL_GetError()), 1;
    int n_disp = 0;
    SDL_DisplayID* ids = SDL_GetDisplays(&n_disp);
    SDL_DisplayID screen_id = !screen ? SDL_GetPrimaryDisplay() : screen > 0 && screen <= n_disp ? ids[screen - 1] : 0;
    SDL_free(ids);
    if (!screen_id) return fprintf(stderr, "no screen %d: this machine has %d (see --check)\n", screen, n_disp), 2;
    // hidden until we know the video size; Vulkan needs the window first to choose the GPU
    SDL_Window* win = SDL_CreateWindow("Open Player", 1280, 720,
                                       SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIDDEN | SDL_WINDOW_HIGH_PIXEL_DENSITY);
    if (!win) return fprintf(stderr, "SDL: %s\n", SDL_GetError()), 1;

    int rc = 1;
    {  // scope: tear everything down before SDL_Quit
        Display disp;
        if (!disp.init(win)) return 1;
        cudaSetDevice(disp.cuda_device());
        cudaStream_t st;
        cudaStreamCreate(&st);

        Media media;
        if (!media.open(path, disp.cuda_device())) return 1;
        int w = media.width(), h = media.height();
        // Like RTX VSR: only upscale video smaller than the screen in both directions. A 4K video on a 4K screen,
        // including 3840x1608 widescreen, would be blown up to 8K (7 fps on an RTX 5080) just to be shrunk back.
        int screen_w = 0, screen_h = 0;
        if (const SDL_DisplayMode* m = SDL_GetDesktopDisplayMode(screen_id))
            screen_w = (int)std::lround(m->w * m->pixel_density), screen_h = (int)std::lround(m->h * m->pixel_density);
        bool worth_it = !screen_h || (w < screen_w && h < screen_h);

        // show the window now: building a TensorRT engine for a new resolution takes up to a minute
        SDL_Rect area;
        SDL_GetDisplayUsableBounds(screen_id, &area);
        float fit = std::fmin(1.f, std::fmin(0.9f * area.w / w, 0.9f * area.h / h));
        SDL_SetWindowSize(win, (int)(w * fit), (int)(h * fit));
        SDL_SetWindowPosition(win, SDL_WINDOWPOS_CENTERED_DISPLAY(screen_id), SDL_WINDOWPOS_CENTERED_DISPLAY(screen_id));
        SDL_SetWindowTitle(win, "Open Player: preparing the upscaler (a new resolution takes up to a minute)");
        SDL_ShowWindow(win);
        SDL_SyncWindow(win);
        // Wayland compositors (WSLg's too) ignore window positions; fullscreen is the one way to pick a monitor
        if (screen && SDL_GetDisplayForWindow(win) != screen_id) {
            SDL_SetWindowFullscreenMode(win, SDL_GetDesktopDisplayMode(screen_id));
            SDL_SetWindowFullscreen(win, true);
            SDL_SyncWindow(win);
            fprintf(stderr, "screen %d: the window system won't place windows, so playing fullscreen there\n", screen);
        }
        SDL_PumpEvents();

        // when the network's fixed factor would overshoot the screen (1440p x2 = 5K on a 4K screen), shrink the
        // video first so the output lands exactly on it: 2560x1440 -> 1920x1080 -> 3840x2160. Gives up a little
        // source detail for 25 ms per frame instead of 46 (which dropped 1440p30 to 18 fps with the standard model).
        int ms = worth_it ? Vsr::model_scale(model) : 0, nw = w, nh = h;
        if (ms > 0 && screen_h && (w * ms > screen_w || h * ms > screen_h)) {
            double k = std::fmin((double)screen_w / (w * ms), (double)screen_h / (h * ms));
            nw = (int)std::lround(w * k) & ~1, nh = (int)std::lround(h * k) & ~1;
            fprintf(stderr, "VSR: shrinking %dx%d to %dx%d first, so %dx lands on the %dx%d screen\n", w, h, nw, nh,
                    ms, screen_w, screen_h);
        }
        Vsr vsr;
        bool have_vsr = worth_it && vsr.open(model, nw, nh, st);
        bool shrink = have_vsr && (nw != w || nh != h);
        __half* full = nullptr;  // full-size RGB frame, when it gets shrunk before the network
        if (shrink) cudaMalloc(&full, 3ull * w * h * 2);
        if (!worth_it)
            fprintf(stderr, "VSR off: the video (%dx%d) already fills the screen (%dx%d)\n", w, h, screen_w, screen_h);
        else if (!have_vsr) fprintf(stderr, "VSR unavailable (no %s?), playing without upscaling\n", model);
        vsr_on &= have_vsr;
        int s = have_vsr ? vsr.scale() : 1;
        __half* plain = nullptr;  // RGB frame when there is no VSR engine to borrow the input buffer from
        if (!have_vsr) cudaMalloc(&plain, 3ull * w * h * 2);
        if (!disp.alloc(std::max(w, nw * s), std::max(h, nh * s))) return fprintf(stderr, "display: out of GPU memory\n"), 1;

        const double dur = media.duration();
        bool running = true, paused = false, reset = true, step = false, size_warned = false;
        bool dirty = true, last_bar = false;  // redraw only when the picture changed: each redraw costs a full copy
        int serial = media.serial(), sw = 0, sh = 0, frames_shown = 0;
        float volume = 1;
        const __half* shown = nullptr;
        double pos = 0, wall_base = 0, last_mouse = -10, last_title = -10, last_report = 0;
        Diag diag;
        bool info_done = false;
        cudaEvent_t ev[5];  // GPU timestamps: before convert, after convert, after VSR, before/after rgba
        for (auto& e : ev) cudaEventCreate(&e);
        auto gpu_ms = [&](int a, int b) {
            float ms = 0;
            cudaEventElapsedTime(&ms, ev[a], ev[b]);
            return ms;
        };
        Uint64 wall_t0 = SDL_GetTicksNS();
        auto now = [] { return SDL_GetTicksNS() / 1e9; };
        // Follow the audio while it plays; when it runs dry (ended early, gap, not started) carry on in wall time
        // from where it was, so video never waits on audio that isn't coming.
        auto clock = [&] {
            double a = media.has_audio() && !paused ? media.audio_clock() : NAN;
            if (!std::isnan(a)) {
                wall_base = a, wall_t0 = SDL_GetTicksNS();
                return a;
            }
            return paused ? wall_base : wall_base + (SDL_GetTicksNS() - wall_t0) / 1e9;
        };
        auto seek = [&](double t) {
            media.seek(t);
            wall_base = std::fmax(0, t), wall_t0 = SDL_GetTicksNS();
        };
        auto set_paused = [&](bool p) {
            if (p == paused) return;
            if (p) wall_base = clock();
            else wall_t0 = SDL_GetTicksNS();
            paused = p;
            media.set_paused(p);
        };

        while (running) {
            SDL_Event e;
            while (SDL_PollEvent(&e)) {
                if (e.type == SDL_EVENT_QUIT) running = false;
                else if (e.type == SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED) disp.resized(), dirty = true;
                else if (e.type == SDL_EVENT_WINDOW_EXPOSED) dirty = true;
                else if (e.type == SDL_EVENT_MOUSE_MOTION) last_mouse = now();
                else if (e.type == SDL_EVENT_MOUSE_BUTTON_DOWN && e.button.button == SDL_BUTTON_LEFT) {
                    int ww, wh;
                    SDL_GetWindowSize(win, &ww, &wh);
                    // the picture is letterboxed inside the window (same fit as Display::present); the seek bar
                    // is drawn along the picture's bottom edge, so clicks are measured against the picture
                    float pw = sw ? sw : w, ph = sh ? sh : h, k = std::fmin(ww / pw, wh / ph);
                    float x0 = (ww - pw * k) / 2, y0 = (wh - ph * k) / 2, bottom = y0 + ph * k;
                    if (e.button.clicks == 2) SDL_SetWindowFullscreen(win, !(SDL_GetWindowFlags(win) & SDL_WINDOW_FULLSCREEN));
                    else if (dur > 0 && e.button.y > bottom - ph * k * 0.08f && e.button.y <= bottom)
                        seek(std::clamp((e.button.x - x0) / (pw * k), 0.f, 1.f) * dur);
                } else if (e.type == SDL_EVENT_KEY_DOWN) {
                    bool full = SDL_GetWindowFlags(win) & SDL_WINDOW_FULLSCREEN;
                    switch (e.key.key) {
                    case SDLK_SPACE: set_paused(!paused); break;
                    case SDLK_LEFT: seek(pos - 5); break;
                    case SDLK_RIGHT: seek(pos + 5); break;
                    case SDLK_UP: media.set_volume(volume = std::fmin(volume + 0.1f, 2.f)); break;
                    case SDLK_DOWN: media.set_volume(volume = std::fmax(volume - 0.1f, 0.f)); break;
                    case SDLK_V: if (have_vsr) vsr_on = !vsr_on, reset = true, step |= paused; break;  // paused: show it now
                    case SDLK_F: SDL_SetWindowFullscreen(win, !full); break;
                    case SDLK_ESCAPE: if (full) { SDL_SetWindowFullscreen(win, false); break; } [[fallthrough]];
                    case SDLK_Q: running = false; break;
                    }
                    last_title = -10;  // refresh the title now
                }
            }
            if (media.serial() != serial) {  // a seek finished: VSR history is stale; show the new spot even if paused
                serial = media.serial();
                reset = step = true;
            }

            AVFrame* f;
            double pts;
            if ((!paused || step) && media.take_due(clock(), step, f, pts)) {
                if (f->width == w && f->height == h) {
                    __half* in = shrink ? full : have_vsr ? vsr.input() : plain;
                    Uint64 u0 = SDL_GetTicksNS();
                    cudaEventRecord(ev[0], st);
                    YuvFrame yuv = media.to_gpu(f, st);
                    diag.upload.add((SDL_GetTicksNS() - u0) / 1e6);
                    yuv_to_rgb(yuv, in, st);
                    if (vsr_on && shrink) shrink_rgb(full, w, h, vsr.input(), nw, nh, st);
                    cudaEventRecord(ev[1], st);
                    if (vsr_on) shown = vsr.run(reset), sw = nw * s, sh = nh * s, reset = false;
                    else shown = in, sw = w, sh = h;
                    cudaEventRecord(ev[2], st);
                    cudaStreamSynchronize(st);  // done reading the decoder surface: hand it back to FFmpeg
                    frames_shown++;
                    diag.shown++;
                    dirty = true;
                    diag.convert.add(gpu_ms(0, 1));
                    if (vsr_on) diag.vsr.add(gpu_ms(1, 2));
                    diag.late.add((clock() - pts) * 1000);
                } else if (!size_warned) {  // buffers and engine are sized for the first frame: skip, don't overrun
                    fprintf(stderr, "video changes size mid-file (%dx%d -> %dx%d): skipping those frames\n", w, h,
                            f->width, f->height);
                    size_warned = true;
                }
                av_frame_free(&f);
                pos = pts, step = false;
            } else if (!paused && media.finished()) {
                set_paused(true);  // stay on the last frame, like a normal player
            }

            bool bar = dur > 0 && (paused || now() - last_mouse < 2), drawn = false;
            if (bar != last_bar) last_bar = bar, dirty = true;
            if (shown && dirty) {
                uint8_t* target = disp.begin_frame();
                cudaEventRecord(ev[3], st);
                rgb_to_rgba8(shown, sw, sh, target, bar ? pos / dur : -1, st);
                cudaEventRecord(ev[4], st);
                drawn = disp.present(sw, sh, st);
                if (drawn) {
                    dirty = false;
                    diag.redraws++;
                    diag.rgba.add(gpu_ms(3, 4));
                    diag.copy.add(disp.last_copy_ms);
                    diag.map.add(disp.last_map_ms);
                    diag.present.add(disp.last_present_ms);
                }
            }
            if (!drawn) SDL_Delay(2);  // nothing new to show (or minimized); a redraw is paced by vsync instead
            if (!paused) diag.queue.add(media.queued_frames());

            if (stats && now() - last_report >= 1) {
                last_report = now();
                if (!info_done && !media.decode_desc().empty()) {
                    printf("info {\"gpu\":\"%s\",\"display\":\"%s\",\"decode\":\"%s\",\"video\":\"%dx%d@%.2f\","
                           "\"screen\":\"%dx%d\",\"window_on_planned_screen\":%s,\"vsr\":\"%s\",\"tensorrt\":\"%s\"}\n",
                           disp.gpu_name(), disp.zero_copy() ? "zero-copy" : "CPU copy", media.decode_desc().c_str(),
                           w, h, media.fps(), screen_w, screen_h,
                           SDL_GetDisplayForWindow(win) == screen_id ? "true" : "false",
                           have_vsr ? (std::to_string(s) + "x" +
                                       (shrink ? " from " + std::to_string(nw) + "x" + std::to_string(nh) : "")).c_str()
                                    : worth_it ? "none" : "off (fills screen)",
                           trt_version().c_str());
                    info_done = true;
                }
                size_t free_b = 0, total_b = 0;
                cudaMemGetInfo(&free_b, &total_b);
                diag.report(pos, media.fps(), media.take_dropped(), media.audio_buffered(),
                            (total_b - free_b) / 1048576.0, vsr_on, paused);
            }

            if (now() - last_title > 0.5) {
                last_title = now();
                std::string t = std::string(paused ? "|| " : "") + clock_text(pos) + " / " + clock_text(dur) +
                                (have_vsr ? (vsr_on ? "   VSR " + std::to_string(s) + "x" : "   VSR off") : "") +
                                "   vol " + std::to_string((int)std::lround(volume * 100)) + "%   " + path;
                SDL_SetWindowTitle(win, t.c_str());
            }
        }
        for (auto& e : ev) cudaEventDestroy(e);
        cudaFree(plain);
        cudaFree(full);
        fprintf(stderr, "shown %d frames, stopped at %s\n", frames_shown, clock_text(pos).c_str());
        rc = 0;
    }
    SDL_DestroyWindow(win);
    SDL_Quit();
    return rc;
}
