// Open Player: video player that upscales with our VSR network on the GPU.
// decode (NVDEC) -> YUV->RGB (CUDA) -> VSR (TensorRT) -> RGBA (CUDA) -> Vulkan blit to the window
#include <SDL3/SDL.h>
#include <dlfcn.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <string>

#include "diag.h"
#include "imgui.h"
#include "imgui_impl_sdl3.h"
#include "imgui_impl_vulkan.h"
#include "display.h"
#include "kernels.h"
#include "media.h"
#include "theme.h"
#include "ui.h"
#include "vsr.h"

static const char* USAGE =
    "usage: open-player <video> [--model vsr.onnx] [--no-vsr] [--stats] [--screen N] [--fullscreen]\n"
    "--screen N: play on monitor N (1 = first; --check lists them); upscaling is planned for that monitor\n"
    "       open-player --selftest [vsr.onnx]   compare the TensorRT engine against PyTorch\n"
    "       open-player --check [vsr.onnx]      test each part of the setup on its own\n"
    "--stats: print a JSON line of timings every second (the control panel's Diagnostics reads these)\n"
    "keys:  space pause | left/right seek 5s | up/down volume | v VSR on/off | f or double-click fullscreen\n"
    "       click the bar at the bottom to seek | q quit\n";

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
    bool vsr_on = true, selftest = false, checking = false, stats = false, fullscreen = false;
    int screen = 0;  // --screen N, counted from 1; 0 = the primary display
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "--screen") && i + 1 < argc) screen = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--no-vsr")) vsr_on = false;
        else if (!strcmp(argv[i], "--fullscreen")) fullscreen = true;
        else if (!strcmp(argv[i], "--stats")) stats = true;
        else if (!strcmp(argv[i], "--selftest")) selftest = true;
        else if (!strcmp(argv[i], "--check")) checking = true;
        else if (!strcmp(argv[i], "--uitest")) return ui_selftest();
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
    if (SDL_Surface* icon = SDL_LoadPNG((assets_dir() + "/icon-64.png").c_str()))  // missing: the default icon
        SDL_SetWindowIcon(win, icon), SDL_DestroySurface(icon);

    int rc = 1;
    ImGui::CreateContext();
    ImGui::GetIO().IniFilename = nullptr;  // no imgui.ini next to the video
    load_theme(assets_dir() + "/fonts");   // missing fonts: one line on stderr, default font
    {  // scope: tear everything down before SDL_Quit
        Display disp;
        if (!disp.init(win)) return 1;
        bool ui = ImGui_ImplSDL3_InitForVulkan(win) && disp.init_imgui();
        if (!ui) fprintf(stderr, "display: on-screen controls unavailable\n");
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
            // shortcut: WSLg reports the requested screen even when the window stays on the first, so this can't
            // catch that case; native compositors report the truth
            if (SDL_GetDisplayForWindow(win) == screen_id)
                fprintf(stderr, "screen %d: the window system won't place windows, so playing fullscreen there\n", screen);
            else fprintf(stderr, "screen %d: the window system kept the window elsewhere; drag it across\n", screen);
        }
        if (fullscreen && !(SDL_GetWindowFlags(win) & SDL_WINDOW_FULLSCREEN))
            SDL_SetWindowFullscreen(win, true), SDL_SyncWindow(win);
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
        bool dirty = true;   // redraw only when the picture or the UI changed
        bool fresh = false;  // a new frame is waiting to be converted and copied (redraws of the UI alone skip that)
        bool show_info = false, last_vis = false, named = false, cursor_hidden = false;
        int ui_redraw = 0;  // a few redraws after input, so ImGui's hover and click states catch up
        std::string info;   // info overlay, refreshed once a second
        std::error_code ec;
        std::string abs_path = std::filesystem::absolute(path, ec).string();
        std::vector<std::string> recent = add_recent(load_recent(recent_file()), ec ? path : abs_path);
        save_recent(recent_file(), recent);
        recent = existing_only(recent);  // once: on WSL each check of a /mnt/c path is slow
        int serial = media.serial(), sw = 0, sh = 0, frames_shown = 0;
        float volume = 1;
        const __half* shown = nullptr;
        double pos = 0, wall_base = 0, last_mouse = -10, last_report = 0;
        Diag diag;
        bool info_done = false;
        auto gpu_ms = [](cudaEvent_t a, cudaEvent_t b) {
            float ms = 0;
            cudaEventElapsedTime(&ms, a, b);
            return ms;
        };
        cudaEvent_t rgba_ev[2];  // GPU timestamps around the display conversion
        for (auto& e : rgba_ev) cudaEventCreate(&e);
        // display stream: rgba + copy never wait behind the next frame's network on `st`. High priority, or its
        // small kernel queues behind the network's and the copy took 5-8 ms instead of 1 at 1080p60 -> 4K.
        cudaStream_t ds;
        int prio_low, prio_high;
        cudaDeviceGetStreamPriorityRange(&prio_low, &prio_high);
        cudaStreamCreateWithPriority(&ds, cudaStreamNonBlocking, prio_high);
        // Render-ahead: frames whose convert + network are queued on the GPU, shown once due. With VSR on, the
        // next frame's network runs while the current one is copied and presented (done one after the other,
        // 1080p60 -> 4K reached only 57 fps on WSL). ev: start, after convert, after VSR (= ready).
        struct Prepared { AVFrame* f; double pts; const __half* out; int w, h; bool vsr; cudaEvent_t ev[3]; };
        std::deque<Prepared> ahead;
        auto release = [](Prepared& p) {
            cudaEventSynchronize(p.ev[2]);  // the GPU is done with the decoder surface: hand it back to FFmpeg
            av_frame_free(&p.f);
            for (auto& e : p.ev) cudaEventDestroy(e);
        };
        auto drop_ahead = [&] {  // after a seek or a VSR toggle the prepared frames are stale
            for (auto& p : ahead) release(p);
            ahead.clear();
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
        UiNotice notice;
        auto post = [&](const char* icon, std::string text, float level = -1, double hold = 1) {
            notice = UiNotice{icon, std::move(text), level, now(), hold};
            if (paused) dirty = true;  // playing: the next video frame draws it
        };
        // shared by the keys and the on-screen controls
        auto toggle_pause = [&] {
            set_paused(!paused);
            post(paused ? ICON_PAUSE : ICON_PLAY, "");
        };
        auto set_volume = [&](float v) {
            media.set_volume(volume = std::clamp(v, 0.f, 2.f));
            post(volume > 0 ? ICON_VOLUME : ICON_VOLUME_OFF, "Volume " + std::to_string((int)std::lround(volume * 100)) + "%",
                 notice_level(volume));
        };
        auto toggle_vsr = [&] {  // paused: show the change now
            if (!have_vsr) return;
            vsr_on = !vsr_on, reset = true, step |= paused, drop_ahead();
            post(ICON_SPARKLES, vsr_on ? "VSR on, " + std::to_string(s) + "x" : "VSR off");
        };
        auto key_seek = [&](double dt) {  // the seek bar shows its own position, so only keys post a notice
            double t = std::fmax(0, pos + dt);
            seek(t);
            post(dt > 0 ? ICON_FORWARD_5 : ICON_BACKWARD_5, std::string(dt > 0 ? "+5 s" : "−5 s") + " · " + clock_text(t));
        };
        auto toggle_fullscreen = [&] { SDL_SetWindowFullscreen(win, !(SDL_GetWindowFlags(win) & SDL_WINDOW_FULLSCREEN)); };
        // Opening another file restarts the player on it with the same arguments: every buffer, the engine and
        // the window size depend on the video, so a fresh start is the simple, safe way to switch.
        auto open_video = [&](const std::string& p) {
            // the new process records it in recent files once it opened; a folder or a missing file would
            // otherwise end playback for nothing
            if (!can_open(p)) return (void)fprintf(stderr, "can't open %s: not a file\n", p.c_str());
            std::vector<std::string> args = restart_args(std::vector<std::string>(argv, argv + argc), p);
            std::vector<char*> cargs;
            for (auto& a : args) cargs.push_back(a.data());
            cargs.push_back(nullptr);
            execv("/proc/self/exe", cargs.data());
            fprintf(stderr, "can't open %s: restart failed (%s)\n", p.c_str(), strerror(errno));
        };
        // the dialog answers from another thread: hand the path to the main loop as an event
        auto show_open_dialog = [&] {
            static const SDL_DialogFileFilter filters[] = {
                {"Videos", "mp4;mkv;webm;mov;avi;m4v;ts;h264;hevc;wmv;ogv"}, {"All files", "*"}};
            SDL_ShowOpenFileDialog(
                [](void*, const char* const* files, int) {
                    if (!files || !files[0]) return;  // cancelled or failed
                    SDL_Event ev{SDL_EVENT_USER};
                    ev.user.data1 = SDL_strdup(files[0]);
                    SDL_PushEvent(&ev);
                },
                nullptr, win, filters, 2, nullptr, false);
        };
        auto dims = [](int a, int b) { return std::to_string(a) + "x" + std::to_string(b); };
        const std::string vsr_desc = have_vsr ? std::to_string(s) + "x, " + dims(nw, nh) + " -> " + dims(nw * s, nh * s) +
                                                    (shrink ? " (shrunk from " + dims(w, h) + ")" : "")
                                     : worth_it ? "none" : "off (fills screen)";
        SDL_SetWindowTitle(win, std::filesystem::path(path).filename().string().c_str());

        while (running) {
            SDL_Event e;
            while (SDL_PollEvent(&e)) {
                if (ui) ImGui_ImplSDL3_ProcessEvent(&e), ui_redraw = 2;
                if (e.type == SDL_EVENT_QUIT) running = false;
                else if (e.type == SDL_EVENT_DROP_FILE && e.drop.data) open_video(e.drop.data);
                else if (e.type == SDL_EVENT_USER) {  // a file picked in the open dialog
                    std::string p = (char*)e.user.data1;
                    SDL_free(e.user.data1);
                    open_video(p);
                }
                else if (e.type == SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED) disp.resized(), dirty = true;
                else if (e.type == SDL_EVENT_WINDOW_EXPOSED) dirty = true;
                else if (e.type == SDL_EVENT_MOUSE_MOTION) last_mouse = now(), SDL_ShowCursor();
                else if (e.type == SDL_EVENT_MOUSE_BUTTON_DOWN && e.button.button == SDL_BUTTON_LEFT &&
                         e.button.clicks == 2 && !(ui && ImGui::GetIO().WantCaptureMouse))
                    toggle_fullscreen();
                else if (e.type == SDL_EVENT_KEY_DOWN && !(ui && ui_has_keyboard())) {
                    switch (e.key.key) {
                    case SDLK_SPACE: toggle_pause(); break;
                    case SDLK_LEFT: key_seek(-5); break;
                    case SDLK_RIGHT: key_seek(5); break;
                    case SDLK_UP: set_volume(volume + 0.1f); break;
                    case SDLK_DOWN: set_volume(volume - 0.1f); break;
                    case SDLK_V: toggle_vsr(); break;
                    case SDLK_F: toggle_fullscreen(); break;
                    case SDLK_I: show_info = !show_info, dirty = true; break;
                    case SDLK_ESCAPE:
                        if (SDL_GetWindowFlags(win) & SDL_WINDOW_FULLSCREEN) { toggle_fullscreen(); break; }
                        [[fallthrough]];
                    case SDLK_Q: running = false; break;
                    }
                }
            }
            if (media.serial() != serial) {  // a seek finished: VSR history is stale; show the new spot even if paused
                serial = media.serial();
                reset = step = true;
                drop_ahead();
            }

            // Prepare a frame up to one frame early (VSR on). A second one only once the first is due, so it's
            // shown in this same pass: the network's two output buffers then never overwrite the picture on screen.
            double lead = vsr_on && !paused && media.fps() > 0 ? 1 / media.fps() : 0;
            bool slot = ahead.empty() || (vsr_on && ahead.size() == 1 && ahead.front().pts <= clock());
            AVFrame* f;
            double pts;
            if (slot && (!paused || step) && media.take_due(clock(), lead, step && ahead.empty(), f, pts)) {
                if (f->width == w && f->height == h) {
                    Prepared p{f, pts, nullptr, w, h, vsr_on, {}};
                    for (auto& e : p.ev) cudaEventCreate(&e);
                    __half* in = shrink ? full : have_vsr ? vsr.input() : plain;
                    Uint64 u0 = SDL_GetTicksNS();
                    cudaEventRecord(p.ev[0], st);
                    YuvFrame yuv = media.to_gpu(f, st);
                    diag.upload.add((SDL_GetTicksNS() - u0) / 1e6);
                    yuv_to_rgb(yuv, in, st);
                    if (vsr_on && shrink) shrink_rgb(full, w, h, vsr.input(), nw, nh, st);
                    cudaEventRecord(p.ev[1], st);
                    if (vsr_on) p.out = vsr.run(reset), p.w = nw * s, p.h = nh * s, reset = false;
                    else p.out = in;
                    cudaEventRecord(p.ev[2], st);
                    ahead.push_back(p);
                } else {
                    if (!size_warned) {  // buffers and engine are sized for the first frame: skip, don't overrun
                        fprintf(stderr, "video changes size mid-file (%dx%d -> %dx%d): skipping those frames\n", w, h,
                                f->width, f->height);
                        size_warned = true;
                    }
                    av_frame_free(&f);
                    pos = pts, step = false;
                }
            }
            if (!ahead.empty() && (ahead.front().pts <= clock() || step)) {  // the oldest prepared frame is due
                Prepared& p = ahead.front();
                cudaEventSynchronize(p.ev[2]);  // usually long done: the GPU had up to a frame's time
                shown = p.out, sw = p.w, sh = p.h, pos = p.pts, step = false, dirty = fresh = true;
                frames_shown++, diag.shown++;
                diag.convert.add(gpu_ms(p.ev[0], p.ev[1]));
                if (p.vsr) diag.vsr.add(gpu_ms(p.ev[1], p.ev[2]));
                diag.late.add((clock() - p.pts) * 1000);
                release(p);
                ahead.pop_front();
            } else if (!paused && ahead.empty() && media.finished()) {
                set_paused(true);  // stay on the last frame, like a normal player
            }

            if (ui) {
                Uint64 ui0 = SDL_GetTicksNS();
                ImGui_ImplVulkan_NewFrame();
                ImGui_ImplSDL3_NewFrame();
                ImGui::NewFrame();
                bool full = SDL_GetWindowFlags(win) & SDL_WINDOW_FULLSCREEN;
                if (shown && !named) named = true, post(nullptr, std::filesystem::path(path).filename().string(), -1, 2);
                int pw = 0, ph = 0;
                SDL_GetWindowSizeInPixels(win, &pw, &ph);
                auto acts = draw_ui({pos, dur, now(), last_mouse, paused, have_vsr, vsr_on, full, show_info, volume, info, recent,
                                     std::clamp(ph / 1080.f, 1.f, 2.5f), notice, s});
                ImGui::Render();
                diag.ui.add((SDL_GetTicksNS() - ui0) / 1e6);
                diag.ui_vtx.add(ImGui::GetDrawData()->TotalVtxCount);
                for (const UiAction& a : acts) switch (a.kind) {
                    case UiAction::TogglePause: toggle_pause(); break;
                    case UiAction::Seek: seek(a.value); break;
                    case UiAction::SetVolume: set_volume((float)a.value); break;
                    case UiAction::ToggleVsr: toggle_vsr(); break;
                    case UiAction::ToggleFullscreen: toggle_fullscreen(); break;
                    case UiAction::ToggleInfo: show_info = !show_info; break;
                    case UiAction::Quit: running = false; break;
                    case UiAction::Open: show_open_dialog(); break;
                    case UiAction::OpenPath: open_video(a.path); break;
                    }
                // fullscreen: no cursor over a hidden capsule (mouse motion shows it again)
                bool hide = full && ui_controls_hidden();
                if (hide != cursor_hidden) { cursor_hidden = hide; if (hide) SDL_HideCursor(); else SDL_ShowCursor(); }
                bool vis = paused || now() - last_mouse < 2;
                bool ui_dirty = !acts.empty() || vis != last_vis || ui_redraw > 0 || ImGui::IsAnyItemActive() ||
                                ImGui::IsAnyItemHovered() || ui_animating(now()) || ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId);
                last_vis = vis, ui_redraw = std::max(ui_redraw - 1, 0);
                // while playing, the UI rides along with the next video frame: a redraw of its own would take a
                // vsync slot from the video
                if (paused && ui_dirty) dirty = true;
            }
            bool drawn = false;
            if (shown && dirty) {
                if (fresh) {
                    uint8_t* target = disp.begin_frame();
                    cudaEventRecord(rgba_ev[0], ds);
                    rgb_to_rgba8(shown, sw, sh, target, ds);
                    cudaEventRecord(rgba_ev[1], ds);
                }
                drawn = disp.present(sw, sh, ds, fresh);
                if (drawn) {
                    diag.redraws++;
                    if (fresh) {
                        diag.rgba.add(gpu_ms(rgba_ev[0], rgba_ev[1]));
                        diag.copy.add(disp.last_copy_ms);
                        diag.map.add(disp.last_map_ms);
                    }
                    diag.present.add(disp.last_present_ms);
                    dirty = fresh = false;
                }
            }
            if (!drawn) SDL_Delay(2);  // nothing new to show (or minimized); a redraw is paced by vsync instead
            if (!paused) diag.queue.add(media.queued_frames());

            if (now() - last_report >= 1) {  // info overlay; with --stats also one JSON line
                last_report = now();
                int dropped = media.take_dropped();
                char buf[512];
                snprintf(buf, sizeof buf,
                         "GPU      %s (%s)\nDecode   %s\nVideo    %dx%d @ %.2f fps\nScreen   %dx%d\n"
                         "VSR      %s%s\nShown    %d fps, %d dropped\nNetwork  %.1f ms per frame\nUI       %.2f ms per pass (max %.2f)",
                         disp.gpu_name(), disp.zero_copy() ? "zero-copy" : "CPU copy", media.decode_desc().c_str(), w, h,
                         media.fps(), screen_w, screen_h, vsr_desc.c_str(), have_vsr && !vsr_on ? " (switched off)" : "",
                         diag.shown, dropped, diag.vsr.mean(), diag.ui.mean(), diag.ui.max);
                if (info != buf) info = buf, dirty |= show_info && paused;
                if (stats && !info_done && !media.decode_desc().empty()) {
                    printf("info {\"gpu\":\"%s\",\"display\":\"%s\",\"decode\":\"%s\",\"video\":\"%dx%d@%.2f\","
                           "\"screen\":\"%dx%d\",\"window_on_planned_screen\":%s,\"vsr\":\"%s\",\"tensorrt\":\"%s\"}\n",
                           disp.gpu_name(), disp.zero_copy() ? "zero-copy" : "CPU copy", media.decode_desc().c_str(),
                           w, h, media.fps(), screen_w, screen_h,
                           SDL_GetDisplayForWindow(win) == screen_id ? "true" : "false", vsr_desc.c_str(),
                           trt_version().c_str());
                    info_done = true;
                }
                if (stats) {
                    size_t free_b = 0, total_b = 0;
                    cudaMemGetInfo(&free_b, &total_b);
                    diag.report(pos, media.fps(), dropped, media.audio_buffered(), (total_b - free_b) / 1048576.0,
                                vsr_on, paused);
                } else {
                    diag = Diag();
                }
            }
        }
        drop_ahead();
        for (auto& e : rgba_ev) cudaEventDestroy(e);
        cudaStreamDestroy(ds);
        cudaFree(plain);
        cudaFree(full);
        fprintf(stderr, "shown %d frames, stopped at %s\n", frames_shown, clock_text(pos).c_str());
        rc = 0;
    }
    if (ImGui::GetIO().BackendPlatformUserData) ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();
    SDL_DestroyWindow(win);
    SDL_Quit();
    return rc;
}
