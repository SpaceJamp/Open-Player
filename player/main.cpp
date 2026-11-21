// Open Player: video player that upscales with our VSR network on the GPU.
// decode (NVDEC) -> YUV->RGB (CUDA) -> VSR (TensorRT) -> RGBA (CUDA) -> Vulkan blit to the window
#include <SDL3/SDL.h>
#include <atomic>
#include <cuda.h>
#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <process.h>
#include <windows.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#endif

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
#include "collect.h"
#include "settings.h"
#include "build_info.h"  // generated: OPEN_PLAYER_BUILD
#include "logfile.h"
#ifdef HAVE_RTX_VSR
#include "rtx_vsr.h"
#endif

static const char* USAGE =
    "usage: open-player [video] [--model vsr.onnx] [--no-vsr] [--no-rtx] [--stats] [--screen N] [--fullscreen]\n"
    "no video (or double-clicked): pick one in a file dialog; cancel quits\n"
    "--no-rtx: on Windows, upscale with Pixel Flow (our model) instead of NVIDIA RTX VSR\n"
    "--cpu-decode: decode on the CPU instead of NVDEC (upscaling stays on the GPU)\n"
    "--start T [--paused]: start T seconds in (paused); the player reopens itself this way after a settings change\n"
    "--collect DIR: on Windows with RTX VSR, save clips and their RTX upscale to DIR for train.py --pairs (2 GB cap)\n"
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
// The connected monitors in SDL's order (index + 1 = --screen N); ids, when given, gets their SDL ids.
static std::vector<ScreenInfo> list_screens(std::vector<SDL_DisplayID>* ids = nullptr) {
    std::vector<ScreenInfo> out;
    if (ids) ids->clear();
    int n = 0;
    SDL_DisplayID* d = SDL_GetDisplays(&n);
    for (int i = 0; i < n; i++) {
        const SDL_DisplayMode* m = SDL_GetDesktopDisplayMode(d[i]);
        const char* name = SDL_GetDisplayName(d[i]);
        out.push_back({name ? name : "?", m ? (int)std::lround(m->w * m->pixel_density) : 0,
                       m ? (int)std::lround(m->h * m->pixel_density) : 0, m ? m->refresh_rate : 0,
                       d[i] == SDL_GetPrimaryDisplay()});
        if (ids) ids->push_back(d[i]);
    }
    SDL_free(d);
    return out;
}

static int check(const char* model) {
    printf("Open Player %s\nLog: %s\n", OPEN_PLAYER_BUILD, log_file().c_str());
    int fails = 0;
    auto line = [&](const char* verdict, const char* what, const std::string& detail) {
        printf("%-4s  %-16s %s\n", verdict, what, detail.c_str());
        fails += !strcmp(verdict, "FAIL");
    };
    int n = 0, drv = 0;
    cudaGetDeviceCount(&n);
    cudaDriverGetVersion(&drv);
    int major = 0, minor = 0;  // driver API and attributes, not cudaGetDeviceProperties: see rtx_vsr.cpp
    char name[256] = "";
    CUdevice cd;
    if (n > 0 && cuDeviceGet(&cd, 0) == CUDA_SUCCESS && cuDeviceGetName(name, sizeof name, cd) == CUDA_SUCCESS &&
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0) == cudaSuccess &&
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0) == cudaSuccess)
        line("PASS", "CUDA", std::string(name) + ", sm_" + std::to_string(major) + std::to_string(minor) +
                                 ", driver CUDA " + std::to_string(drv / 1000) + "." + std::to_string(drv % 1000 / 10));
    else line("FAIL", "CUDA", "no CUDA GPU visible (driver missing, or on WSL: Windows driver too old)");

#ifdef _WIN32
    HMODULE nvcuvid = LoadLibraryA("nvcuvid.dll");  // what FFmpeg's NVDEC loads
#else
    void* nvcuvid = dlopen("libnvcuvid.so.1", RTLD_NOW);
#endif
    line(nvcuvid ? "PASS" : "WARN", "NVDEC", nvcuvid ? "nvcuvid found" : "nvcuvid missing: videos decode on the CPU");
#ifdef _WIN32
    if (nvcuvid) FreeLibrary(nvcuvid);
#else
    if (nvcuvid) dlclose(nvcuvid);
#endif

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
        auto screens = list_screens();
        for (size_t i = 0; i < screens.size(); i++) {
            char d[160];
            snprintf(d, sizeof d, "%s, %dx%d @ %.0f Hz%s (--screen %d)", screens[i].name.c_str(), screens[i].w,
                     screens[i].h, screens[i].hz, screens[i].primary ? ", primary" : "", (int)i + 1);
            line("INFO", ("Screen " + std::to_string(i + 1)).c_str(), d);
        }
        SDL_AudioDeviceID dev = SDL_OpenAudioDevice(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, nullptr);
        line(dev ? "PASS" : "WARN", "Audio", dev ? SDL_GetAudioDeviceName(dev) : "no audio output: video plays silent");
        if (dev) SDL_CloseAudioDevice(dev);
        SDL_Quit();
    } else line("FAIL", "SDL", SDL_GetError());

    line("PASS", "TensorRT", trt_version());
#ifdef HAVE_RTX_VSR
    bool rtx = RtxVsr::selftest() == 0;
    line(rtx ? "PASS" : "WARN", "RTX VSR", rtx ? "NVIDIA RTX Video Super Resolution works"
                                              : "unavailable (GPU or driver): Pixel Flow upscales instead");
#endif
    if (!std::filesystem::exists(model)) line("FAIL", "Model", std::string(model) + " not found: train, then export");
    else line(Vsr::selftest(model) == 0 ? "PASS" : "FAIL", "Model", model);
    printf(fails ? "%d problem(s) found\n" : "all good\n", fails);
    return fails ? 1 : 0;
}

static const SDL_DialogFileFilter VIDEO_FILTERS[] = {{"Videos", "mp4;mkv;webm;mov;avi;m4v;ts;h264;hevc;wmv;ogv"},
                                                      {"All files", "*"}};

int main(int argc, char** argv) {
    const char *path = nullptr, *model = "vsr.onnx", *collect_dir = nullptr;
    bool vsr_on = true, selftest = false, checking = false, stats = false, fullscreen = false, want_rtx = true,
         cpu_decode = false, start_paused = false;
    int screen = 0;  // --screen N, counted from 1; 0 = the primary display
    double start_at = -1;  // --start T: seconds; < 0: from the beginning
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "--start") && i + 1 < argc) start_at = std::max(0.0, atof(argv[++i]));
        else if (!strcmp(argv[i], "--paused")) start_paused = true;
        else if (!strcmp(argv[i], "--screen") && i + 1 < argc) screen = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--collect") && i + 1 < argc) collect_dir = argv[++i];
        else if (!strcmp(argv[i], "--no-vsr")) vsr_on = false;
        else if (!strcmp(argv[i], "--no-rtx")) want_rtx = false;
        else if (!strcmp(argv[i], "--cpu-decode")) cpu_decode = true;
        else if (!strcmp(argv[i], "--fullscreen")) fullscreen = true;
        else if (!strcmp(argv[i], "--stats")) stats = true;
        else if (!strcmp(argv[i], "--selftest")) selftest = true;
        else if (!strcmp(argv[i], "--check")) checking = true;
        else if (!strcmp(argv[i], "--uitest")) return ui_selftest();
        else if (!strcmp(argv[i], "--version")) return printf("Open Player %s\n", OPEN_PLAYER_BUILD), 0;
        else if (argv[i][0] != '-') (selftest || checking ? model : path) = argv[i];
        else return fputs(USAGE, stderr), 2;
    }
    // open-player.log: everything below that goes to stderr is also saved (with the libraries' own messages)
    if (start_log(log_file())) {
        atexit(stop_log);
        fprintf(stderr, "Open Player %s\n", OPEN_PLAYER_BUILD);
    } else fprintf(stderr, "log: can't write %s, running without a log file\n", log_file().c_str());
    // saved settings, with this run's flags on top; only the settings card and the volume change `saved`
    Settings saved = load_settings(settings_file()), cfg = saved;
    if (!want_rtx && cfg.upscaler == Upscaler::Rtx) cfg.upscaler = Upscaler::PixelFlow;
    if (!vsr_on) cfg.upscaler = Upscaler::Off;
    cfg.cpu_decode |= cpu_decode;
    cfg.start_fullscreen |= fullscreen;
    vsr_on = cfg.upscaler != Upscaler::Off, want_rtx = cfg.upscaler == Upscaler::Rtx;
    cpu_decode = cfg.cpu_decode, fullscreen = cfg.start_fullscreen;
    // double-clicked or "Open with": the working folder is anywhere, so also look next to the program
    std::string near_exe;
    if (!std::filesystem::exists(model) && std::filesystem::path(model).is_relative())
        for (const char* up : {".", ".."}) {
            auto m = std::filesystem::path(exe_path()).parent_path() / up / model;
            if (std::filesystem::exists(m)) {
                model = (near_exe = m.lexically_normal().string()).c_str();
                break;
            }
        }
    if (selftest) return Vsr::selftest(model);
    if (checking) return check(model);
    // double-clicked with no video: ask for one, like the Open button (cancel quits)
    struct Pick {
        std::string file;
        std::atomic<bool> done{false};
        bool failed = false;
    } pick;
    if (!path && (stats || getenv("SSH_CONNECTION"))) return fputs(USAGE, stderr), 2;  // never wait on a dialog
    if (!path) {
        if (!SDL_Init(SDL_INIT_VIDEO)) return fprintf(stderr, "SDL: %s\n%s", SDL_GetError(), USAGE), 2;
        const char* drv = SDL_GetCurrentVideoDriver();  // headless (tests): no one to answer a dialog
        if (drv && (!strcmp(drv, "offscreen") || !strcmp(drv, "dummy"))) return fputs(USAGE, stderr), 2;
        SDL_ShowOpenFileDialog(
            [](void* u, const char* const* files, int) {  // may answer from another thread
                auto* p = (Pick*)u;
                if (files && files[0]) p->file = files[0];
                if (!files) p->failed = true, fprintf(stderr, "no file dialog (%s)\n%s", SDL_GetError(), USAGE);
                p->done = true;
            },
            &pick, nullptr, VIDEO_FILTERS, 2, nullptr, false);
        for (SDL_Event e; !pick.done;)  // Linux's portal answers through events
            if (SDL_WaitEventTimeout(&e, 50) && e.type == SDL_EVENT_QUIT) return 0;  // Ctrl+C
        if (pick.file.empty()) return pick.failed ? 2 : 0;
        path = pick.file.c_str();
    }

    if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO)) return fprintf(stderr, "SDL: %s\n", SDL_GetError()), 1;
    std::vector<SDL_DisplayID> ids;
    std::vector<ScreenInfo> screens = list_screens(&ids);
    int n_disp = (int)ids.size();
    if (!screen && !cfg.screen.empty()) {  // the saved monitor, found by name and size; gone: the primary
        int i = match_screen(screens, cfg.screen);
        if (i < n_disp && !screens[i].primary) screen = i + 1;
    }
    if (screen > n_disp) fprintf(stderr, "no screen %d: this machine has %d, playing on the primary one\n", screen, n_disp),
                         screen = 0;
    SDL_DisplayID screen_id = !screen ? SDL_GetPrimaryDisplay() : screen > 0 && screen <= n_disp ? ids[screen - 1] : 0;
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
        if (!media.open(path, disp.cuda_device(), !cpu_decode)) return 1;
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
        // NVIDIA RTX VSR goes first on Windows; Pixel Flow when the GPU or driver lacks it, when chosen, and on Linux.
        // shortcut: sized from our model's factor and screen plan, so it needs the model too; RTX can scale freely
#ifdef HAVE_RTX_VSR
        RtxVsr rtx;
        bool rtx_tried = have_vsr && want_rtx;  // otherwise opened on first use (settings card, V after --no-vsr)
        bool rtx_ok = rtx_tried && rtx.open(w, h, nw * s, nh * s, st);
        bool use_rtx = rtx_ok;
        if (have_vsr) fprintf(stderr, "VSR: %s\n", use_rtx ? "NVIDIA RTX Video Super Resolution" : "Pixel Flow");
        Collector col;
        bool collecting = collect_dir && use_rtx && col.open(collect_dir, w, h, nw * s, nh * s, s, st);
        if (collect_dir)
            fprintf(stderr, collecting ? "collect: saving RTX VSR training pairs in %s\n"
                                       : "collect: off (needs RTX VSR at exactly 2x - no shrinking to fit the screen -, "
                                         "a video of at least 128x128, and %s writable and under 2 GB)\n",
                    collect_dir);
#else
        bool use_rtx = false, rtx_ok = false;
        (void)want_rtx;
        if (collect_dir) fprintf(stderr, "collect: needs NVIDIA RTX VSR, which this build doesn't have\n");
#endif
        __half* plain = nullptr;  // RGB frame when there is no VSR engine to borrow the input buffer from
        if (!have_vsr) cudaMalloc(&plain, 3ull * w * h * 2);
        if (!disp.alloc(std::max(w, nw * s), std::max(h, nh * s))) return fprintf(stderr, "display: out of GPU memory\n"), 1;

        const double dur = media.duration();
        bool running = true, paused = false, reset = true, step = false, size_warned = false;
        bool dirty = true;   // redraw only when the picture or the UI changed
        bool fresh = false;  // a new frame is waiting to be converted and copied (redraws of the UI alone skip that)
        bool show_info = cfg.info_at_start, last_vis = false, named = false;
        int ui_redraw = 0;  // a few redraws after input, so ImGui's hover and click states catch up
        std::string info;   // info overlay, refreshed once a second
        std::error_code ec;
        std::string abs_path = std::filesystem::absolute(path, ec).string();
        const std::string video_path = ec ? std::string(path) : abs_path;  // what a reopen plays again
        std::vector<std::string> recent = add_recent(load_recent(recent_file()), ec ? path : abs_path);
        save_recent(recent_file(), recent);
        recent = existing_only(recent);  // once: on WSL each check of a /mnt/c path is slow
        int serial = media.serial(), sw = 0, sh = 0, frames_shown = 0;
        float volume = cfg.remember_volume ? cfg.volume : 1;
        media.set_volume(volume);
        bool volume_unsaved = false;  // saved at most once a second while it changes, and on quit
        double volume_saved_at = 0;
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
#ifdef HAVE_RTX_VSR
            col.restart();  // the clock jumps over the pause: frames on either side aren't one clip
#endif
        };
        UiNotice notice;
        auto post = [&](const char* icon, std::string text, float level = -1, double hold = 1) {
            notice = UiNotice{icon, std::move(text), level, now(), hold};
            if (paused) dirty = true;  // playing: the next video frame draws it
        };
        // shared by the keys and the on-screen controls
        auto toggle_pause = [&] {
            if (paused && media.finished()) seek(0);  // at the end: play it again
            set_paused(!paused);
            post(paused ? ICON_PAUSE : ICON_PLAY, "");
        };
        auto set_volume = [&](float v) {
            media.set_volume(volume = std::clamp(v, 0.f, 2.f));
            if (saved.remember_volume) saved.volume = volume, volume_unsaved = true;
            post(volume > 0 ? ICON_VOLUME : ICON_VOLUME_OFF, "Volume " + std::to_string((int)std::lround(volume * 100)) + "%",
                 notice_level(volume));
        };
        auto dims = [](int a, int b) { return std::to_string(a) + "x" + std::to_string(b); };
        auto describe_vsr = [&] {  // the info overlay's upscaler line
            return !have_vsr ? std::string(worth_it ? "none" : "off (fills screen)")
                   : use_rtx ? "NVIDIA RTX, " + dims(w, h) + " -> " + dims(nw * s, nh * s)  // full-size in
                             : "Pixel Flow " + std::to_string(s) + "x, " + dims(nw, nh) + " -> " + dims(nw * s, nh * s) +
                                   (shrink ? " (shrunk from " + dims(w, h) + ")" : "");
        };
        std::string vsr_desc = describe_vsr();
        // the upscaler V and the badge turn back on: the last one chosen in settings (or the default)
        Upscaler chosen = vsr_on ? (use_rtx ? Upscaler::Rtx : Upscaler::PixelFlow)  // what actually runs
                          : cfg.upscaler != Upscaler::Off ? cfg.upscaler
                          : saved.upscaler != Upscaler::Off ? saved.upscaler : Upscaler::Rtx;
#ifndef HAVE_RTX_VSR
        if (chosen == Upscaler::Rtx) chosen = Upscaler::PixelFlow;  // this build has no RTX VSR: no false warning
#endif
        // live switch; false when that upscaler can't run here (no model, or no RTX VSR on this GPU/driver)
        auto set_upscaler = [&](Upscaler u) {
            if (!have_vsr) return false;
            bool ok = true;
            if (u == Upscaler::Rtx) {
#ifdef HAVE_RTX_VSR
                if (!rtx_tried) rtx_tried = true, rtx_ok = rtx.open(w, h, nw * s, nh * s, st);
#endif
                if (!rtx_ok) u = Upscaler::PixelFlow, ok = false;
            }
            if (u != Upscaler::Off) chosen = u;
            vsr_on = u != Upscaler::Off, use_rtx = u == Upscaler::Rtx;
            reset = true, step |= paused, drop_ahead();  // paused: show the change now
            if (paused && media.finished()) seek(pos);   // at the end no frame is queued: decode the last one again
            vsr_desc = describe_vsr();
            post(ICON_SPARKLES, !ok          ? std::string("NVIDIA RTX isn't available on this GPU or driver")
                                : !vsr_on   ? std::string("VSR off")
                                : use_rtx   ? "NVIDIA RTX, " + std::to_string(s) + "x"
                                            : "Pixel Flow, " + std::to_string(s) + "x",
                 -1, ok ? 1 : 2);
            return ok;
        };
        auto toggle_vsr = [&] { set_upscaler(vsr_on ? Upscaler::Off : chosen); };  // V, the badge: not saved
        double seek_to = NAN;  // a key seek whose frame isn't shown yet: the next key seek builds on it
        auto key_seek = [&](double dt) {  // the seek bar shows its own position, so only keys post a notice
            double t = std::fmax(0, (std::isnan(seek_to) ? pos : seek_to) + dt);
            seek_to = t;
            if (dur > 0) t = std::fmin(t, dur);
            seek(t);
            post(dt > 0 ? ICON_FORWARD_5 : ICON_BACKWARD_5, seek_notice_text(dt, t, dur));
        };
        auto toggle_fullscreen = [&] { SDL_SetWindowFullscreen(win, !(SDL_GetWindowFlags(win) & SDL_WINDOW_FULLSCREEN)); };
        // replaces this process with the player run with args (open another video, reopen after a settings change)
        auto restart_with = [&](std::vector<std::string> args, const std::string& video) {
            if (volume_unsaved) save_settings(settings_file(), saved);
            fprintf(stderr, "reopening: %s\n", video.c_str());
            stop_log();  // the new process starts its own log; this one's last lines must reach the file first
#ifdef _WIN32
            for (auto& a : args) a = quote_arg(a);  // _execv joins the arguments with spaces and doesn't quote them
#endif
            std::vector<char*> cargs;
            for (auto& a : args) cargs.push_back(a.data());
            cargs.push_back(nullptr);
#ifdef _WIN32
            _execv(exe_path().c_str(), cargs.data());
#else
            execv("/proc/self/exe", cargs.data());
#endif
            fprintf(stderr, "can't open %s: restart failed (%s)\n", video.c_str(), strerror(errno));
        };
        // Opening another file restarts the player on it with the same arguments: every buffer, the engine and
        // the window size depend on the video, so a fresh start is the simple, safe way to switch.
        auto open_video = [&](const std::string& p) {
            // the new process records it in recent files once it opened; a folder or a missing file would
            // otherwise end playback for nothing
            if (!can_open(p) || !Media::has_video(p.c_str())) {
                post(ICON_X, "Can't play " + std::filesystem::path(p).filename().string(), -1, 2);
                return (void)fprintf(stderr, "can't open %s: not a playable video\n", p.c_str());
            }
            std::vector<std::string> args = restart_args(std::vector<std::string>(argv, argv + argc), p);
            // keep how it plays now: fullscreen and VSR off survive the restart
            args.erase(std::remove_if(args.begin() + 1, args.end(),
                                      [](const std::string& a) { return a == "--fullscreen" || a == "--no-vsr"; }),
                       args.end());
            if (SDL_GetWindowFlags(win) & SDL_WINDOW_FULLSCREEN) args.push_back("--fullscreen");
            if (have_vsr && !vsr_on) args.push_back("--no-vsr");
            restart_with(args, p);
        };
        // the dialog answers from another thread: hand the path to the main loop as an event
        auto show_open_dialog = [&] {
            SDL_ShowOpenFileDialog(
                [](void*, const char* const* files, int) {
                    if (!files || !files[0]) return;  // cancelled or failed
                    SDL_Event ev{SDL_EVENT_USER};
                    ev.user.data1 = SDL_strdup(files[0]);
                    SDL_PushEvent(&ev);
                },
                nullptr, win, VIDEO_FILTERS, 2, nullptr, false);
        };
        SDL_SetWindowTitle(win, std::filesystem::path(path).filename().string().c_str());
        if (start_at >= 0) seek(start_position(start_at, dur));  // reopened after a settings change
        if (start_paused) set_paused(true), step = true;         // ... while paused: show that frame, stay paused

        // the settings card (S, the gear, the menu); OPEN_PLAYER_SHOW_SETTINGS=1 opens it at start (timing tests)
        bool settings_open = getenv("OPEN_PLAYER_SHOW_SETTINGS") != nullptr;
        std::vector<SDL_DisplayID> screen_ids;
        std::vector<ScreenInfo> monitors = list_screens(&screen_ids);
        std::vector<std::string> pending_restart;  // a reopen waits a moment so its notice gets drawn
        double restart_at = 0;
        auto save_prefs = [&] {
            if (!save_settings(settings_file(), saved))
                fprintf(stderr, "settings: can't save %s\n", settings_file().c_str());
            volume_unsaved = false;
        };
        // keep_screen: a decoder change stays on this monitor; a monitor change goes to the saved one
        auto reopen = [&](const std::string& why, bool keep_screen) {
            post(ICON_SETTINGS, why, -1, 2);
            dirty = true;
            pending_restart = reopen_args(std::vector<std::string>(argv, argv + argc), video_path, pos, paused);
            int here = 0;
            for (size_t i = 0; keep_screen && i < screen_ids.size(); i++)
                if (screen_ids[i] == screen_id) here = (int)i + 1;
            for (auto& f : live_flags(SDL_GetWindowFlags(win) & SDL_WINDOW_FULLSCREEN, have_vsr && !vsr_on,
                                      vsr_on && !use_rtx, here))
                pending_restart.push_back(f);
            restart_at = now() + 0.25;
        };
        auto toggle_settings = [&] {
            settings_open = !settings_open, dirty = true;
            if (settings_open) monitors = list_screens(&screen_ids);
        };

        while (running) {
            if (volume_unsaved && now() - volume_saved_at >= 1)
                save_settings(settings_file(), saved), volume_unsaved = false, volume_saved_at = now();
            if (!pending_restart.empty() && now() >= restart_at) {
                restart_with(pending_restart, video_path);  // returns only if it failed
                pending_restart.clear();
                post(ICON_X, "Couldn't reopen the video", -1, 2);
            }
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
                else if (e.type == SDL_EVENT_WINDOW_EXPOSED || e.type == SDL_EVENT_WINDOW_DISPLAY_CHANGED) dirty = true;
                else if (e.type >= SDL_EVENT_DISPLAY_FIRST && e.type <= SDL_EVENT_DISPLAY_LAST)
                    monitors = list_screens(&screen_ids), dirty = true;  // plugged in or out, moved, mode changed
                else if (e.type == SDL_EVENT_MOUSE_MOTION) last_mouse = now();
                else if (e.type == SDL_EVENT_MOUSE_BUTTON_DOWN && e.button.button == SDL_BUTTON_LEFT &&
                         e.button.clicks == 2 && !(ui && ImGui::GetIO().WantCaptureMouse))
                    toggle_fullscreen();
                else if (e.type == SDL_EVENT_KEY_DOWN && !(ui && ui_has_keyboard()) &&
                         !(e.key.mod & (SDL_KMOD_CTRL | SDL_KMOD_ALT | SDL_KMOD_GUI)) &&
                         // holding a key repeats seeking and volume only; a held toggle would flicker
                         (!e.key.repeat || e.key.key == SDLK_LEFT || e.key.key == SDLK_RIGHT || e.key.key == SDLK_UP ||
                          e.key.key == SDLK_DOWN)) {
                    switch (e.key.key) {
                    case SDLK_SPACE: toggle_pause(); break;
                    case SDLK_LEFT: key_seek(-5); break;
                    case SDLK_RIGHT: key_seek(5); break;
                    case SDLK_UP: set_volume(volume + 0.1f); break;
                    case SDLK_DOWN: set_volume(volume - 0.1f); break;
                    case SDLK_V: toggle_vsr(); break;
                    case SDLK_F: toggle_fullscreen(); break;
                    case SDLK_I: show_info = !show_info, dirty = true; break;
                    case SDLK_S: toggle_settings(); break;
                    case SDLK_ESCAPE:
                        if (settings_open) { toggle_settings(); break; }  // the card first, then fullscreen, then quit
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
#ifdef HAVE_RTX_VSR
                col.restart();
#endif
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
                    if (vsr_on && shrink && !use_rtx) shrink_rgb(full, w, h, vsr.input(), nw, nh, st);
                    cudaEventRecord(p.ev[1], st);
#ifdef HAVE_RTX_VSR
                    if (vsr_on && use_rtx && !(p.out = rtx.run(in))) {  // full-size input
                        use_rtx = rtx_ok = false;  // NVIDIA's upscaler failed: Pixel Flow from here on
                        chosen = Upscaler::PixelFlow;
                        fprintf(stderr, "VSR: NVIDIA RTX failed, switching to Pixel Flow\n");
                        vsr_desc = describe_vsr();
                        if (shrink) shrink_rgb(full, w, h, vsr.input(), nw, nh, st);
                    }
                    if (vsr_on && use_rtx) {
                        p.w = nw * s, p.h = nh * s;
                        if (collecting && !paused) col.add(in, p.out, pts);
                    } else
#endif
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
                shown = p.out, sw = p.w, sh = p.h, pos = p.pts, step = false, dirty = fresh = true, seek_to = NAN;
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
                UiState us{pos, dur, now(), last_mouse, paused, have_vsr, vsr_on, full, show_info, volume, info, recent,
                           ui_scale(ImGui::GetIO().DisplaySize.y), notice, s, use_rtx};
                us.settings_open = settings_open;
                us.settings = cfg;
                us.settings.upscaler = !vsr_on ? Upscaler::Off : use_rtx ? Upscaler::Rtx : Upscaler::PixelFlow;
                us.settings.cpu_decode = cfg.cpu_decode || !media.nvdec();
                us.screens = monitors;
                for (size_t i = 0; i < screen_ids.size(); i++) {
                    if (screen_ids[i] == screen_id) us.screen_index = (int)i;
                    if (screen_ids[i] == SDL_GetDisplayForWindow(win)) us.playing_on = (int)i;
                }
#ifdef HAVE_RTX_VSR
                us.rtx_available = have_vsr && (!rtx_tried || rtx_ok);  // not tried yet: offered, opened on first use
#endif
                us.pxf_available = have_vsr;
                us.nvdec_available = media.nvdec() || cfg.cpu_decode;  // CPU chosen: NVDEC wasn't even tried
                if (!have_vsr)
                    us.vsr_unavailable = worth_it ? "no Pixel Flow model (" + std::string(model) + ") found"
                                                  : "the video already fills the screen";
                us.build = OPEN_PLAYER_BUILD;
                us.log_path = log_file();
                auto acts = draw_ui(us);
                if (full && ui_controls_hidden() && !ui_has_keyboard()) ImGui::SetMouseCursor(ImGuiMouseCursor_None);  // the SDL3 backend hides the OS cursor at its next NewFrame
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
                    case UiAction::ToggleSettings: toggle_settings(); break;
                    case UiAction::SetUpscaler:  // live; saved only when that upscaler can run here
                        if (set_upscaler((Upscaler)(int)a.value))
                            saved.upscaler = cfg.upscaler = (Upscaler)(int)a.value, save_prefs();
                        break;
                    case UiAction::SetDecoder:
                        saved.cpu_decode = cfg.cpu_decode = a.value != 0, save_prefs();
                        reopen(a.value != 0 ? "Switching to CPU decoding..." : "Switching to GPU decoding...", true);
                        break;
                    case UiAction::SetScreen:
                        if (a.value >= 0 && a.value < monitors.size()) {
                            saved.screen = cfg.screen = screen_key(monitors, (int)a.value), save_prefs();
                            reopen("Reopening on screen " + std::to_string((int)a.value + 1) + "...", false);
                        }
                        break;
                    case UiAction::SetStartFullscreen: saved.start_fullscreen = cfg.start_fullscreen = a.value != 0, save_prefs(); break;
                    case UiAction::SetRememberVolume:
                        saved.remember_volume = cfg.remember_volume = a.value != 0;
                        if (saved.remember_volume) saved.volume = volume;
                        save_prefs();
                        break;
                    case UiAction::SetHideAfter: saved.hide_after = cfg.hide_after = (int)a.value, save_prefs(); break;
                    case UiAction::SetInfoAtStart: saved.info_at_start = cfg.info_at_start = a.value != 0, save_prefs(); break;
                    }
                bool vis = paused || settings_open || now() - last_mouse < cfg.hide_after;
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
        if (volume_unsaved) save_settings(settings_file(), saved);
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
