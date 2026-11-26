// Open Player: video player that upscales with our VSR network on the GPU.
// decode (NVDEC) -> YUV->RGB (CUDA) -> VSR (TensorRT) -> RGBA (CUDA) -> Vulkan blit to the window
#include <SDL3/SDL.h>
#include <atomic>
#include <thread>
#include <ctime>
#include <fstream>
#include <functional>
#include <memory>
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
#include "trt_loader.h"
#include "addon.h"
#include "training_addon.h"
#include "training_run.h"
#include "captions.h"
#include "update.h"
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

static int check(const char* model, TrtFrom trt_from, const std::string& trt_ver, const std::string& trt_why) {
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

    std::string tl = trt_line(trt_from, trt_ver, trt_why).substr(10);  // without "TensorRT: "
    line(trt_from != TrtFrom::None ? "PASS" : "WARN", "TensorRT", tl);
#ifdef HAVE_RTX_VSR
    bool rtx = RtxVsr::selftest() == 0;
    line(rtx ? "PASS" : "WARN", "RTX VSR", rtx ? "NVIDIA RTX Video Super Resolution works"
                                              : "unavailable (GPU or driver): Pixel Flow upscales instead");
#else
    bool rtx = false;
#endif
    printf("INFO  %s\nINFO  %s\n", rtx_vsr_line(true, rtx).c_str(), driver_line().c_str());
    // no model is normal: nobody gets one, everyone trains their own Pixel Flow
    if (!std::filesystem::exists(model)) line("INFO", "Model", std::string(model) + " not found: train your own Pixel Flow");
    else if (trt_from == TrtFrom::None) line("WARN", "Model", std::string(model) + ": needs TensorRT to run");
    else line(Vsr::selftest(model) == 0 ? "PASS" : "FAIL", "Model", model);
    printf(fails ? "%d problem(s) found\n" : "all good\n", fails);
    return fails ? 1 : 0;
}

static const SDL_DialogFileFilter VIDEO_FILTERS[] = {{"Videos", "mp4;mkv;webm;mov;avi;m4v;ts;h264;hevc;wmv;ogv"},
                                                      {"All files", "*"}};

int main(int argc, char** argv) {
#ifdef _WIN32
    // DLLs never come from the current folder (with "Open with" that's the video's folder: Downloads, a share)
    SetDllDirectoryW(L"");
    // A windowed program (no console window of its own). Started from a terminal, print there (--check, --version,
    // usage); redirected (tests, the control panel) the pipes are used as they are. The log file gets it either way.
    auto no_handle = [](DWORD h) {  // not a pipe or a file: nothing, or a console handle we must attach to first
        DWORD t = GetFileType(GetStdHandle(h));
        // shortcut: ">NUL" is a character device too and so still prints to the terminal; telling it apart from an
        // inherited console handle needs the console attached first. Fine until someone relies on >NUL.
        return t != FILE_TYPE_PIPE && t != FILE_TYPE_DISK;
    };
    if ((no_handle(STD_OUTPUT_HANDLE) || no_handle(STD_ERROR_HANDLE)) && AttachConsole(ATTACH_PARENT_PROCESS)) {
        if (no_handle(STD_OUTPUT_HANDLE)) freopen("CONOUT$", "w", stdout);
        if (no_handle(STD_ERROR_HANDLE)) freopen("CONOUT$", "w", stderr);
    }
#endif
    const char *path = nullptr, *model = "vsr.onnx", *collect_dir = nullptr;
    bool vsr_on = true, selftest = false, checking = false, stats = false, fullscreen = false, want_rtx = true,
         cpu_decode = false, start_paused = false, model_given = false;
    int screen = 0;  // --screen N, counted from 1; 0 = the primary display
    double start_at = -1;  // --start T: seconds; < 0: from the beginning
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i], model_given = true;
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
        else if (argv[i][0] != '-') (selftest || checking ? (model_given = true, model) : path) = argv[i];
        else return fputs(USAGE, stderr), 2;
    }
    // saved settings, with this run's flags on top; only the settings card and the volume change `saved`
    Settings saved = load_settings(settings_file()), cfg = saved;
    if (!want_rtx && cfg.upscaler == Upscaler::Rtx) cfg.upscaler = Upscaler::PixelFlow;
    if (!vsr_on) cfg.upscaler = Upscaler::Off;
    cfg.cpu_decode |= cpu_decode;
    cfg.start_fullscreen |= fullscreen;
    vsr_on = cfg.upscaler != Upscaler::Off, want_rtx = cfg.upscaler == Upscaler::Rtx;
    cpu_decode = cfg.cpu_decode, fullscreen = cfg.start_fullscreen;
    // Where vsr.onnx is, when given by name only: the current folder (not for an installed copy: "Open with" makes
    // that the video's folder), the user's data folder (where an installed copy keeps your Pixel Flow, untouched by
    // updates), then next to the program (developer builds).
    std::string found_model;
    if (std::filesystem::path(model).is_relative() && ((installed_copy() && !model_given) || !std::filesystem::exists(model))) {
        std::vector<std::filesystem::path> where = {std::filesystem::u8path(user_data_dir()) / model};
        for (const char* up : {".", ".."}) where.push_back(std::filesystem::path(exe_path()).parent_path() / up / model);
        for (auto& m : where)
            if (std::error_code ec; std::filesystem::exists(m, ec)) {
                model = (found_model = m.lexically_normal().u8string()).c_str();
                break;
            }
        if (found_model.empty() && installed_copy() && !model_given)  // not the current folder's: where it belongs
            model = (found_model = where[0].u8string()).c_str();
    }
    // TensorRT, for Pixel Flow: loaded now from known folders only, so the player runs without it
    std::string trt_ver, trt_why;
    finish_pending_addon_removal();  // "Remove" while its DLLs were in use
    finish_pending_training_removal();
    TrtFrom trt_from = load_tensorrt(saved.trt_folder, trt_ver, trt_why);
    if (selftest)
        return trt_from == TrtFrom::None ? fprintf(stderr, "%s\n", trt_line(trt_from, trt_ver, trt_why).c_str()), 1
                                         : Vsr::selftest(model);
    if (checking) return check(model, trt_from, trt_ver, trt_why);
    // open-player.log, for playback only (a --check must not push the last crash's log out to .old.log): everything
    // below that goes to stderr is also saved, with the libraries' own messages
    bool logging = start_log(log_file());
    if (logging) {
        atexit(stop_log);
        fprintf(stderr, "Open Player %s\n", OPEN_PLAYER_BUILD);
    } else fprintf(stderr, "log: can't write %s, running without a log file\n", log_file().c_str());
    fprintf(stderr, "%s\n%s\n", trt_line(trt_from, trt_ver, trt_why).c_str(), driver_line().c_str());
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
        media.set_paused(true);  // nothing plays while the upscalers get ready (a TensorRT engine build takes a minute)
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
        // The plan uses Pixel Flow's factor, or 2x without a model (RTX VSR follows the same plan).
        int ms = worth_it ? Vsr::model_scale(model) : 0, nw = w, nh = h;
        if (worth_it && ms <= 0) ms = 2;
        if (ms > 0 && screen_h && (w * ms > screen_w || h * ms > screen_h)) {
            double k = std::fmin((double)screen_w / (w * ms), (double)screen_h / (h * ms));
            nw = (int)std::lround(w * k) & ~1, nh = (int)std::lround(h * k) & ~1;
            fprintf(stderr, "VSR: shrinking %dx%d to %dx%d first, so %dx lands on the %dx%d screen\n", w, h, nw, nh,
                    ms, screen_w, screen_h);
        }
        // Pixel Flow: needs your own trained model and TensorRT. Nobody gets a model with the player.
        Vsr vsr;
        bool model_there = std::filesystem::exists(std::filesystem::u8path(model));
        bool have_pxf = worth_it && model_there && trt_from != TrtFrom::None && vsr.open(model, nw, nh, st);
        bool shrink = have_pxf && (nw != w || nh != h);
        __half* full = nullptr;  // full-size RGB frame, when it gets shrunk before the network
        if (shrink) cudaMalloc(&full, 3ull * w * h * 2);
        if (!worth_it)
            fprintf(stderr, "VSR off: the video (%dx%d) already fills the screen (%dx%d)\n", w, h, screen_w, screen_h);
        else if (!have_pxf)
            fprintf(stderr, "Pixel Flow: %s\n", !model_there ? "no model (vsr.onnx): train your own to use it"
                                                : trt_from == TrtFrom::None ? "needs the TensorRT add-on"
                                                                           : "couldn't load the model");
        int s = have_pxf ? vsr.scale() : worth_it ? ms : 1;
        // NVIDIA RTX VSR goes first on Windows, and whenever Pixel Flow can't run; Pixel Flow when chosen, and on Linux.
#ifdef HAVE_RTX_VSR
        RtxVsr rtx;
        // otherwise opened on first use (settings card, V after --no-vsr)
        bool rtx_tried = worth_it && vsr_on && (want_rtx || !have_pxf);
        bool rtx_ok = rtx_tried && rtx.open(w, h, nw * s, nh * s, st);
        bool use_rtx = rtx_ok;
        fprintf(stderr, "%s\n", rtx_vsr_line(rtx_tried, rtx_ok).c_str());
        Collector col;
        bool collecting = collect_dir && use_rtx && col.open(collect_dir, w, h, nw * s, nh * s, s, st);
        if (collect_dir)
            fprintf(stderr, collecting ? "collect: saving RTX VSR training pairs in %s\n"
                                       : "collect: off (needs RTX VSR at exactly 2x - no shrinking to fit the screen -, "
                                         "a video of at least 128x128, and %s writable and under 2 GB)\n",
                    collect_dir);
#else
        bool use_rtx = false, rtx_ok = false, rtx_tried = false;
        (void)want_rtx;
        if (collect_dir) fprintf(stderr, "collect: needs NVIDIA RTX VSR, which this build doesn't have\n");
#endif
        bool have_vsr = have_pxf || rtx_ok;  // something can upscale (grows when RTX VSR opens later)
        vsr_on &= have_vsr;
        if (have_vsr) fprintf(stderr, "VSR: %s\n", !vsr_on ? "off" : use_rtx ? "NVIDIA RTX Video Super Resolution"
                                                                             : "Pixel Flow");
        __half* plain = nullptr;  // RGB frame when there is no Pixel Flow engine to borrow the input buffer from
        if (!have_pxf) cudaMalloc(&plain, 3ull * w * h * 2);
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
                   : !vsr_on && !have_pxf ? std::string("off")
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
            if (!worth_it) return false;
            std::string why;  // the chosen upscaler can't run here: say why, keep what can
            if (u == Upscaler::Rtx) {
#ifdef HAVE_RTX_VSR
                if (!rtx_tried) rtx_tried = true, rtx_ok = rtx.open(w, h, nw * s, nh * s, st);
#endif
                if (!rtx_ok) why = "NVIDIA RTX isn't available on this GPU or driver";
            } else if (u == Upscaler::PixelFlow && !have_pxf) {
                why = !model_there                ? "Pixel Flow needs your own trained model (vsr.onnx)"
                      : trt_from == TrtFrom::None ? "Pixel Flow needs the TensorRT add-on (Settings)"
                                                  : "Pixel Flow: the model couldn't be loaded (see the log)";
#ifdef HAVE_RTX_VSR
                if (!rtx_tried) rtx_tried = true, rtx_ok = rtx.open(w, h, nw * s, nh * s, st);  // the fallback
#endif
            }
            have_vsr = have_pxf || rtx_ok;
            if (!why.empty()) u = u == Upscaler::Rtx && have_pxf ? Upscaler::PixelFlow
                                 : u == Upscaler::PixelFlow && rtx_ok ? Upscaler::Rtx : Upscaler::Off;
            const bool ok = why.empty();
            if (u != Upscaler::Off) chosen = u;
            vsr_on = u != Upscaler::Off, use_rtx = u == Upscaler::Rtx;
            reset = true, step |= paused, drop_ahead();  // paused: show the change now
            if (paused && media.finished()) seek(pos);   // at the end no frame is queued: decode the last one again
            vsr_desc = describe_vsr();
            post(ICON_SPARKLES, !ok          ? why
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
        // the height of the screen the window is on, in window coordinates like ImGui's DisplaySize (a fullscreen
        // window is exactly this tall); 0 when SDL can't say
        auto window_screen_h = [&]() -> float {
            const SDL_DisplayMode* m = SDL_GetDesktopDisplayMode(SDL_GetDisplayForWindow(win));
            return m ? (float)m->h : 0.f;
        };
        auto toggle_fullscreen = [&] { SDL_SetWindowFullscreen(win, !(SDL_GetWindowFlags(win) & SDL_WINDOW_FULLSCREEN)); };
        // replaces this process with the player run with args (open another video, reopen after a settings change)
        double position_saved_at = now();
        TrainingStatus training;  // the Training tab's training.json, read once a second while it's open
        bool training_known = false;
        double training_read_at = -10;
        auto save_position = [&] {  // this video's spot, for "Remember where I left off"
            position_saved_at = now();
            if (!frames_shown) return;  // quit before the first frame: pos is still 0, keep the saved spot
            save_positions(positions_file(), positions_add(load_positions(positions_file()), video_path, pos));
        };
        std::function<void()> stop_downloads;  // set below, once the downloads exist
        auto restart_with = [&](std::vector<std::string> args, const std::string& video) {
            if (stop_downloads) stop_downloads();  // cancelled (partial files deleted), not cut off by the restart
            if (cfg.remember_position) save_position();
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
            int err = errno;
            if (logging) start_log(log_file(), true);  // still this run: carry on in the same log
            fprintf(stderr, "can't open %s: restart failed (%s)\n", video.c_str(), strerror(err));
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
        auto show_folder_dialog = [&] {  // your own TensorRT 11.x; arrives as SDL_EVENT_USER with code 1
            SDL_ShowOpenFolderDialog(
                [](void*, const char* const* dirs, int) {
                    if (!dirs || !dirs[0]) return;
                    SDL_Event ev{SDL_EVENT_USER};
                    ev.user.code = 1, ev.user.data1 = SDL_strdup(dirs[0]);
                    SDL_PushEvent(&ev);
                },
                nullptr, win, nullptr, false);
        };
        // the TensorRT add-on download runs on its own thread; the card shows its progress
        std::thread addon_thread;
        std::atomic<bool> addon_running{false}, addon_done{false}, addon_cancel{false};
        std::atomic<uint64_t> addon_got{0};
        bool addon_ok = false;
        std::string addon_error;
        auto stop_addon = [&] {
            if (addon_thread.joinable()) addon_cancel = true, addon_thread.join();
        };
        // Training from the player: the Training add-on's download (then the start it was for), the export, and a
        // count of the videos in the chosen folder each run on a thread of their own
        std::thread tr_thread, ex_thread, count_thread;
        std::atomic<bool> tr_running{false}, tr_done{false}, tr_cancel{false}, ex_running{false}, ex_done{false};
        std::atomic<uint64_t> tr_got{0};
        std::atomic<int> video_count{-1};
        std::atomic<bool> count_cancel{false};
        bool tr_ok = false, ex_ok = false, tr_alive = false, tr_current = false;
        std::string tr_error, ex_error, training_error;
        StartMode tr_mode = StartMode::New;
        int tr_addon = training_addon_installed();
        bool tr_ckpt = has_checkpoint(), tr_best = has_best();
        const bool gpu_ok = [] {
            int n = 0;
            return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
        }();
        auto count_videos = [&](const std::string& dir) {
            if (count_thread.joinable()) count_cancel = true, count_thread.join();
            video_count = -1, count_cancel = false;
            count_thread = std::thread([&video_count, &count_cancel, dir] {
                static const char* exts[] = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v", ".ts"};  // train.py's
                int n = 0;
                std::error_code ec;
                for (auto it = std::filesystem::recursive_directory_iterator(
                         std::filesystem::u8path(dir), std::filesystem::directory_options::skip_permission_denied, ec);
                     !ec && !count_cancel && it != std::filesystem::recursive_directory_iterator(); it.increment(ec)) {
                    std::string x = it->path().extension().string();
                    for (auto& c : x) c = (char)tolower((unsigned char)c);
                    for (const char* e : exts) n += x == e;
                }
                if (!count_cancel) video_count = n;
            });
        };
        if (!cfg.train_videos.empty()) count_videos(cfg.train_videos);

        auto start_own_training = [&](StartMode m) {
            std::string err;
            if (start_training(cfg.train_videos, m, err)) training_error.clear(), training_read_at = -10;
            else training_error = "Couldn't start training: " + err;
        };
        auto show_train_folder_dialog = [&] {  // the videos to train on; arrives as SDL_EVENT_USER with code 2
            SDL_ShowOpenFolderDialog(
                [](void*, const char* const* dirs, int) {
                    if (!dirs || !dirs[0]) return;
                    SDL_Event ev{SDL_EVENT_USER};
                    ev.user.code = 2, ev.user.data1 = SDL_strdup(dirs[0]);
                    SDL_PushEvent(&ev);
                },
                nullptr, win, nullptr, false);
        };
        auto stop_training_threads = [&] {
            if (tr_thread.joinable()) tr_cancel = true, tr_thread.join();
            if (ex_thread.joinable()) ex_thread.join();  // an export takes seconds; its model would be half written
            if (count_thread.joinable()) count_cancel = true, count_thread.join();
        };
        SDL_SetWindowTitle(win, std::filesystem::path(path).filename().string().c_str());
        if (start_at >= 0) seek(start_position(start_at, dur));  // reopened after a settings change
        else if (cfg.remember_position) {  // where you left off last time
            double t = resume_point(load_positions(positions_file()), video_path, dur);
            if (t >= 0) seek(t), post(ICON_CLOCK, "Resumed at " + clock_text(t), -1, 2);
        }
        if (start_paused) set_paused(true), step = true;         // ... while paused: show that frame, stay paused
        else media.set_paused(false), wall_t0 = SDL_GetTicksNS();  // ready: play from the start, not from mid-setup

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
        // Captions: the file's English tracks; on: the chosen one (the default: SDH first), off: a forced one only
        const std::vector<CaptionTrack>& cap_tracks = media.caption_tracks();
        std::vector<std::string> cap_names;
        for (auto& t : cap_tracks) cap_names.push_back(t.name);
        int cap_track = -1, cue_seen = -1;
        bool cap_forced = false;
        std::vector<Cue> cue_copy;  // the UI's copy, refreshed when Media's cues change
        auto apply_captions = [&](int pick) {  // pick < 0: the default track
            if (!cfg.captions) {
                const int f = forced_track(cap_tracks);
                cap_track = -1, cap_forced = f >= 0;
                media.set_captions(-1, f);
                return;
            }
            cap_track = pick >= 0 && pick < (int)cap_tracks.size() ? pick : default_track(cap_tracks), cap_forced = false;
            media.set_captions(cap_track, -1);
            if (cap_track >= 0) fprintf(stderr, "captions: %s\n", cap_tracks[cap_track].name.c_str());
        };
        auto set_captions_on = [&](bool on, int pick) {
            saved.captions = cfg.captions = on, save_prefs();  // remembered even for a file without captions
            if (cap_tracks.empty()) return post(ICON_BADGE_CC, on ? "Captions on (this video has no English ones)" : "Captions off", -1, 2);
            apply_captions(pick);
            post(ICON_BADGE_CC, cap_track >= 0 ? "Captions: " + cap_tracks[cap_track].name : std::string("Captions off"), -1, 2);
        };
        apply_captions(-1);

        auto toggle_settings = [&] {
            settings_open = !settings_open, dirty = true;
            if (settings_open) monitors = list_screens(&screen_ids);
        };

        // Updates, installed copies only: at most one check a day on its own thread (state shared, so the thread may
        // outlive a quick exit), then "Update now" in About downloads the signed installer and runs it.
        auto upd = std::make_shared<UpdateCheck>();
        std::thread upd_check, upd_dl;
        std::atomic<bool> upd_running{false}, upd_done{false}, upd_cancel{false};
        std::atomic<uint64_t> upd_got{0}, upd_total{0};
        std::string upd_path, upd_error;
        bool upd_ok = false, upd_checking = false;
        const int my_build = build_number(OPEN_PLAYER_BUILD);
        if (installed_copy() && my_build > 0 &&
            due_for_check((long long)time(nullptr), saved.last_update_check, saved.check_updates)) {
            upd_checking = true;
            upd_check = std::thread([upd, my_build] { check_for_update(*upd, my_build); });
        }
        stop_downloads = [&] {
            stop_addon();
            stop_training_threads();
            if (upd_dl.joinable()) upd_cancel = true, upd_dl.join();
        };

        while (running) {
            if (upd_checking && upd->done) {
                upd_check.join(), upd_checking = false;
                // a failed check (offline, rate limit) is tried again at the next start, not a day later
                if (upd->error.empty()) saved.last_update_check = (long long)time(nullptr), save_prefs();
                if (upd->available)
                    post(ICON_SPARKLES, "Update available: build " + std::to_string(upd->info.build) + " (Settings)", -1, 4);
                else if (!upd->error.empty()) fprintf(stderr, "update check: %s\n", upd->error.c_str());
            }
            if (upd_done) {  // the installer is downloaded and checked: run it, the player closes
                upd_dl.join(), upd_running = false, upd_done = false;
                if (!upd_ok) post(ICON_X, "Update failed: " + upd_error, -1, 4);
                else if (save_prefs(), fprintf(stderr, "updating to build %d\n", upd->info.build), run_installer(upd_path, upd->info.sha256))
                    running = false;
                else post(ICON_X, "Update failed: couldn't start the installer", -1, 4);
            }
            if (upd_running && settings_open) dirty = true;
            if (cfg.remember_position && !paused && now() - position_saved_at >= 10) save_position();
            if (volume_unsaved && now() - volume_saved_at >= 1)
                save_settings(settings_file(), saved), volume_unsaved = false, volume_saved_at = now();
            if (addon_done) {  // the add-on download finished: load it by reopening, or say what went wrong
                addon_thread.join(), addon_running = false, addon_done = false;
                if (addon_ok) reopen("TensorRT installed: reopening...", true);
                else if (!addon_cancel) post(ICON_X, "TensorRT: " + addon_error, -1, 4);
            }
            if (addon_running && settings_open) dirty = true;  // the progress bar moves
            if (tr_done) {  // the Training add-on arrived (or didn't): start the run it was downloaded for
                tr_thread.join(), tr_running = false, tr_done = false, tr_addon = training_addon_installed();
                if (tr_ok) start_own_training(tr_mode);
                else if (!tr_cancel) training_error = "Training add-on: " + tr_error;
            }
            if (tr_running && settings_open) dirty = true;
            if (ex_done) {  // the export finished: load the new Pixel Flow by reopening
                ex_thread.join(), ex_running = false, ex_done = false;
                if (!ex_ok) training_error = ex_error;
                else if (trt_from == TrtFrom::None)
                    post(ICON_SPARKLES, "Pixel Flow is ready; it also needs TensorRT: Settings > Upscaling > Download", -1, 6);
                else reopen("Loading your Pixel Flow...", true);
            }
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
                else if (e.type == SDL_EVENT_USER) {  // a file picked in the open dialog, or (code 1) a TensorRT folder
                    std::string p = (char*)e.user.data1;
                    SDL_free(e.user.data1);
                    std::error_code ec;
                    auto f = std::filesystem::u8path(p);
                    // NVIDIA's zip keeps the DLLs in bin\, so picking the unzipped folder itself works too
                    if (e.user.code == 1 && !std::filesystem::exists(f / "nvinfer_11.dll", ec) &&
                        std::filesystem::exists(f / "bin" / "nvinfer_11.dll", ec))
                        p = (f / "bin").u8string();
                    if (e.user.code == 2) {
                        saved.train_videos = cfg.train_videos = p, save_prefs(), count_videos(p);
                        continue;
                    }
                    if (e.user.code != 1) open_video(p);
                    else if (!std::filesystem::exists(std::filesystem::u8path(p) / "nvinfer_11.dll", ec))
                        post(ICON_X, "No TensorRT 11 in that folder (nvinfer_11.dll)", -1, 3);
                    else saved.trt_folder = cfg.trt_folder = p, save_prefs(), reopen("Loading your TensorRT...", true);
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
                    case SDLK_C: set_captions_on(!cfg.captions, -1), dirty = true; break;
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
                    __half* in = shrink ? full : have_pxf ? vsr.input() : plain;
                    Uint64 u0 = SDL_GetTicksNS();
                    cudaEventRecord(p.ev[0], st);
                    YuvFrame yuv = media.to_gpu(f, st);
                    diag.upload.add((SDL_GetTicksNS() - u0) / 1e6);
                    yuv_to_rgb(yuv, in, st);
                    if (vsr_on && shrink && !use_rtx) shrink_rgb(full, w, h, vsr.input(), nw, nh, st);
                    cudaEventRecord(p.ev[1], st);
#ifdef HAVE_RTX_VSR
                    if (vsr_on && use_rtx && !(p.out = rtx.run(in))) {  // full-size input
                        use_rtx = rtx_ok = false;  // NVIDIA's upscaler failed: Pixel Flow from here on, if any
                        chosen = Upscaler::PixelFlow;
                        have_vsr = vsr_on = have_pxf;
                        fprintf(stderr, have_pxf ? "VSR: NVIDIA RTX failed, switching to Pixel Flow\n"
                                                 : "VSR: NVIDIA RTX failed, playing without upscaling\n");
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
                           ui_scale(ImGui::GetIO().DisplaySize.x, ImGui::GetIO().DisplaySize.y, window_screen_h()),
                           notice, s, use_rtx};
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
                us.rtx_available = worth_it && (!rtx_tried || rtx_ok);  // not tried yet: offered, opened on first use
#endif
                us.vsr_available = have_vsr || us.rtx_available;
                us.pxf_available = have_pxf;
                us.pxf_why = !model_there ? "No Pixel Flow model: train your own (vsr.onnx)"
                             : trt_from == TrtFrom::None ? "Needs the TensorRT add-on"
                                                         : "The model couldn't be loaded (see the log)";
                us.nvdec_available = media.nvdec() || cfg.cpu_decode;  // CPU chosen: NVDEC wasn't even tried
                if (!worth_it) us.vsr_unavailable = "the video already fills the screen";
                else if (!have_pxf && !us.rtx_available)
                    us.vsr_unavailable = "NVIDIA RTX VSR isn't available here, and " + us.pxf_why;
                us.build = OPEN_PLAYER_BUILD;
                us.versions = {trt_line(trt_from, trt_ver, trt_why), rtx_vsr_line(rtx_tried, rtx_ok), driver_line()};
                us.trt_addon = trt_from == TrtFrom::AddOn;
                if (now() - training_read_at >= 1) {  // training.json, once a second
                    training_read_at = now();
                    tr_ckpt = has_checkpoint(), tr_best = has_best();
                    std::ifstream tf(std::filesystem::u8path(training_file()), std::ios::binary);
                    std::string text(1 << 20, '\0');
                    tf.read(text.data(), (std::streamsize)text.size());
                    text.resize((size_t)tf.gcount());
                    training_known = parse_training_status(text, training);
                    // the player's own run, from run.txt (any player that started it, before a restart too)
                    const bool own = training_known && own_run(training);
                    RunInfo run;
                    const bool have_run = read_run(run);
                    tr_alive = have_run && process_alive(run.pid, training_python());
                    tr_current = have_run && own && status_current(training, run.started);
                    if (have_run && !tr_alive) {  // it has ended: how?
                        if (tr_current && training.state == "finished" && !ex_running) {  // export it: Pixel Flow just works
                            ex_running = true, ex_done = false;
                            ex_thread = std::thread([&] { ex_ok = export_and_install(ex_error), ex_done = true; });
                        } else if (!(tr_current && (training.state == "stopped" || training.state == "failed" ||
                                                     training.state == "finished")))
                            training_error = "Training stopped unexpectedly; see " + train_log();
                        clear_run();
                    }
                    if (settings_open && cfg.settings_tab == 1)
                        ui_redraw = std::max(ui_redraw, 1);  // paused, nothing else redraws: the tab must still update
                }
                us.training = training, us.training_known = training_known, us.now_unix = (long long)time(nullptr);
                if (media.cues_version() != cue_seen)  // new cues while paused must still show
                    cue_seen = media.cues_version(), cue_copy = media.cues(), ui_redraw = std::max(ui_redraw, 1);
                us.cues = &cue_copy, us.cue_pos = pos, us.video_w = w, us.video_h = h;
                us.caption_names = cap_names, us.caption_track = cap_track, us.caption_forced = cap_forced;
                us.train_video_count = video_count, us.training_addon = tr_addon, us.training_error = training_error;
                us.training_addon_progress = tr_running ? (float)((double)tr_got / training_addon_size()) : -1;
                us.training_own = training_known && own_run(training), us.training_log = train_log();
                us.has_checkpoint = tr_ckpt, us.has_best = tr_best, us.exporting = ex_running, us.gpu_ok = gpu_ok;
                us.training_alive = tr_alive, us.training_current = tr_current;
                if (upd_checking) us.update_status = "Checking for updates...";
                else if (upd_running) {
                    us.update_status = "Downloading build " + std::to_string(upd->info.build) + "...";
                    us.update_progress = upd_total ? (float)((double)upd_got / upd_total) : 0;
                } else if (upd->available)
                    us.update_status = "Update available: build " + std::to_string(upd->info.build),
                    us.update_build = upd->info.build;
                else if (!installed_copy()) us.update_status = "Updates: only for installed copies";
                else if (upd->done && !upd->error.empty()) us.update_status = "Last check: " + upd->error;
                us.trt_status = trt_line(trt_from, trt_ver, trt_why);
                if (addon_running) {
                    us.addon_progress = (float)((double)addon_got / TRT_WHEEL_SIZE);
                    us.trt_status = addon_got >= TRT_WHEEL_SIZE  // then the 2 GB hash and the unpacking
                                        ? std::string("Checking and unpacking TensorRT...")
                                        : "Downloading TensorRT from NVIDIA: " + std::to_string(addon_got / (1 << 20)) +
                                              " of " + std::to_string(TRT_WHEEL_SIZE >> 20) + " MB";
                }
                if (logging) us.log_path = log_file();
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
                    case UiAction::DownloadTensorRT:
                        if (addon_running) break;
                        stop_addon();
                        addon_running = true, addon_done = false, addon_cancel = false, addon_got = 0;
                        addon_thread = std::thread([&] {
                            addon_ok = install_tensorrt_addon([&](uint64_t got, uint64_t) { addon_got = got; },
                                                              addon_cancel, addon_error);
                            addon_done = true;
                        });
                        break;
                    case UiAction::CancelTensorRT: addon_cancel = true; break;
                    case UiAction::ChooseTensorRTFolder: show_folder_dialog(); break;
                    case UiAction::SetCheckUpdates: saved.check_updates = cfg.check_updates = a.value != 0, save_prefs(); break;
                    case UiAction::SetSettingsTab: saved.settings_tab = cfg.settings_tab = (int)a.value, save_prefs(); break;
                    case UiAction::ToggleCaptions: set_captions_on(!cfg.captions, -1); break;
                    case UiAction::SetCaptionTrack: set_captions_on(a.value >= 0, (int)a.value); break;
                    case UiAction::SetRememberPosition:
                        saved.remember_position = cfg.remember_position = a.value != 0, save_prefs();
                        break;
                    case UiAction::InstallUpdate:
                        if (upd_running || !upd->available) break;
                        upd_running = true, upd_done = false, upd_cancel = false, upd_got = 0;
                        upd_dl = std::thread([&] {
                            upd_ok = download_update(*upd, [&](uint64_t g, uint64_t t) { upd_got = g, upd_total = t; },
                                                     upd_cancel, upd_path, upd_error);
                            upd_done = true;
                        });
                        break;
                    case UiAction::ChooseTrainVideos: show_train_folder_dialog(); break;
                    case UiAction::StartTraining: {
                        if (tr_running || tr_alive || cfg.train_videos.empty()) break;
                        tr_mode = has_checkpoint() ? StartMode::Resume : StartMode::New;
                        if (training_addon_installed() == 1) {
                            start_own_training(tr_mode);
                            break;
                        }
                        tr_running = true, tr_done = false, tr_cancel = false, tr_got = 0, training_error.clear();
                        tr_thread = std::thread([&] {
                            tr_ok = install_training_addon([&](uint64_t got, uint64_t) { tr_got = got; }, tr_cancel,
                                                           tr_error);
                            tr_done = true;
                        });
                        break;
                    }
                    case UiAction::CancelTrainingAddon: tr_cancel = true; break;
                    case UiAction::StopTraining: request_stop(); break;
                    case UiAction::UseModel:
                        if (ex_running) break;
                        ex_running = true, ex_done = false, training_error.clear();
                        ex_thread = std::thread([&] { ex_ok = export_and_install(ex_error), ex_done = true; });
                        break;
                    case UiAction::StartOver: {
                        const SDL_MessageBoxButtonData btns[] = {
                            {SDL_MESSAGEBOX_BUTTON_RETURNKEY_DEFAULT, 1, "Start over"},
                            {SDL_MESSAGEBOX_BUTTON_ESCAPEKEY_DEFAULT, 0, "Cancel"}};
                        const SDL_MessageBoxData box{SDL_MESSAGEBOX_WARNING, win, "Start over?",
                                                     "This deletes the model trained so far (the Pixel Flow you're "
                                                     "using stays). The next Start trains a new one from zero.",
                                                     2, btns, nullptr};
                        int pick = 0;
                        std::string err;
                        if (SDL_ShowMessageBox(&box, &pick) && pick == 1 && !start_over(err)) training_error = err;
                        break;
                    }
                    case UiAction::RemoveTrainingAddon:
                        remove_training_addon(training_alive() || ex_running);
                        tr_addon = training_addon_installed();
                        break;
                    case UiAction::RemoveTensorRT:
                        remove_tensorrt_addon(trt_from == TrtFrom::AddOn);  // in use: removed when the player reopens
                        if (saved.trt_folder.empty()) reopen("Removing TensorRT...", true);
                        break;
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
                           trt_from == TrtFrom::None ? "none" : trt_ver.c_str());
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
        if (cfg.remember_position) save_position();
        stop_addon();  // a download in progress is cancelled (its partial file is deleted)
        stop_training_threads();  // the training run itself keeps going: it's a process of its own
        if (upd_dl.joinable()) upd_cancel = true, upd_dl.join();
        if (upd_check.joinable()) upd_check.detach();  // owns its shared state; may still be waiting on the network
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
