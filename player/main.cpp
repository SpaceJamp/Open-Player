// vsrplayer: video player that upscales with our VSR network on the GPU.
// decode (NVDEC) -> YUV->RGB (CUDA) -> VSR (TensorRT) -> RGBA (CUDA) -> Vulkan blit to the window
#include <SDL3/SDL.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

#include "display.h"
#include "kernels.h"
#include "media.h"
#include "vsr.h"

static const char* USAGE =
    "usage: vsrplayer <video> [--model vsr.onnx] [--no-vsr]\n"
    "       vsrplayer --selftest [vsr.onnx]\n"
    "keys:  space pause | left/right seek 5s | up/down volume | v VSR on/off | f or double-click fullscreen\n"
    "       click the bar at the bottom to seek | q quit\n";

static std::string clock_text(double t) {
    int s = (int)std::fmax(t, 0);
    char buf[32];
    if (s >= 3600) snprintf(buf, sizeof buf, "%d:%02d:%02d", s / 3600, s / 60 % 60, s % 60);
    else snprintf(buf, sizeof buf, "%02d:%02d", s / 60, s % 60);
    return buf;
}

int main(int argc, char** argv) {
    const char *path = nullptr, *model = "vsr.onnx";
    bool vsr_on = true, selftest = false;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "--no-vsr")) vsr_on = false;
        else if (!strcmp(argv[i], "--selftest")) selftest = true;
        else if (argv[i][0] != '-') (selftest ? model : path) = argv[i];
        else return fputs(USAGE, stderr), 2;
    }
    if (selftest) return Vsr::selftest(model);
    if (!path) return fputs(USAGE, stderr), 2;

    if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO)) return fprintf(stderr, "SDL: %s\n", SDL_GetError()), 1;
    // hidden until we know the video size; Vulkan needs the window first to choose the GPU
    SDL_Window* win = SDL_CreateWindow("vsrplayer", 1280, 720,
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
        Vsr vsr;
        bool have_vsr = vsr.open(model, w, h, st);
        if (!have_vsr) fprintf(stderr, "VSR unavailable (no %s?), playing without upscaling\n", model);
        vsr_on &= have_vsr;
        int s = have_vsr ? vsr.scale() : 1;
        __half* plain = nullptr;  // RGB frame when there is no VSR engine to borrow the input buffer from
        if (!have_vsr) cudaMalloc(&plain, 3ull * w * h * 2);
        if (!disp.alloc(w * s, h * s)) return fprintf(stderr, "display: out of GPU memory\n"), 1;

        SDL_Rect area;
        SDL_GetDisplayUsableBounds(SDL_GetDisplayForWindow(win), &area);
        float fit = std::fmin(1.f, std::fmin(0.9f * area.w / w, 0.9f * area.h / h));
        SDL_SetWindowSize(win, (int)(w * fit), (int)(h * fit));
        SDL_SetWindowPosition(win, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED);
        SDL_ShowWindow(win);

        const double dur = media.duration();
        bool running = true, paused = false, reset = true, step = false;
        int serial = media.serial(), sw = 0, sh = 0;
        float volume = 1;
        const __half* shown = nullptr;
        double pos = 0, wall_base = 0, last_mouse = -10, last_title = -10;
        Uint64 wall_t0 = SDL_GetTicksNS();
        auto now = [] { return SDL_GetTicksNS() / 1e9; };
        // audio is the clock when there is audio; otherwise wall time
        auto clock = [&] {
            if (media.has_audio()) return media.audio_clock();
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
                else if (e.type == SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED) disp.resized();
                else if (e.type == SDL_EVENT_MOUSE_MOTION) last_mouse = now();
                else if (e.type == SDL_EVENT_MOUSE_BUTTON_DOWN && e.button.button == SDL_BUTTON_LEFT) {
                    int ww, wh;
                    SDL_GetWindowSize(win, &ww, &wh);
                    if (e.button.clicks == 2) SDL_SetWindowFullscreen(win, !(SDL_GetWindowFlags(win) & SDL_WINDOW_FULLSCREEN));
                    else if (dur > 0 && e.button.y > wh * 0.92f) seek(e.button.x / ww * dur);
                } else if (e.type == SDL_EVENT_KEY_DOWN) {
                    bool full = SDL_GetWindowFlags(win) & SDL_WINDOW_FULLSCREEN;
                    switch (e.key.key) {
                    case SDLK_SPACE: set_paused(!paused); break;
                    case SDLK_LEFT: seek(pos - 5); break;
                    case SDLK_RIGHT: seek(pos + 5); break;
                    case SDLK_UP: media.set_volume(volume = std::fmin(volume + 0.1f, 2.f)); break;
                    case SDLK_DOWN: media.set_volume(volume = std::fmax(volume - 0.1f, 0.f)); break;
                    case SDLK_V: if (have_vsr) vsr_on = !vsr_on, reset = true; break;
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
                __half* in = have_vsr ? vsr.input() : plain;
                yuv_to_rgb(media.to_gpu(f, st), in, st);
                if (vsr_on) shown = vsr.run(reset), sw = w * s, sh = h * s, reset = false;
                else shown = in, sw = w, sh = h;
                cudaStreamSynchronize(st);  // done reading the decoder surface: hand it back to FFmpeg
                av_frame_free(&f);
                pos = pts, step = false;
            } else if (!paused && media.finished()) {
                set_paused(true);  // stay on the last frame, like a normal player
            }

            bool bar = dur > 0 && (paused || now() - last_mouse < 2), drawn = false;
            if (shown) {  // redrawn every vsync so the seek bar and resizes update while paused
                rgb_to_rgba8(shown, sw, sh, disp.begin_frame(), bar ? pos / dur : -1, st);
                drawn = disp.present(sw, sh, st);
            }
            if (!drawn) SDL_Delay(5);  // nothing to show yet, or minimized; otherwise vsync paces the loop

            if (now() - last_title > 0.5) {
                last_title = now();
                std::string t = std::string(paused ? "|| " : "") + clock_text(pos) + " / " + clock_text(dur) +
                                (have_vsr ? (vsr_on ? "   VSR " + std::to_string(s) + "x" : "   VSR off") : "") +
                                "   vol " + std::to_string((int)std::lround(volume * 100)) + "%   " + path;
                SDL_SetWindowTitle(win, t.c_str());
            }
        }
        cudaFree(plain);
        rc = 0;
    }
    SDL_DestroyWindow(win);
    SDL_Quit();
    return rc;
}
