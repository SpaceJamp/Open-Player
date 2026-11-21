<p align="center"><img src="assets/icon-256.png" width="128" alt="Open Player logo"></p>

# Open Player

An open video player for Linux and Windows that upscales video on NVIDIA GPUs. On Windows it uses NVIDIA RTX
Video Super Resolution; everywhere it can use its own video super-resolution network, which is trained, exported and
played entirely from this repo.

```
decode (FFmpeg, NVDEC) -> YUV->RGB (CUDA) -> upscale: RTX VSR or our network (TensorRT) -> RGBA (CUDA) -> Vulkan -> window
```

**The network** ([model.py](model.py)) works on one frame at a time and remembers the previous one: a small flow
net estimates motion, the previous upscaled frame is warped onto the current one, and conv blocks (with pixel
attention, except in `lean`) add detail before a pixel-shuffle 2x output. A scene-cut check resets the memory.
Three sizes: `standard` (about 317K parameters), `fast` and `lean`.

No trained model ships with the repo: train one on your own videos (step 2).

## 1. Setup

**Linux** (Arch, Fedora 44, Ubuntu 26.04; native or WSL2), with the NVIDIA driver already installed:

```bash
bash setup.sh
python3 -m venv ~/vsr-venv && ~/vsr-venv/bin/pip install torch numpy onnx onnxscript --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple
```

[setup.sh](setup.sh) installs the build tools, FFmpeg, SDL3, Vulkan and CUDA; on Arch and Fedora TensorRT comes
from NVIDIA's tarball unpacked to `/opt/tensorrt`. On a fresh Arch WSL, first run
`sudo pacman-key --init && sudo pacman-key --populate archlinux`, or downloads fail with "signature is unknown trust".

**Windows** needs:
- Visual Studio 2022 or newer with the C++ tools (the Build Tools are enough), CUDA 13, and the Vulkan SDK
- [vcpkg](https://github.com/microsoft/vcpkg) cloned and bootstrapped in `C:\vcpkg`
- TensorRT for Windows unzipped to `C:\TensorRT\TensorRT-11.5.0.63` (other versions: pass `-DTENSORRT_ROOT=...`)
- for RTX VSR: the [NVIDIA RTX Video SDK](https://developer.nvidia.com/rtx-video-sdk) unzipped to `C:\rtx_video_sdk`
  (download it yourself: its license keeps it out of this repo; without it the player builds with our network only)
- for training: Python with `py -m pip install torch numpy onnx onnxscript --index-url https://download.pytorch.org/whl/cu130`,
  and FFmpeg on the PATH (`winget install Gyan.FFmpeg`)

## 2. Train a model

Put videos in `videos/` next to `train.py` (subfolders are fine), or pass their folder:

```bash
python train.py --out vsr.pt                              # or: python train.py "D:\Videos" --out vsr.pt
python train.py --out vsr.pt --resume vsr.pt              # continue after Ctrl+C (saved every 2000 steps and on Ctrl+C)
python export.py vsr_best.pt                              # -> vsr_best.onnx + .ref; rename to vsr.onnx for the player
```

(On Linux use `~/vsr-venv/bin/python`, on Windows `py`.) Training runs natively on Windows too.

How it works, and what helps:
- **Input:** frames are shrunk and then given random blur, grain and JPEG-style block compression, so the network
  learns to clean up real compressed video instead of sharpening its artifacts.
- **Validation:** every 1000 steps it prints `validation +X dB vs bilinear`, scored on 32 clips from two videos held
  out of training. They are saved to `<out>.val.pt` and reused, so scores stay comparable when you add videos. The
  best model so far is kept as `<out>_best.pt`.
- **Diagnostics:** after each validation a `suggest: ...` line names what to improve (data loading too slow,
  plateau, overfitting, unstable gradients, a weak clip, flicker, soft edges, or `ok`). Timings and metrics go to
  `<out>.diag.jsonl`; `python train_diag.py <out>.diag.jsonl` prints a summary.
- **Sampling:** each title (a top-level video file or folder, such as a whole series) is picked equally often.
- **Data sets the ceiling:** the network can't get sharper than its targets. High-bitrate sources (Blu-ray quality,
  4K shrunk to 1080p) beat 2-3 Mbit/s web rips, and varied content (animation, bright scenes, sports, film grain)
  generalises better than one genre.

Options: `--size standard|fast|lean`, `--steps`, `--lr`, `--batch`, `--crop`, `--frames`, `--workers`,
`--fft-loss W` (frequency loss for sharper edges; try 0.05-0.3), `--teacher ck.pt --distill W` (a smaller model
learns to copy a bigger one), `--pairs DIR --pairs-weight W` (learn from RTX VSR, see Windows below). A resumed run
keeps its steps, learning rate and loss settings.

`python tests/train_check.py` checks the degradations, losses, data and diagnostics in seconds.

## 3. Build and play

```bash
cmake -B build -S player && cmake --build build -j       # Linux
build-windows.bat                                         # Windows; the first run builds SDL3 and FFmpeg (about 30 min)
```

```bash
open-player movie.mkv                    # finds vsr.onnx in the current folder or next to the program
open-player --check vsr.onnx             # PASS/WARN/FAIL for CUDA, NVDEC, Vulkan, audio, TensorRT, RTX VSR, the model
open-player --selftest vsr.onnx          # TensorRT output vs PyTorch
```

Options: `--model FILE`, `--no-vsr`, `--no-rtx` (Windows: our network instead of RTX VSR), `--screen N`,
`--fullscreen`, `--stats` (one JSON line of timings per second), `--collect DIR` (see Windows below).

The first time our network plays a resolution, TensorRT builds an engine for it (up to a minute) and caches it next
to the model (`.win.engine` on Windows, so a shared folder works for both).

**How much it upscales.** The network upscales 2x and the window stretches the result to fit. It's planned for the
screen given with `--screen N` (default: the primary; `--check` lists them):
- video at least as large as the screen in either direction: no upscaling (a 4K video on a 4K screen plays as is);
- video whose 2x would overshoot the screen: shrunk first so 2x lands exactly on it (on a 4K screen, 1440p becomes
  1080p -> 2160p);
- smaller video: 2x, then stretched to fit.

**Speed** on an RTX 5080, 1080p -> 4K, from `--stats`:

| Network size | Time per frame | Plays smoothly |
|---|---|---|
| `standard` | 23-25 ms on an idle GPU (45-60 ms measured while the GPU was busy) | up to 30 fps |
| `fast` | 9-13 ms | about 60 fps |

The player prepares the next frame's upscale while the current one is shown.

### Controls

Move the mouse to show the floating control bar: play/pause, time, seek bar, volume (hover the speaker for a slider),
upscaling badge, open and fullscreen. It fades out 2 s after the mouse stops (the cursor hides in fullscreen) and
stays while paused. Volume, seek keys, pause and upscaling changes show a short notice. Right-click for a menu with
the same actions, recent files and the info overlay. Drop a video on the window, or use Open, to switch videos.

| Key | Action |
|---|---|
| Space | pause; at the end, play again |
| ← / → | seek 5 s |
| ↑ / ↓ | volume |
| V | upscaling on/off (compare) |
| F, double-click | fullscreen |
| I | info overlay: GPU, decoder, upscaler, frame rate, dropped frames |
| Q, Esc | quit (Esc leaves fullscreen first) |

### Windows and NVIDIA RTX VSR

On Windows the player upscales with **NVIDIA RTX Video Super Resolution** when the GPU and driver support it, and
falls back to our network otherwise (`--no-rtx` forces our network). CUDA and Vulkan share memory there, so frames
never pass through the CPU.

RTX VSR runs only on Windows, but our network can learn to imitate it, and then upscales that way on Linux too:
1. Play videos with `--collect pairs`. About every 10 s of playback it saves 8 consecutive frames of a 128 px crop
   next to RTX VSR's 2x upscale of the same region (only while RTX upscales exactly 2x; the folder stops at 2 GB).
2. Train with `--pairs pairs` (weight `--pairs-weight`, default 0.5), next to your usual videos.

This software uses the NVIDIA RTX Video SDK. NVIDIA, GeForce RTX and NVIDIA RTX are trademarks of NVIDIA Corporation.

## Control panel

`python panel.py` gives one window for the workflow: train with live progress and the dB gain, export with the
self-test, play a video, and watch live diagnostics (fps, dropped frames, per-stage timings, the bottleneck), plus a
system check and a copyable report. It remembers your choices in `panel.json`.

## Tests

```bash
bash tests/smoke.sh vsr.onnx             # plays generated clips, runs the model and UI self-tests
bash tests/codecs.sh                     # every format FFmpeg can write, from H.264 to ProRes and 7.1 audio
```

On Linux both run offscreen and silent (prefix `SDL_VIDEO_DRIVER=wayland SDL_AUDIO_DRIVER=pipewire` to watch). On
Windows run them from Git Bash with `OPEN_PLAYER=build-win/open-player.exe`; NVIDIA's Windows driver can't render
offscreen, so the test windows open on the second monitor (`OPEN_PLAYER_SCREEN=N` to change, the primary if
there's only one).

[upscale.py](upscale.py) upscales a file without the player: `python upscale.py in.mp4 -o out.mp4`.

## Limits

- SDR only: HDR video plays with washed-out colours.
- Interlaced video (DVDs, TV recordings) isn't deinterlaced, so motion shows comb lines.
- Audio-only files don't open, and there's no switching between audio or subtitle tracks yet.
- On WSL, CUDA and Vulkan can't share memory, so each frame goes through the CPU; native Linux and Windows are
  zero-copy. WSLg also tears; native systems don't.
- Opening another video restarts the player process; volume resets.

## License

Open Player is licensed under the [GNU Affero General Public License v3.0](LICENSE), with an
[additional permission](LICENSE-EXCEPTION) to link NVIDIA's CUDA, TensorRT and RTX Video SDK libraries. Bundled
third-party code and fonts keep their own licenses: Dear ImGui (MIT, `player/third_party/imgui/LICENSE.txt`),
Inter (SIL OFL) and Tabler Icons (MIT, both in `assets/fonts/`).
