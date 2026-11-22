<p align="center"><img src="assets/icon-256.png" width="128" alt="Open Player logo"></p>

# Open Player

An open video player for Linux and Windows that upscales video on NVIDIA GPUs. On Windows it uses NVIDIA RTX
Video Super Resolution; everywhere it can use its own video super-resolution network, **Pixel Flow** (PXF), which is
trained, exported and played entirely from this repo.

> [!IMPORTANT]
> **Open Player does not come with a Pixel Flow model, and never will.** Every user trains their own Pixel Flow
> on their own videos with `train.py` and exports it to `vsr.onnx` (see [2. Train a model](#2-train-a-model)).
> No trained model is in this repo, in its releases, or in any download. Without a `vsr.onnx` the player still
> plays video and, on Windows, upscales it with NVIDIA RTX VSR; Pixel Flow needs your own model.

## Install (Windows)

1. Download `OpenPlayer-Setup-<build>.exe` from [Releases](https://github.com/SpaceJamp/Open-Player/releases) and
   run it. It installs for your user only (no admin prompt) into `%LOCALAPPDATA%\Programs\Open Player`. The
   installer isn't code-signed yet, so Windows SmartScreen may say "Windows protected your PC": click *More info*,
   then *Run anyway*.
2. Play videos. On an RTX GPU, NVIDIA RTX VSR upscales them right away; no model and no extra download needed.
3. **Optional, for Pixel Flow:** train your own model (step 2 below), put `vsr.onnx` and `vsr.ref` in
   `%LOCALAPPDATA%\Open Player\`, then in Settings > Upscaling click **Download** to get TensorRT (1.9 GB, from
   NVIDIA's own server, checked against a fixed SHA-256), or **Choose folder...** for your own TensorRT 11.x.

**Updates.** Once a day the player asks GitHub whether a newer release exists (turn it off in Settings > About).
"Update now" downloads the new installer and runs it only if it carries a valid signature from Open Player's
release key and matches the signed checksum; anything else is refused. Nothing about you is sent.

**About** in Settings shows the build number and the TensorRT, NVIDIA RTX VSR and driver versions; the log file
and `--check` show them too.

```
decode (FFmpeg, NVDEC) -> YUV->RGB (CUDA) -> upscale: RTX VSR or Pixel Flow (TensorRT) -> RGBA (CUDA) -> Vulkan -> window
```

**Pixel Flow** ([model.py](model.py)) works on one frame at a time and remembers the previous one: a small flow
net estimates motion, the previous upscaled frame is warped onto the current one, and conv blocks (with pixel
attention, except in `lean`) add detail before a pixel-shuffle 2x output. A scene-cut check resets the memory.
Three sizes: Pixel Flow (`standard`, about 317K parameters), Pixel Flow Fast (`fast`) and Pixel Flow Lean
(`lean`).

Again: **no trained model ships.** You train your own Pixel Flow (step 2); the quality depends on your videos and
how long you train.

## 1. Setup

**Linux** (Arch, Fedora 44, Ubuntu 26.04; native or WSL2), with the NVIDIA driver already installed:

```bash
bash setup.sh
python3 -m venv ~/vsr-venv && ~/vsr-venv/bin/pip install torch numpy onnx onnxscript --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple
```

[setup.sh](setup.sh) installs the build tools, FFmpeg, SDL3, Vulkan and CUDA; on Arch and Fedora TensorRT comes
from NVIDIA's tarball unpacked to `/opt/tensorrt`. On a fresh Arch WSL, first run
`sudo pacman-key --init && sudo pacman-key --populate archlinux`, or downloads fail with "signature is unknown trust".

**Windows**, with the NVIDIA driver already installed:

```powershell
powershell -ExecutionPolicy Bypass -File setup-windows.ps1     # -Check: only report; -NoTraining: skip Python/PyTorch/FFmpeg
```

[setup-windows.ps1](setup-windows.ps1) installs with winget what's missing: the Visual Studio 2022 Build Tools with
C++, CUDA 13, the Vulkan SDK, Git, Python with PyTorch, and FFmpeg. It also clones
[vcpkg](https://github.com/microsoft/vcpkg) to `C:\vcpkg` (or `OPEN_PLAYER_VCPKG`). Two downloads need an NVIDIA
login, so it only checks for them:
- TensorRT for Windows, unzipped to `C:\TensorRT\TensorRT-11.5.0.63` (other versions: pass `-DTENSORRT_ROOT=...`
  to build-windows.bat; the script checks only this path)
- for RTX VSR: the [NVIDIA RTX Video SDK](https://developer.nvidia.com/rtx-video-sdk) unzipped to `C:\rtx_video_sdk`
  (its license keeps it out of this repo; without it the player builds with Pixel Flow only)

## 2. Train a model

This step is not optional if you want Pixel Flow: **there is no ready-made model to download.** Your Pixel Flow is
the one you train here.

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
- **Watch it in the player:** every 100 steps and after each validation, `train.py` writes `training.json` to the
  player's settings folder (`XDG_CONFIG_HOME` if set, else `%APPDATA%\open-player\` or `~/.config/open-player/`).
  Settings > Training shows it once a second.
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
open-player                              # no video (or double-clicked): asks for one in a file dialog
open-player --check vsr.onnx             # PASS/WARN/FAIL for CUDA, NVDEC, Vulkan, audio, TensorRT, RTX VSR, the model
open-player --selftest vsr.onnx          # TensorRT output vs PyTorch
open-player --version                    # the build number, e.g. "Open Player build 69 (11d35c8)"
```

Options: `--model FILE`, `--no-vsr`, `--no-rtx` (Windows: Pixel Flow instead of RTX VSR), `--cpu-decode` (decode on
the CPU; upscaling stays on the GPU), `--screen N`, `--fullscreen`, `--start T` and `--paused` (start T seconds in,
paused), `--stats` (one JSON line of timings per second), `--collect DIR` (see Windows below). Flags win over the
saved settings for that run and aren't saved.

The first time Pixel Flow plays a resolution, TensorRT builds an engine for it (up to a minute) and caches it next
to the model (`.win.engine` on Windows, so a shared folder works for both).

**How much it upscales.** Pixel Flow upscales 2x and the window stretches the result to fit. It's planned for the
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
upscaling badge (`RTX 2x`, `PXF 2x` or `VSR off`), settings, open and fullscreen. It fades out 2 s after the mouse
stops (the cursor hides in fullscreen) and stays while paused. Volume, seek keys, pause and upscaling changes show a
short notice. Right-click for a menu with the same actions, recent files, the info overlay and settings. Drop a video
on the window, or use Open, to switch videos.

**Settings** (`S`, the gear, or the menu) open a card over the video, which keeps playing. Tabs run down its left
side (icons only on a narrow window), and the card opens on the tab you used last:
- **General:** the upscaler (NVIDIA RTX / Pixel Flow / Off) and how long the controls stay after the mouse stops
  change at once. The decoder (GPU or CPU) and the monitor (every connected one is listed with its resolution and
  refresh rate) reopen the video where it is, on that monitor. Start fullscreen, remember the volume, show the
  info overlay at start, and remember where you left off.
- **Training:** a running `train.py` live: step and progress bar, loss, validation score (and best), speed, time
  left, the latest `suggest:` line and where the model is saved. It says "Not running" when a run has been silent
  for 5 minutes (stopped without saying so, or crashed).
- **About:** the build number, versions, updates, the log file and credits.

**Remember where I left off** (on by default): the player saves your spot in each video every 10 seconds, on quit
and when you open another one, and picks up there next time with a "Resumed at 12:34" notice. It skips spots in
the first 10 seconds or the last 30, so a finished video starts over. `--start` wins over it. The last 200 videos
are kept in `positions.txt`.

Choices are saved in `settings.txt` next to the recent-files list (`%APPDATA%\open-player\` on Windows,
`~/.config/open-player/` on Linux). `V` and the badge only compare for the moment; they don't change the saved
upscaler.

**Log file.** Everything the player writes to its error output (its messages and errors, plus FFmpeg's, SDL's,
TensorRT's and NVIDIA's; not the `--stats` lines) is saved with a timestamp per line to `open-player.log` in the same folder; the run before is kept as
`open-player.old.log`, so a crash's log survives the next start. A crash adds a last `crashed: ...` line. One run's
log stops at 5 MB. `--check` and Settings > About show the path; attach the file when you report a problem. It
stays on your PC and is never sent anywhere; it contains the paths of the videos you played, so look it over before
you post it publicly.

| Key | Action |
|---|---|
| Space | pause; at the end, play again |
| ← / → | seek 5 s |
| ↑ / ↓ | volume |
| V | upscaling on/off (compare) |
| F, double-click | fullscreen |
| I | info overlay: GPU, decoder, upscaler, frame rate, dropped frames |
| S | settings |
| Q, Esc | quit (Esc closes settings, then leaves fullscreen, first) |

### Windows and NVIDIA RTX VSR

On Windows the player upscales with **NVIDIA RTX Video Super Resolution** when the GPU and driver support it, and
falls back to Pixel Flow otherwise (choose it in settings, or `--no-rtx` for one run). CUDA and Vulkan share memory
there, so frames never pass through the CPU.

RTX VSR runs only on Windows, but Pixel Flow can learn to imitate it, and then upscales that way on Linux too:
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

## Releasing (maintainers)

The Windows build is a windowed program (no console window); started from a terminal it prints there (`--check`,
`--version`), and the log file has everything.

- `installer\build-installer.bat` builds the player and `installer\out\OpenPlayer-Setup-<build>.exe` (Inno Setup 6;
  `setup-windows.ps1` installs it). The installer never contains TensorRT or a model.
- `tools\release.ps1 -Notes "..."` (from a clean `main` only) builds it, writes `update.json`, signs it with the
  update key (asks for its passphrase), checks the signature against the key built into the player, and publishes
  the GitHub release after you type `yes`.
- The build number is the number of commits on `main`, so `main`'s history must never be rewritten.
- The update key: `tools\make-update-key.ps1` made it once; its encrypted private half lives only in the private
  `Open-Player-Keys` repo, its passphrase only in the maintainer's password manager, its public half in
  `player\update_pubkey.h`. Losing it means installed copies can't update themselves any more.

## Limits

- SDR only: HDR video plays with washed-out colours.
- Interlaced video (DVDs, TV recordings) isn't deinterlaced, so motion shows comb lines.
- Audio-only files don't open, and there's no switching between audio or subtitle tracks yet.
- On WSL, CUDA and Vulkan can't share memory, so each frame goes through the CPU; native Linux and Windows are
  zero-copy. WSLg also tears; native systems don't.
- Opening another video restarts the player process.

## License

Open Player is licensed under the [GNU Affero General Public License v3.0](LICENSE), with an
[additional permission](LICENSE-EXCEPTION) to link NVIDIA's CUDA, TensorRT and RTX Video SDK libraries. Bundled
third-party code and fonts keep their own licenses: Dear ImGui (MIT, `player/third_party/imgui/LICENSE.txt`),
Inter (SIL OFL) and Tabler Icons (MIT, both in `assets/fonts/`).
