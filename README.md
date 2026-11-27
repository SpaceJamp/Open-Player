<p align="center"><img src="assets/icon-256.png" width="128" alt="Open Player logo"></p>

# Open Player

A video player for Windows that makes video sharper on NVIDIA graphics cards, with English captions drawn crisply on
top. It upscales with **NVIDIA RTX Video Super Resolution**, or with **Pixel Flow**: a video super-resolution network
you train yourself, on your own videos, inside the player.

> [!IMPORTANT]
> **Open Player doesn't come with a Pixel Flow model.** Every Pixel Flow is trained by its user on their own videos.
> Without one, the player still plays everything and, on an RTX card, upscales with NVIDIA RTX VSR.

**For everyone:** [Get started](#get-started) · [Upscaling](#upscaling) · [Captions](#captions) ·
[Train your own Pixel Flow](#train-your-own-pixel-flow) · [Everyday use](#everyday-use) ·
[Privacy and updates](#privacy-and-updates) · [Limits](#limits)
**For developers:** [Build from source](#build-from-source) · [Command line](#command-line) ·
[Training from the command line](#training-from-the-command-line) · [Tests](#tests) ·
[Releasing](#releasing-maintainers) · [License](#license)

## Get started

1. Download `OpenPlayer-Setup-<build>.exe` from [Releases](https://github.com/SpaceJamp/Open-Player/releases) and run
   it. It installs for your user only (no admin prompt) into `%LOCALAPPDATA%\Programs\Open Player`. It isn't
   code-signed yet, so SmartScreen may say "Windows protected your PC": click **More info**, then **Run anyway**.
2. Open a video: double-click it, drop it on the window, or click **Open**. On an RTX card, NVIDIA RTX VSR upscales it
   straight away, with nothing else to download.
3. Want captions? Press **C**. Want your own upscaler? [Train a Pixel Flow](#train-your-own-pixel-flow).

You need an NVIDIA graphics card with a current driver for upscaling (an RTX card for RTX VSR and for Pixel Flow's
default engine). Without one, Open Player still plays video, just without upscaling.

## Upscaling

Settings > General > **Upscaler**: **NVIDIA RTX**, **Pixel Flow** or **Off**. **V** (or the badge in the control bar,
`RTX 2x` / `PXF 2x` / `VSR off`) switches it off and on to compare, without changing the setting.

**How much.** Upscaling is 2x, planned for your screen: a video as big as the screen plays as it is; one whose 2x
would overshoot is shrunk first so 2x lands exactly on the screen (1440p on a 4K screen becomes 1080p, then 2160p);
smaller video is doubled, then stretched to fit.

**Pixel Flow's engine.** Pixel Flow runs on one of NVIDIA's inference libraries, chosen in Settings > General >
Upscaling:

| Engine | Download | 1080p to 4K, `standard` model, RTX 5080 |
|---|---|---|
| **TensorRT for RTX** (the default) | **Download** button, 89 MB, the first time you need it | about 29 ms per frame |
| **Full TensorRT** (optional, faster) | **Faster: full TensorRT**, 1.9 GB | about 23 ms per frame |

Both come from NVIDIA's own servers and are checked against a fingerprint (SHA-256) built into the player before
anything is unpacked. When both are there, the full TensorRT is used. **Choose folder...** takes a TensorRT 11 or a
TensorRT for RTX 1.8 you already have (the unzipped folder or its `bin`). The first time Pixel Flow plays a new
video size, the engine prepares for it: a few seconds with TensorRT for RTX, up to a minute with the full TensorRT.
Settings > About (and `--check`) shows which engine runs.

**When Pixel Flow can't keep up.** If, over about 10 seconds of playback, more than 5% of frames are dropped or
Pixel Flow needs longer per frame than the video gives it, the player switches that video to NVIDIA RTX VSR and says
so. Your setting doesn't change: the next video tries Pixel Flow again. (Without RTX VSR it keeps Pixel Flow and
tells you once.)

## Captions

Press **C**, click **CC** in the control bar, or pick a track under **Captions** in the right-click menu.

- **What shows:** the English subtitle tracks in the file, both text tracks and Blu-ray picture subtitles, and
  `.srt` files next to the video (`movie.srt`, `movie.en.srt`, also Windows-1252 and UTF-16 ones).
- **Which track:** the SDH one (closed captions, with sound descriptions like *[door creaks]*) when there is one,
  otherwise plain English. Pick another in the menu.
- **How they look:** white text with a black outline, drawn after upscaling, so they're sharp at any size; they move up
  while the control bar shows.
- **Remembered:** on/off stays as you left it (also in Settings > General > Captions). A **forced** English track
  (foreign-language lines translated) shows even with captions off.

## Train your own Pixel Flow

Everything happens in Settings > **Training**:

1. **Choose videos folder...** and pick a folder of videos (subfolders count). The tab shows how many it found.
2. **Start.** The first time, the player downloads the **Training add-on** (about 2.3 GB, once): Python 3.13,
   PyTorch with CUDA and FFmpeg's tools, each from its own project's server and fingerprint-checked. You don't need
   Python or anything else installed.
3. Training runs in the background: step, loss, a quality score, time left and a suggestion, updated every 2 seconds.
   A full run is 100,000 steps (about 4 hours on an RTX 5080, longer on slower cards). Keep watching videos if you
   like; closing the player doesn't stop training, and the tab picks it up again when you reopen it.
4. When it finishes, the player turns the best model so far into your Pixel Flow and loads it; choose **Pixel Flow**
   as the upscaler. **Use this model** does the same at any time. The Pixel Flow it replaces is kept as
   `vsr.prev.onnx` / `vsr.prev.ref`.

The other buttons: **Stop** saves and stops (**Start** continues later). **Train more**, after a finished run, adds
another 100,000 steps with the folder you choose now, for when you've added videos. **Start over** deletes the model
being trained (the Pixel Flow you use stays). **Remove add-on** deletes the Training add-on.

**A good Pixel Flow** can't get sharper than what it learns from: high-quality sources (Blu-ray rips, 4K shrunk to
1080p) beat low-bitrate web downloads, and a mix of content (animation, film, bright scenes, sports, grain) works on
more videos than one genre. HDR videos train poorly. The quality score compares Pixel Flow with plain resizing on
clips held out of training: it starts below 0 dB and should climb above it.

**Your graphics card:** training picks its batch size from the card's memory and halves it if it ever runs out, so
cards with less memory train too, just slower; if even the smallest batch doesn't fit, the tab says so.

## Everyday use

Move the mouse to show the control bar: play/pause, time and seek bar, volume (hover the speaker for a slider),
captions, the upscaling badge, settings, open and fullscreen. It fades 2 seconds after the mouse stops and stays
while paused. Right-click for a menu with the same actions, recent files, captions and the info overlay.

| Key | Action |
|---|---|
| Space | pause; at the end, play again |
| ← / → | seek 5 s |
| ↑ / ↓ | volume |
| C | captions on/off |
| V | upscaling on/off, to compare |
| F, double-click | fullscreen |
| I | info overlay: GPU, decoder, upscaler, frame rate, dropped frames |
| S | settings |
| Q, Esc | quit (Esc first closes settings, then leaves fullscreen) |

**Settings** open as a card over the video, which keeps playing, with tabs down the left side:
- **General:** upscaler and Pixel Flow's engine, captions, decoder (GPU or CPU), the monitor to play on, how long the
  controls stay, start fullscreen, remember the volume, the info overlay at start, remember where I left off.
- **Training:** your own Pixel Flow (above); a `train.py` started from the command line shows here too.
- **About:** build number, engine, TensorRT, RTX VSR and driver versions, updates, the log file and credits.

**Remember where I left off** (on by default) saves your spot in each video and picks up there next time ("Resumed at
12:34"), except in the first 10 seconds or the last 30, so a finished video starts over.

## Privacy and updates

- **Updates:** once a day the player asks GitHub for a newer release (turn it off in Settings > About). **Update
  now** runs the new installer only if it carries a valid signature from Open Player's release key and matches the
  signed checksum.
- **Downloads** (TensorRT for RTX, TensorRT, the Training add-on) come only from fixed addresses at NVIDIA,
  python.org, PyTorch, PyPI and FFmpeg's builder on GitHub, each checked against its fingerprint. Nothing about you
  is sent; your videos never leave your PC.
- **The log file** (`%APPDATA%\open-player\open-player.log`, shown in About) has the player's messages and errors,
  the previous run's in `open-player.old.log`. It contains the paths of videos you played: look it over before
  posting it. Attach it when you report a problem.
- **Where things live:** settings, recent files and resume positions in `%APPDATA%\open-player\`; your Pixel Flow,
  training and the add-ons in `%LOCALAPPDATA%\Open Player\`. Uninstalling asks whether to delete that folder too.

## Limits

- SDR only: HDR video plays with washed-out colours (and trains poorly).
- Interlaced video (DVDs, TV recordings) isn't deinterlaced, so motion shows comb lines.
- Audio-only files don't open; audio tracks can't be switched yet.
- Captions: English only; styled (ASS) subtitles show as plain text.
- TensorRT for RTX needs an RTX card; on older NVIDIA cards Pixel Flow needs the full TensorRT.
- Opening another video restarts the player process.
- Windows is the supported system for now; Linux builds from source (below), without the downloads and training
  from the player.

---

## Build from source

**Windows**, with the NVIDIA driver installed:

```powershell
powershell -ExecutionPolicy Bypass -File setup-windows.ps1     # -Check: only report; -NoTraining: skip Python/PyTorch/FFmpeg
build-windows.bat                                              # -> build-win\open-player.exe; the first run builds SDL3 and FFmpeg (~10-30 min)
```

[setup-windows.ps1](setup-windows.ps1) installs what's missing with winget (Visual Studio 2022 Build Tools with
C++, CUDA, the Vulkan SDK, Git, Inno Setup, Python with PyTorch 2.15 for CUDA 13.2, FFmpeg) and clones
[vcpkg](https://github.com/microsoft/vcpkg) to `C:\vcpkg` (or `OPEN_PLAYER_VCPKG`). NVIDIA's SDKs need your NVIDIA
login, so it only checks for them:

- **TensorRT 11.5** (zip) in `C:\TensorRT\TensorRT-11.5.0.63` (or `-DTENSORRT_ROOT=...`). Required.
- **TensorRT for RTX 1.7** and **1.8** SDKs ([download](https://developer.nvidia.com/tensorrt-rtx)), each unzipped
  to `C:\` (`C:\TensorRT-RTX-1.7.<build>`, `C:\TensorRT-RTX-1.8.<build>`). Optional, but a release needs 1.7: it is
  the version the add-on downloads. Each SDK found is compiled into the player separately (`trt_rtx.cpp`, under its
  own namespace, its DLL loaded at run time), because their headers share names with the full TensorRT's.
- **NVIDIA RTX Video SDK** ([download](https://developer.nvidia.com/rtx-video-sdk)) in `C:\rtx_video_sdk`.
  Optional; without it there's no RTX VSR. Its licence keeps it out of this repo.

Open a new terminal after setup so the new PATH applies.

**Linux** (Arch, Fedora 44, Ubuntu 26.04; native or WSL2), with the NVIDIA driver installed:

```bash
bash setup.sh
python3 -m venv ~/vsr-venv && ~/vsr-venv/bin/pip install torch numpy onnx onnxscript --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple
cmake -B build -S player && cmake --build build -j
```

[setup.sh](setup.sh) installs the build tools, FFmpeg, SDL3, Vulkan and CUDA; on Arch and Fedora TensorRT comes from
NVIDIA's tarball in `/opt/tensorrt`. On a fresh Arch WSL, first run
`sudo pacman-key --init && sudo pacman-key --populate archlinux`.

**How it works:** decode (FFmpeg, NVDEC) → YUV to RGB (CUDA) → upscale with RTX VSR or Pixel Flow (TensorRT or
TensorRT for RTX) → Vulkan → window, with the controls and captions drawn by Dear ImGui on top. CUDA and Vulkan
share memory, so frames never pass through the CPU (except on WSL). **Pixel Flow** ([model.py](model.py)) works one
frame at a time and remembers the previous one: a small flow net estimates motion, the previous upscaled frame is
warped onto the current one, and conv blocks (with pixel attention, except in `lean`) add detail before a
pixel-shuffle 2x output; a scene-cut check resets the memory. Sizes: `standard` (about 317K parameters), `fast`,
`lean`.

## Command line

```bash
open-player movie.mkv                    # finds vsr.onnx in the current folder or next to the program
open-player                              # no video: asks for one
open-player --check vsr.onnx             # PASS/WARN/FAIL for CUDA, NVDEC, Vulkan, audio, the engine, RTX VSR, the model
open-player --selftest vsr.onnx          # Pixel Flow's output against PyTorch's (.ref from export.py)
open-player --version                    # e.g. "Open Player build 150 (abc1234)"
```

Options: `--model FILE`, `--no-vsr`, `--no-rtx` (Pixel Flow instead of RTX VSR), `--cpu-decode`, `--screen N`,
`--fullscreen`, `--start T`, `--paused`, `--stats` (a JSON line of timings per second), `--collect DIR` (below),
`--engine full|rtx` (force Pixel Flow's engine, for testing). Flags win over saved settings for that run. The
Windows build is a windowed program; started from a terminal it still prints there.

## Training from the command line

The Training tab runs exactly this; the command line has every option. Python 3.10 or newer with a CUDA build of
PyTorch (on Python 3.15, PyTorch 2.15 or newer); `setup-windows.ps1` installs a working set.

```bash
py train.py "D:\Videos" --out vsr.pt                 # default folder: videos/ next to train.py
py train.py "D:\Videos" --out vsr.pt --resume vsr.pt # continue (saved every 2000 steps, on Ctrl+C and on Stop)
py train.py "D:\Videos" --out vsr.pt --resume vsr.pt --more 100000   # another 100000 steps on a finished model
py export.py vsr_best.pt                             # -> vsr_best.onnx + vsr_best.ref; rename to vsr.onnx / vsr.ref
```

(`~/vsr-venv/bin/python` on Linux.) For an installed player, copy `vsr.onnx` and `vsr.ref` to
`%LOCALAPPDATA%\Open Player\`.

- **Input:** frames are shrunk and given random blur, grain and JPEG-style blocks, so the network learns to clean up
  compressed video rather than sharpen its artifacts.
- **Validation:** every 1000 steps, `validation +X dB vs bilinear` on 32 clips from two held-out videos, saved to
  `<out>.val.pt` so scores stay comparable as you add videos; the best model is kept as `<out>_best.pt`.
- **Diagnostics:** a `suggest:` line after each validation (slow data loading, plateau, overfitting, unstable
  gradients, a weak clip, flicker, soft edges, or `ok`); `py train_diag.py <out>.diag.jsonl` summarises the metrics.
- **Status for the player:** `training.json` in the player's settings folder, every 2 seconds, with `state`
  training/finished/stopped/failed (and a `reason`); a `training.stop` file next to it makes the run save and stop.
- **Sampling:** each title (a top-level file or folder, such as a series) is picked equally often.
- **Batch size:** from the GPU's memory unless `--batch` is given (8 at most), halved on out-of-memory.

Options: `--size standard|fast|lean`, `--steps`, `--more N`, `--lr`, `--batch`, `--crop`, `--frames`, `--workers`,
`--fft-loss W` (sharper edges; try 0.05-0.3), `--teacher ck.pt --distill W` (a smaller model learns from a bigger
one), `--pairs DIR --pairs-weight W` (learn from RTX VSR). A resumed run keeps its steps, learning rate and losses.

**Learning from RTX VSR:** play videos with `--collect pairs` (about every 10 seconds the player saves 8 frames of a
128 px crop next to RTX VSR's 2x upscale of it, up to 2 GB), then train with `--pairs pairs`.

[upscale.py](upscale.py) upscales a file without the player (`py upscale.py in.mp4 -o out.mp4`);
[panel.py](panel.py) is an older control panel for training and playing (Linux-first).

## Tests

```bash
build-win\open-player.exe --uitest       # the player's own checks: UI, settings, captions, add-ons, engines, training
py tests/train_check.py                  # train.py: degradations, losses, data, stop/--more/failed, batch size
bash tests/smoke.sh vsr.onnx             # plays generated clips (captions too), runs the model and UI self-tests
bash tests/codecs.sh                     # every format FFmpeg can write, from H.264 to ProRes and 7.1 audio
```

On Windows run the shell tests from Git Bash with `OPEN_PLAYER=build-win/open-player.exe`; NVIDIA's Windows driver
can't render offscreen, so their windows open on the second monitor (`OPEN_PLAYER_SCREEN=N` to change). On Linux
they run offscreen and silent.

## Releasing (maintainers)

- `installer\build-installer.bat` builds the player and `installer\out\OpenPlayer-Setup-<build>.exe` (Inno Setup 6):
  the player and the training scripts; never TensorRT, TensorRT for RTX, the Training add-on or a model.
- `tools\release.ps1 -Notes "..."` (from a clean `main` only) builds it, signs `update.json` with the update key
  (asks for its passphrase), checks the signature against the key built into the player, and publishes the GitHub
  release after you type `yes`.
- The build number is the number of commits on `main`, so `main`'s history is never rewritten.
- The update key was made once with `tools\make-update-key.ps1`; its encrypted private half lives only in the private
  `Open-Player-Keys` repo, its passphrase only in the maintainer's password manager, its public half in
  `player\update_pubkey.h`. Without it, installed copies can't update themselves.
- Pinned downloads: the TensorRT and TensorRT for RTX add-ons in `player\addon.h`, the Training add-on in
  `player\training_addon.cpp` (URL, size, SHA-256). A new TensorRT for RTX version needs its SDK for the build and a
  new pin; a new Training add-on set must pass `tests/train_check.py` on its own Python, then raise
  `TRAINING_ADDON_VERSION`.

## License

Open Player is licensed under the [GNU Affero General Public License v3.0](LICENSE), with an
[additional permission](LICENSE-EXCEPTION) to link NVIDIA's CUDA, TensorRT, TensorRT for RTX and RTX Video SDK
libraries. Bundled code and fonts keep their own licences: Dear ImGui (MIT, `player/third_party/imgui/LICENSE.txt`),
Inter (SIL OFL) and Tabler Icons (MIT, both in `assets/fonts/`). The downloaded add-ons keep theirs; see
`installer/THIRD-PARTY-NOTICES.txt`.

This software uses the NVIDIA RTX Video SDK. NVIDIA, GeForce RTX and NVIDIA RTX are trademarks of NVIDIA Corporation.
