<p align="center"><img src="assets/icon-256.png" width="128" alt="Open Player logo"></p>

# Open Player

A video player for Windows (and Linux) that makes video sharper on NVIDIA graphics cards. It upscales with
**NVIDIA RTX Video Super Resolution**, or with **Pixel Flow**, a video super-resolution network you train yourself
on your own videos, right in the player.

> [!IMPORTANT]
> **No Pixel Flow model comes with Open Player, and none ever will.** Every Pixel Flow is trained by its user, on
> their own videos. Without one, the player still plays everything and, on an RTX card, upscales with NVIDIA RTX
> VSR.

- [Get started](#get-started) · [Train your own Pixel Flow](#train-your-own-pixel-flow) ·
  [Using the player](#using-the-player) · [Privacy and updates](#privacy-and-updates) · [Limits](#limits)
- For developers: [Build from source](#build-from-source) · [Training from the command line](#training-from-the-command-line) ·
  [Tests](#tests) · [Releasing](#releasing-maintainers) · [License](#license)

## Get started

1. Download `OpenPlayer-Setup-<build>.exe` from [Releases](https://github.com/SpaceJamp/Open-Player/releases) and
   run it. It installs for your user only, without an admin prompt, into `%LOCALAPPDATA%\Programs\Open Player`.
   The installer isn't code-signed yet, so SmartScreen may say "Windows protected your PC": click **More info**, then
   **Run anyway**.
2. Open a video (double-click it, drop it on the window, or use **Open**). On an RTX card NVIDIA RTX VSR upscales it
   straight away; nothing else to download.
3. Optional: [train your own Pixel Flow](#train-your-own-pixel-flow) for an upscaler tuned to the videos you watch.

You need an NVIDIA graphics card for upscaling (RTX for RTX VSR; any recent NVIDIA card for Pixel Flow) and a
current NVIDIA driver. Without one, the player plays video without upscaling.

## Train your own Pixel Flow

Everything happens in **Settings > Training**:

1. **Choose videos folder...** and pick a folder of videos (subfolders count). The tab shows how many it found.
2. **Start.** The first time, the player downloads the **Training add-on** (about 2.3 GB, once): Python 3.13,
   PyTorch with CUDA, and FFmpeg's tools, each from its own project's server and checked against a fingerprint
   (SHA-256) built into the player. You don't need Python or anything else installed.
3. Training runs in the background, with step, loss, a quality score, time left and a suggestion in the tab. A full
   run is 100,000 steps; on an RTX 5080 that's about 9 hours. You can keep watching videos, and closing the player
   doesn't stop it: reopen the tab to see where it is.
4. When it finishes, the player turns the best model so far into your Pixel Flow and loads it; pick **Pixel Flow**
   as the upscaler in Settings > General. Want to try it earlier? **Use this model** does the same at any time. The
   Pixel Flow it replaces is kept as `vsr.prev.onnx` / `vsr.prev.ref`, in case you want it back.

Pixel Flow also needs **TensorRT** (NVIDIA's inference library): in Settings > General > Upscaling, click
**Download** (1.9 GB, from NVIDIA's own server, fingerprint-checked), or **Choose folder...** if you have TensorRT 11
already (the folder you unzipped is fine).

The other buttons:
- **Stop** saves and stops; **Start** later continues where it stopped.
- **Train more** (after a run finished): another 100,000 steps on the same model, with the folder you choose now.
  Use it when you've added videos.
- **Start over** deletes the model trained so far (the Pixel Flow you're using stays) and starts from zero.
- **Remove add-on** deletes the Training add-on; your trained model stays.

**What makes a good Pixel Flow.** It can't get sharper than what it learns from: high-quality sources (Blu-ray
rips, 4K shrunk to 1080p) beat low-bitrate web downloads, and a mix (animation, film, bright scenes, sports, grain)
works on more videos than one genre. HDR videos train poorly (see [Limits](#limits)). The quality score is measured
against plain resizing on clips held out of training: it starts below 0 dB and should climb above it.

**Your graphics card.** Training picks its batch size from your card's memory and halves it if it ever runs out,
so cards with less memory train too, a bit slower. If even the smallest batch doesn't fit, the tab says so.

Training reads your videos only on your PC; nothing is uploaded. Your model and training files live in
`%LOCALAPPDATA%\Open Player\` (`vsr.onnx` is your Pixel Flow, `training-work\` the run in progress).

## Using the player

Move the mouse to show the control bar: play/pause, time and seek bar, volume (hover the speaker for a slider),
the upscaling badge (`RTX 2x`, `PXF 2x` or `VSR off`), settings, open and fullscreen. It fades 2 seconds after the
mouse stops and stays while paused. Right-click for a menu with the same actions, recent files and the info overlay.

| Key | Action |
|---|---|
| Space | pause; at the end, play again |
| ← / → | seek 5 s |
| ↑ / ↓ | volume |
| V | upscaling on/off, to compare |
| F, double-click | fullscreen |
| C | captions on/off |
| I | info overlay: GPU, decoder, upscaler, frame rate, dropped frames |
| S | settings |
| Q, Esc | quit (Esc first closes settings, then leaves fullscreen) |

**Settings** open as a card over the video, which keeps playing. Tabs run down the left side:
- **General:** the upscaler (NVIDIA RTX, Pixel Flow or Off), TensorRT, the decoder (GPU or CPU), the monitor to play
  on, how long the controls stay, start fullscreen, remember the volume, the info overlay at start, and **remember
  where I left off**.
- **Training:** training your own Pixel Flow (above). A `train.py` started from the command line shows here too.
- **About:** build number, TensorRT, RTX VSR and driver versions, updates, the log file and credits.

**Captions.** Press **C**, click **CC** in the control bar, or pick a track under **Captions** in the right-click
menu. They show English subtitle tracks from the file (text tracks and Blu-ray picture subtitles) and `.srt` files
next to the video (`movie.srt`, `movie.en.srt`). With captions on, the player picks the SDH track (closed captions,
with sound descriptions) when there is one, else plain English. They're drawn after upscaling, so they stay sharp,
and move up while the control bar shows. On/off is remembered (also in Settings > General). A forced English track
(foreign-language lines translated) shows even with captions off.

**Remember where I left off** (on by default) saves your spot in each video and picks up there next time ("Resumed at
12:34"). It skips the first 10 seconds and the last 30, so a finished video starts over.

**How much it upscales.** Upscaling is 2x, planned for your screen: a video as large as the screen plays as it is; a
video whose 2x would overshoot is shrunk first so 2x lands exactly on the screen (1440p on a 4K screen becomes 1080p,
then 2160p); smaller video is doubled, then stretched to fit. The first time Pixel Flow plays a resolution, TensorRT
prepares for it (up to a minute); after that it's instant.

**Speed** on an RTX 5080, 1080p to 4K: Pixel Flow `standard` 23-25 ms per frame (up to 30 fps video), `fast` 9-13 ms
(about 60 fps).

## Privacy and updates

- **Updates:** once a day the player asks GitHub whether there's a newer release (turn it off in Settings > About).
  **Update now** runs the new installer only if it carries a valid signature from Open Player's release key and
  matches the signed checksum. Nothing about you is sent.
- **Downloads** (TensorRT, the Training add-on) come only from fixed addresses at NVIDIA, python.org, PyTorch, PyPI
  and FFmpeg's builder on GitHub, and each file must match the fingerprint built into the player.
- **The log file** (`%APPDATA%\open-player\open-player.log`, shown in About) records the player's messages and
  errors; the previous run's is kept as `open-player.old.log`. It stays on your PC. It contains the paths of videos
  you played, so look it over before posting it publicly. Attach it when you report a problem.
- Settings, recent files and resume positions are in `%APPDATA%\open-player\`.

Uninstalling removes the program and asks whether to delete `%LOCALAPPDATA%\Open Player` too (your model, your
training and the add-ons).

## Limits

- SDR only: HDR video plays with washed-out colours (and trains poorly).
- Interlaced video (DVDs, TV recordings) isn't deinterlaced, so motion shows comb lines.
- Audio-only files don't open; there's no switching between audio tracks yet.
- Captions: English only; styled (ASS) subtitles show as plain text.
- Opening another video restarts the player process.
- On WSL, CUDA and Vulkan can't share memory, so every frame goes through the CPU, and WSLg tears.

---

## Build from source

**Windows**, with the NVIDIA driver installed:

```powershell
powershell -ExecutionPolicy Bypass -File setup-windows.ps1     # -Check: only report; -NoTraining: skip Python/PyTorch/FFmpeg
build-windows.bat                                              # -> build-win\open-player.exe; the first run builds SDL3 and FFmpeg (~10-30 min)
```

[setup-windows.ps1](setup-windows.ps1) installs what's missing with winget: Visual Studio 2022 Build Tools with C++,
CUDA, the Vulkan SDK, Git, Inno Setup, Python with PyTorch (2.15, CUDA 13.2) and FFmpeg, and clones
[vcpkg](https://github.com/microsoft/vcpkg) to `C:\vcpkg` (or `OPEN_PLAYER_VCPKG`). Two downloads need an NVIDIA
login, so it only checks for them:
- **TensorRT 11.5** for Windows, unzipped to `C:\TensorRT\TensorRT-11.5.0.63` (another version: pass
  `-DTENSORRT_ROOT=...` to build-windows.bat). The build needs it.
- **NVIDIA RTX Video SDK** ([download](https://developer.nvidia.com/rtx-video-sdk)), unzipped to `C:\rtx_video_sdk`.
  Optional; without it the player builds without RTX VSR. Its license keeps it out of this repo.

Open a new terminal after setup so the new PATH applies.

**Linux** (Arch, Fedora 44, Ubuntu 26.04; native or WSL2), with the NVIDIA driver installed:

```bash
bash setup.sh
python3 -m venv ~/vsr-venv && ~/vsr-venv/bin/pip install torch numpy onnx onnxscript --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple
cmake -B build -S player && cmake --build build -j
```

[setup.sh](setup.sh) installs the build tools, FFmpeg, SDL3, Vulkan and CUDA; on Arch and Fedora TensorRT comes from
NVIDIA's tarball unpacked to `/opt/tensorrt`. On a fresh Arch WSL, run
`sudo pacman-key --init && sudo pacman-key --populate archlinux` first. Training from the player is Windows-only;
on Linux train from the command line.

### Command line

```bash
open-player movie.mkv                    # finds vsr.onnx in the current folder or next to the program
open-player                              # no video: asks for one
open-player --check vsr.onnx             # PASS/WARN/FAIL for CUDA, NVDEC, Vulkan, audio, TensorRT, RTX VSR, the model
open-player --selftest vsr.onnx          # TensorRT's output against PyTorch's
open-player --version                    # e.g. "Open Player build 69 (11d35c8)"
```

Options: `--model FILE`, `--no-vsr`, `--no-rtx` (Pixel Flow instead of RTX VSR), `--cpu-decode`, `--screen N`,
`--fullscreen`, `--start T`, `--paused`, `--stats` (a JSON line of timings per second), `--collect DIR` (below).
Flags win over saved settings for that run. The Windows build is a windowed program; started from a terminal it
still prints there.

**Pipeline:** decode (FFmpeg, NVDEC) → YUV to RGB (CUDA) → upscale with RTX VSR or Pixel Flow (TensorRT) → Vulkan →
window. CUDA and Vulkan share memory, so frames never pass through the CPU (except on WSL).

**Pixel Flow** ([model.py](model.py)) works one frame at a time and remembers the previous one: a small flow net
estimates motion, the previous upscaled frame is warped onto the current one, and conv blocks (with pixel attention,
except in `lean`) add detail before a pixel-shuffle 2x output. A scene-cut check resets the memory. Sizes: `standard`
(about 317K parameters), `fast` and `lean`.

## Training from the command line

The player's Training tab runs exactly this; the command line gives you every option. Python 3.10 or newer with a
CUDA build of PyTorch (on Python 3.15, PyTorch 2.15 or newer: 2.14 crashes at exit there); setup-windows.ps1
installs a working set.

```bash
py train.py "D:\Videos" --out vsr.pt                 # default folder: videos/ next to train.py
py train.py "D:\Videos" --out vsr.pt --resume vsr.pt # continue (saved every 2000 steps, on Ctrl+C and on Stop)
py train.py "D:\Videos" --out vsr.pt --resume vsr.pt --more 100000   # another 100000 steps on a finished model
py export.py vsr_best.pt                             # -> vsr_best.onnx + vsr_best.ref; rename to vsr.onnx / vsr.ref
```

(`py` on Windows; `~/vsr-venv/bin/python` on Linux.) Copy `vsr.onnx` and `vsr.ref` to `%LOCALAPPDATA%\Open Player\`
for an installed player.

- **Input:** frames are shrunk and given random blur, grain and JPEG-style block compression, so the network learns
  to clean up real compressed video instead of sharpening its artifacts.
- **Validation:** every 1000 steps, `validation +X dB vs bilinear` on 32 clips from two videos held out of training,
  saved to `<out>.val.pt` and reused so scores stay comparable when you add videos. The best model is kept as
  `<out>_best.pt`.
- **Diagnostics:** after each validation a `suggest:` line names what to improve (slow data loading, plateau,
  overfitting, unstable gradients, a weak clip, flicker, soft edges, or `ok`); `py train_diag.py <out>.diag.jsonl`
  summarises the metrics.
- **Status for the player:** `training.json` in the player's settings folder (`%APPDATA%\open-player\`,
  `~/.config/open-player/`, or `XDG_CONFIG_HOME`), with `state` training/finished/stopped/failed (and a `reason`).
  A `training.stop` file next to it makes the run save and stop.
- **Sampling:** each title (a top-level file or folder, such as a series) is picked equally often.
- **Batch size:** from the GPU's memory when `--batch` isn't given (8 at most), halved on out-of-memory.

Options: `--size standard|fast|lean`, `--steps`, `--more N`, `--lr`, `--batch`, `--crop`, `--frames`, `--workers`,
`--fft-loss W` (sharper edges; try 0.05-0.3), `--teacher ck.pt --distill W` (a smaller model learns from a bigger
one), `--pairs DIR --pairs-weight W` (learn from RTX VSR, below). A resumed run keeps its steps, learning rate and
loss settings.

**Learning from RTX VSR.** Play videos with `--collect pairs`: about every 10 seconds the player saves 8 frames of a
128 px crop next to RTX VSR's 2x upscale of it (only while RTX upscales exactly 2x; up to 2 GB). Then train with
`--pairs pairs`, and Pixel Flow learns to imitate RTX VSR, also for Linux.

[panel.py](panel.py) is an older one-window control panel for training, exporting and playing (Linux-first).
[upscale.py](upscale.py) upscales a file without the player: `py upscale.py in.mp4 -o out.mp4`.

## Tests

```bash
build-win\open-player.exe --uitest       # the player's own checks: UI, settings, add-ons, training runs
py tests/train_check.py                  # train.py: degradations, losses, data, stop/--more/failed, batch size
bash tests/smoke.sh vsr.onnx             # plays generated clips, runs the model and UI self-tests
bash tests/codecs.sh                     # every format FFmpeg can write, from H.264 to ProRes and 7.1 audio
```

On Linux the shell tests run offscreen and silent. On Windows run them from Git Bash with
`OPEN_PLAYER=build-win/open-player.exe`; NVIDIA's Windows driver can't render offscreen, so their windows open on
the second monitor (`OPEN_PLAYER_SCREEN=N` to change).

## Releasing (maintainers)

- `installer\build-installer.bat` builds the player and `installer\out\OpenPlayer-Setup-<build>.exe` (Inno Setup 6).
  The installer holds the player and the training scripts; never TensorRT, the Training add-on or a model.
- `tools\release.ps1 -Notes "..."` (from a clean `main` only) builds it, writes and signs `update.json` with the update
  key (asks for its passphrase), checks the signature against the key built into the player, and publishes the GitHub
  release after you type `yes`.
- The build number is the number of commits on `main`, so `main`'s history must never be rewritten.
- The update key: made once with `tools\make-update-key.ps1`; its encrypted private half lives only in the private
  `Open-Player-Keys` repo, its passphrase only in the maintainer's password manager, its public half in
  `player\update_pubkey.h`. Losing it means installed copies can't update themselves any more.
- The Training add-on's files are pinned in `player\training_addon.cpp` (URL, size, SHA-256). To change them, prove
  the new set passes `tests/train_check.py` on the add-on's own Python, then raise `TRAINING_ADDON_VERSION` so
  installed add-ons are offered the update.

## License

Open Player is licensed under the [GNU Affero General Public License v3.0](LICENSE), with an
[additional permission](LICENSE-EXCEPTION) to link NVIDIA's CUDA, TensorRT and RTX Video SDK libraries. Bundled
third-party code and fonts keep their own licenses: Dear ImGui (MIT, `player/third_party/imgui/LICENSE.txt`), Inter
(SIL OFL) and Tabler Icons (MIT, both in `assets/fonts/`). The downloaded add-ons keep theirs; see
`installer/THIRD-PARTY-NOTICES.txt`.

This software uses the NVIDIA RTX Video SDK. NVIDIA, GeForce RTX and NVIDIA RTX are trademarks of NVIDIA Corporation.
