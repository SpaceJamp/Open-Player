# Open Player

A video player for Linux that upscales video 2x on an NVIDIA GPU with our own network, in the spirit of
RTX Video Super Resolution. The network, training, and player are all in this repo.

```
decode (NVDEC) -> YUV->RGB (CUDA) -> VSR network (TensorRT) -> RGBA (CUDA) -> Vulkan blit -> window
```

The network ([model.py](model.py), about 317K parameters) works on one frame at a time and remembers the last one.
A coarse flow net estimates motion, the previous upscaled frame is warped onto the current one, and conv
blocks with pixel attention add detail before a pixel-shuffle output. A scene-cut check resets the memory.

## 1. Setup (once, needs sudo)

[setup.sh](setup.sh) handles Arch, Fedora 44 and Ubuntu 26.04, natively or on WSL2. On Arch and Fedora,
TensorRT comes from NVIDIA's tarball, unpacked to `/opt/tensorrt`. The NVIDIA driver must already be installed.

Fresh Arch on WSL: first run `sudo pacman-key --init && sudo pacman-key --populate archlinux`, or every
download fails with "signature is unknown trust".

```bash
bash setup.sh
python3 -m venv ~/vsr-venv && ~/vsr-venv/bin/pip install torch numpy onnx onnxscript --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple
```

## Control panel

```bash
~/vsr-venv/bin/python panel.py
```

One window for the whole workflow:
- **Training:** pick your video folder and model size, Train / Stop, with live progress and the dB gain over bilinear.
  Stop saves the model, and Train continues the same `vsr.pt`.
- **Model:** export for the player, with the self-test run automatically.
- **Play:** pick a video and open it in the player.
- **Diagnostics:** while a video plays, shows live fps, dropped frames, per-stage timings, a graph of the last
  minute, and which stage is the bottleneck. **System check** tests each part of the setup on its own, and
  **Copy report** puts it all on the clipboard for bug reports.

The panel remembers your choices in `panel.json`. The steps below do the same things from the command line.

## 2. Train on your own videos

Use high-quality videos (1080p or 4K, little compression). They are downscaled to make the training input.
Put them in `videos/` next to `train.py` (subfolders are fine), or make `videos` a link to where you keep them:

```bash
ln -s /mnt/c/Users/you/Videos/training videos             # e.g. a Windows folder, seen from WSL
~/vsr-venv/bin/python train.py --out vsr.pt               # prints the dB gain over bilinear; Ctrl+C any time,
                                                          # a checkpoint is saved every 2000 steps
~/vsr-venv/bin/python export.py vsr.pt                    # -> vsr.onnx + vsr.ref
```

## 3. Build and play

```bash
cmake -B build -S player
cmake --build build -j
./build/open-player --selftest vsr.onnx      # TensorRT output vs PyTorch; should say PASS
./build/open-player movie.mkv --model vsr.onnx
```

The first time it plays a given resolution, it builds a TensorRT engine for it (up to a minute) and caches it
next to the model.

Like RTX Video Super Resolution, VSR only runs when the video is smaller than the screen. A 4K video on a 4K
screen plays as is.

Speed on an RTX 5080, 1080p → 4K, measured with `--stats`:

| Model (`train.py --size`) | Network time per frame | Plays smoothly |
|---|---|---|
| `standard` (default) | 23–25 ms | up to 30 fps |
| `fast` | 9–11 ms | 57–58 fps on WSL, which copies each frame through the CPU (~3 ms); native Linux skips that copy, so 60 is expected there (not yet measured) |

| Key | Action |
|---|---|
| Space | pause |
| ← / → | seek 5 s |
| ↑ / ↓ | volume |
| V | VSR on/off (compare) |
| F, double-click | fullscreen |
| click bottom edge | seek |
| Q | quit |

### Diagnostics

```bash
./build/open-player --check vsr.onnx                     # PASS/WARN/FAIL for CUDA, NVDEC, Vulkan, audio, TensorRT, model
./build/open-player movie.mkv --model vsr.onnx --stats   # one JSON line of timings per second
```

After changing the player, run `bash tests/smoke.sh vsr.onnx`: it plays generated clips (needs a display)
and runs the model self-test.

Each `--stats` line ends with a `verdict`: `ok`, or the stage that cost the most time in a second where frames
were dropped (decode, upload, convert, vsr, display copy, present).

[upscale.py](upscale.py) does the same upscaling without the player (`python upscale.py in.mp4 -o out.mp4`).

## Limits

- SDR only. HDR video plays with washed-out colors.
- On WSL, CUDA and Vulkan can't share memory, so each frame goes through the CPU. Native Linux is zero-copy.
- Training uses a clean bicubic downscale, so compression artifacts in streamed video get sharpened as well.
