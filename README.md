# VSR player

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

## 2. Train on your own videos

Use high-quality videos (1080p or 4K, little compression). They are downscaled to make the training input.

```bash
~/vsr-venv/bin/python train.py ~/Videos --out vsr.pt     # prints the dB gain over bilinear; Ctrl+C any time,
                                                          # a checkpoint is saved every 2000 steps
~/vsr-venv/bin/python export.py vsr.pt                    # -> vsr.onnx + vsr.ref
```

## 3. Build and play

```bash
cmake -B build -S player
cmake --build build -j
./build/vsrplayer --selftest vsr.onnx      # TensorRT output vs PyTorch; should say PASS
./build/vsrplayer movie.mkv --model vsr.onnx
```

The first time it plays a given resolution, it builds a TensorRT engine for it (up to a minute) and caches it
next to the model.

| Key | Action |
|---|---|
| Space | pause |
| ← / → | seek 5 s |
| ↑ / ↓ | volume |
| V | VSR on/off (compare) |
| F, double-click | fullscreen |
| click bottom edge | seek |
| Q | quit |

[upscale.py](upscale.py) does the same upscaling without the player (`python upscale.py in.mp4 -o out.mp4`).

## Limits

- SDR only. HDR video plays with washed-out colors.
- On WSL, CUDA and Vulkan can't share memory, so each frame goes through the CPU. Native Linux is zero-copy.
- Training uses a clean bicubic downscale, so compression artifacts in streamed video get sharpened as well.
