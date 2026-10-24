"""Train VSR on your own videos: high-res clips are downscaled on the fly to make low-res inputs.

    python train.py                 # videos from ./videos (subfolders included)
    python train.py ~/Movies        # or any other folder
"""
import argparse, json, math, random, signal, subprocess, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model import VSR, load, save, warp

VIDEO_EXT = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v", ".ts"}
# (channels, blocks). 1080p -> 4K on an RTX 5080: standard 23 ms per frame (up to 30 fps), fast 9.4 ms (60 fps)
SIZES = {"standard": (48, 6), "fast": (32, 4)}


def probe(path):
    """(width, height, duration_seconds) via ffprobe."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height:format=duration",
         "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    return j["streams"][0]["width"], j["streams"][0]["height"], float(j["format"]["duration"])


class Clips(Dataset):
    """Random runs of `frames` consecutive frames, randomly cropped to crop x crop, as uint8 (T,H,W,3)."""

    def __init__(self, root, frames, crop):
        self.frames, self.crop, self.videos = frames, crop, []
        for p in sorted(Path(root).expanduser().rglob("*")):
            if p.suffix.lower() not in VIDEO_EXT:
                continue
            try:
                w, h, dur = probe(p)
            except (subprocess.CalledProcessError, KeyError, ValueError, IndexError):
                print("skipping unreadable", p)
                continue
            if w >= crop and h >= crop and dur > 2:
                self.videos.append((p, w, h, dur))
        if not self.videos:
            raise SystemExit(f"no usable videos (>= {crop}px, > 2s) under {root}")
        print(f"{len(self.videos)} videos")

    def __len__(self):
        return 1 << 30  # infinite random sampling; training length is set by --steps

    def __getitem__(self, _):
        T, c = self.frames, self.crop
        for _ in range(10):  # retry on reads that come up short (end of file, broken streams)
            p, w, h, dur = random.choice(self.videos)
            t, x, y = random.uniform(0, dur - 1.5), random.randrange(w - c + 1), random.randrange(h - c + 1)
            raw = subprocess.run(
                ["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(p), "-frames:v", str(T),
                 "-vf", f"crop={c}:{c}:{x}:{y}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                capture_output=True).stdout
            if len(raw) == T * c * c * 3:
                return np.frombuffer(raw, np.uint8).reshape(T, c, c, 3)
        raise RuntimeError("could not read a full clip in 10 tries")


def psnr(a, b):
    return -10 * math.log10(F.mse_loss(a.float(), b.float()).item() + 1e-10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="?", default=str(Path(__file__).parent / "videos"),
                    help="folder of high-quality videos, searched recursively (default: videos/ next to this script)")
    ap.add_argument("--out", default="vsr.pt")
    ap.add_argument("--resume", help="checkpoint to continue from")
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--size", choices=SIZES, help="standard (best quality, up to 1080p30 -> 4K) or fast "
                    "(1080p60 -> 4K); default standard. A resumed model keeps its size.")
    ap.add_argument("--frames", type=int, default=8, help="frames per clip the recurrence is trained over")
    ap.add_argument("--crop", type=int, default=256, help="high-res crop size")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.crop % (a.scale * 4):
        raise SystemExit(f"--crop must be a multiple of {a.scale * 4}")

    dev = "cuda"
    if a.resume:
        model = load(a.resume, dev).train()
        have = next((k for k, v in SIZES.items() if v == (model.ch, model.blocks)), f"{model.ch}x{model.blocks}")
        if a.size and a.size != have:
            raise SystemExit(f"{a.resume} is a {have} model; rename or delete it to train a {a.size} one")
    else:
        model = VSR(a.scale, *SIZES[a.size or "standard"]).to(dev)
    s = model.scale
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps, eta_min=a.lr / 20)
    data = DataLoader(Clips(a.videos, a.frames, a.crop), batch_size=a.batch, num_workers=a.workers,
                      persistent_workers=a.workers > 0, pin_memory=True)

    stop = []  # Ctrl+C / the panel's Stop: finish this step, save, exit
    signal.signal(signal.SIGINT, lambda *_: stop.append(1))
    t0, it = time.time(), iter(data)
    for step in range(1, a.steps + 1):
        hr = next(it).to(dev, non_blocking=True).permute(0, 1, 4, 2, 3).float().div_(255)  # (B,T,3,H,W)
        B, T = hr.shape[:2]
        # shortcut: clean bicubic downscale only, so compression artifacts in real input get sharpened too;
        # add codec/noise degradations here if outputs look crunchy on streamed video
        lr = F.interpolate(hr.flatten(0, 1), scale_factor=1 / s, mode="bicubic", antialias=True)
        lr = lr.clamp(0, 1).unflatten(0, (B, T))

        with torch.autocast("cuda", torch.bfloat16):
            prev, prev_hr = model.init_state(lr[:, 0])
            loss = 0
            for t in range(T):
                out, flow = model(lr[:, t], prev, prev_hr)
                loss = loss + F.l1_loss(out.float(), hr[:, t])
                if t:  # teach the flow net real motion: warped previous frame should match the current one
                    loss = loss + 0.1 * F.l1_loss(warp(prev.float(), flow.float()), lr[:, t])
                prev, prev_hr = lr[:, t], out
            loss = loss / T

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 100 == 0:
            bil = F.interpolate(lr[:, -1], scale_factor=s, mode="bilinear", align_corners=False)
            gain = psnr(out.clamp(0, 1), hr[:, -1]) - psnr(bil, hr[:, -1])
            print(f"step {step}  loss {loss.item():.4f}  +{gain:.2f} dB vs bilinear  {(time.time() - t0) / 100:.2f}s/step", flush=True)
            t0 = time.time()
        if step % 2000 == 0 or step == a.steps or stop:
            save(model, a.out, step=step)
        if stop:
            break
    print(f"saved {a.out} at step {step}")


if __name__ == "__main__":
    main()
