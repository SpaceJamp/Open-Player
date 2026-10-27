"""Train VSR on your own videos: high-res clips are downscaled and degraded on the fly to make low-res inputs.

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


# Decoding 1080p limits training speed, so each decoded run of frames gives several crops. Each worker keeps two
# runs and alternates between them, so a batch still mixes scenes. 8 crops: 11.1 batches/s, 4: 5.8 (RTX 5080 box).
CROPS_PER_DECODE, RUNS = 8, 2


class Clips(Dataset):
    """Random runs of `frames` consecutive frames, randomly cropped to crop x crop, as uint8 (T,H,W,3)."""

    def __init__(self, root, frames, crop):
        self.frames, self.crop, self.videos = frames, crop, []
        self.runs, self.turn = [[None, 0] for _ in range(RUNS)], 0  # per worker: [decoded frames, crops left]
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
        slot = self.runs[self.turn]
        self.turn = (self.turn + 1) % RUNS
        if not slot[1]:
            slot[0], slot[1] = self.decode(), CROPS_PER_DECODE
        slot[1] -= 1
        run, c = slot[0], self.crop
        y, x = random.randrange(run.shape[1] - c + 1), random.randrange(run.shape[2] - c + 1)
        return run[:, y:y + c, x:x + c].copy()

    def decode(self):
        """`frames` consecutive full-size frames from a random spot in a random video, as uint8 (T,H,W,3)."""
        T = self.frames
        for _ in range(10):  # retry on reads that come up short (end of file, broken streams)
            p, w, h, dur = random.choice(self.videos)
            # start at the keyframe before t instead of decoding up to t exactly, then skip a few frames so runs
            # don't all open on a keyframe, which is sharper than the frames that follow it
            t, skip = random.uniform(0, dur - 1.5), random.randrange(12)
            raw = subprocess.run(
                ["ffmpeg", "-v", "error", "-noaccurate_seek", "-ss", f"{t:.3f}", "-i", str(p), "-frames:v",
                 str(skip + T), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
            if len(raw) == (skip + T) * w * h * 3:
                return np.frombuffer(raw, np.uint8).reshape(skip + T, h, w, 3)[skip:]
        raise RuntimeError("could not read a full clip in 10 tries")


# Orthonormal 8x8 DCT (the same transform JPEG uses) and JPEG's standard luminance table at quality 50
DCT = torch.tensor([[math.sqrt((1 if k == 0 else 2) / 8) * math.cos(math.pi * (2 * n + 1) * k / 16) for n in range(8)]
                    for k in range(8)])
JPEG_Y = torch.tensor([[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55],
                       [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62],
                       [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
                       [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]], dtype=torch.float32)


def blocky(x, quality):
    """JPEG-style compression of (N,3,h,w) in [0,1], h and w multiples of 8: quantised 8x8 DCT blocks give the
    blocking and ringing that real codecs leave."""
    N, C, h, w = x.shape
    D = DCT.to(x)
    q = (JPEG_Y.to(x) * (50 / quality if quality < 50 else 2 - quality / 50)).clamp(min=1) / 255
    b = x.reshape(N, C, h // 8, 8, w // 8, 8).transpose(3, 4)
    b = D.T @ (torch.round((D @ (b - 0.5) @ D.T) / q) * q) @ D + 0.5
    return b.transpose(3, 4).reshape(N, C, h, w).clamp(0, 1)


def blur(x, sigma):
    """Separable Gaussian blur of (N,C,h,w)."""
    r, C = max(1, math.ceil(2 * sigma)), x.shape[1]
    k = torch.exp(-torch.arange(-r, r + 1, device=x.device, dtype=x.dtype) ** 2 / (2 * sigma ** 2))
    k = k / k.sum()
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1).repeat(C, 1, 1, 1), groups=C)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1).repeat(C, 1, 1, 1), groups=C)


def make_lr(hr, s, g=None):
    """Clean high-res clips (B,T,3,H,W) -> the low-res input the net learns from. Real input is a compressed
    stream, not a clean downscale, so it gets random blur, grain and block compression (one setting per call,
    from generator g). The targets stay clean, so the net learns to remove these instead of sharpening them."""
    B, T = hr.shape[:2]
    x = F.interpolate(hr.flatten(0, 1), scale_factor=1 / s, mode="bicubic", antialias=True).clamp(0, 1)
    u = lambda: torch.rand((), generator=g).item()
    if u() < 0.5:
        x = blur(x, 0.3 + 1.2 * u())
    if u() < 0.5:
        x = x + torch.randn(x.shape, generator=g, device=x.device, dtype=x.dtype) * (0.02 * u())
    if u() < 0.5:
        x = blocky(x.clamp(0, 1), 30 + 60 * u())
    return x.clamp(0, 1).unflatten(0, (B, T))


def unroll(model, lr):
    """Run the recurrent net over clips (B,T,3,h,w); returns each frame's output and flow."""
    prev, prev_hr = model.init_state(lr[:, 0])
    outs, flows = [], []
    for t in range(lr.shape[1]):
        cur = lr[:, t].contiguous(memory_format=torch.channels_last)
        out, flow = model(cur, prev, prev_hr)
        outs.append(out), flows.append(flow)
        prev, prev_hr = cur, out
    return outs, flows


VAL_CLIPS = 32  # fixed clips scored every 1000 steps: a steady "still improving?" number


def make_val(clips, s):
    """VAL_CLIPS clips from the training videos with fixed crops and degradations, identical on every run."""
    saved, g = random.getstate(), torch.Generator().manual_seed(1234)
    random.seed(1234)
    hr, lr = [], []
    for _ in range(VAL_CLIPS):
        run, c = clips.decode(), clips.crop
        y, x = random.randrange(run.shape[1] - c + 1), random.randrange(run.shape[2] - c + 1)
        hr.append(torch.from_numpy(run[:, y:y + c, x:x + c].copy()).permute(0, 3, 1, 2))
        lr.append(make_lr(hr[-1][None].float() / 255, s, g)[0])
    random.setstate(saved)
    return torch.stack(lr).half(), torch.stack(hr)  # (n,T,3,h,w) fp16, (n,T,3,H,W) uint8


@torch.no_grad()
def validate(model, val_lr, val_hr, s):
    """Gain in dB over bilinear upscaling on the validation clips, all frames."""
    se_net = se_bil = 0.0
    for i in range(0, len(val_lr), 8):
        lr, hr = val_lr[i:i + 8].float(), val_hr[i:i + 8].float() / 255
        with torch.autocast("cuda", torch.bfloat16):
            outs, _ = unroll(model, lr)
        for t, out in enumerate(outs):
            bil = F.interpolate(lr[:, t], scale_factor=s, mode="bilinear", align_corners=False)
            se_net += F.mse_loss(out.float().clamp(0, 1), hr[:, t], reduction="sum").item()
            se_bil += F.mse_loss(bil, hr[:, t], reduction="sum").item()
    return 10 * math.log10(se_bil / se_net)


def lr_at(step, steps, peak):
    """Cosine from peak down to peak/20 over the whole run; from the step alone, so resuming continues exactly."""
    return peak / 20 + (peak - peak / 20) * (1 + math.cos(math.pi * step / steps)) / 2


def ignore_sigint(_):
    """DataLoader workers: let only the main process see Ctrl+C. Otherwise PyTorch's workers quit on it and
    the main process crashes ("worker exited unexpectedly") before it can save."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


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
    ap.add_argument("--steps", type=int, default=100_000, help="total steps; a resumed run continues up to this")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.crop % (a.scale * 8):  # the low-res crop must split into the 8x8 blocks of the compression step
        raise SystemExit(f"--crop must be a multiple of {a.scale * 8}")

    dev = "cuda"
    if a.resume:
        model = load(a.resume, dev).train()
        have = next((k for k, v in SIZES.items() if v == (model.ch, model.blocks)), f"{model.ch}x{model.blocks}")
        if a.size and a.size != have:
            raise SystemExit(f"{a.resume} is a {have} model; rename or delete it to train a {a.size} one")
    else:
        model = VSR(a.scale, *SIZES[a.size or "standard"]).to(dev)
    s = model.scale
    model = model.to(memory_format=torch.channels_last)  # fp16 convs run 1.8x faster in this layout (0.132 -> 0.074 s/step)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0)
    start = 0
    if a.resume:
        ck = torch.load(a.resume, map_location=dev)
        start = ck.get("step", 0)
        if "opt" in ck:  # checkpoints from before this was saved restart the optimizer
            opt.load_state_dict(ck["opt"])
        if start >= a.steps:
            raise SystemExit(f"{a.resume} already has {start} steps; raise --steps to train further")
        print(f"resuming at step {start}")
    clips = Clips(a.videos, a.frames, a.crop)
    val_lr, val_hr = (v.to(dev) for v in make_val(clips, s))
    print(f"step {start}  validation {validate(model, val_lr, val_hr, s):+.2f} dB vs bilinear "
          f"({VAL_CLIPS} fixed clips, degraded like the training input)", flush=True)
    data = DataLoader(clips, batch_size=a.batch, num_workers=a.workers,
                      persistent_workers=a.workers > 0, pin_memory=True, worker_init_fn=ignore_sigint)

    stop = []  # Ctrl+C / the panel's Stop: finish this step, save, exit
    signal.signal(signal.SIGINT, lambda *_: stop.append(1))
    t0, it = time.time(), iter(data)
    for step in range(start + 1, a.steps + 1):
        hr = next(it).to(dev, non_blocking=True).permute(0, 1, 4, 2, 3).float().div_(255)  # (B,T,3,H,W)
        T = hr.shape[1]
        lr = make_lr(hr, s)

        with torch.autocast("cuda", torch.bfloat16):
            outs, flows = unroll(model, lr)
            loss = sum(F.l1_loss(out.float(), hr[:, t]) for t, out in enumerate(outs))
            # teach the flow net real motion: the warped previous frame should match the current one
            loss = loss + 0.1 * sum(F.l1_loss(warp(lr[:, t - 1], flows[t].float()), lr[:, t]) for t in range(1, T))
            loss = loss / T

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for g in opt.param_groups:
            g["lr"] = lr_at(step, a.steps, a.lr)
        opt.step()

        if step % 100 == 0:
            print(f"step {step}  loss {loss.item():.4f}  {(time.time() - t0) / 100:.2f}s/step", flush=True)
            t0 = time.time()
        if step % 1000 == 0:
            print(f"step {step}  validation {validate(model, val_lr, val_hr, s):+.2f} dB vs bilinear", flush=True)
        if step % 2000 == 0 or step == a.steps or stop:
            save(model, a.out, step=step, opt=opt.state_dict())
        if stop:
            break
    print(f"saved {a.out} at step {step}")


if __name__ == "__main__":
    main()
