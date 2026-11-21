"""Train VSR on your own videos: high-res clips are downscaled and degraded on the fly to make low-res inputs.

    python train.py                 # videos from ./videos (subfolders included)
    python train.py ~/Movies        # or any other folder
"""
import argparse, copy, json, math, random, signal, subprocess, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model import VSR, load, save, warp
from train_diag import Diag

VIDEO_EXT = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v", ".ts"}
# (channels, blocks, attention gates). 1080p -> 4K on an RTX 5080: standard 23 ms per frame (up to 30 fps),
# fast 9.4 ms (60 fps). lean drops the gates, which cost a full pass over the activations, for 10 blocks: 48%
# more parameters at no more time (best of 3 builds: 25.0 ms vs standard's 28.3 in the same session)
SIZES = {"standard": (48, 6, True), "fast": (32, 4, True), "lean": (48, 10, False)}


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


def title_weights(paths, root):
    """Sampling weight per video so each title (a top-level file or folder under root, e.g. a whole series) is
    picked equally often: 8 episodes of one show otherwise outweigh a film 8 to 1."""
    key = [Path(p).relative_to(root).parts[0] for p in paths]
    return [1 / key.count(k) for k in key]


class Pairs(Dataset):
    """The player's --collect files: real decoded frames and NVIDIA RTX VSR's upscale of them (format in
    player/collect.h). Learning them teaches our model to imitate RTX VSR. Items are uint8 (T,3,h,w), (T,3,H,W)."""

    def __init__(self, folder, scale):
        """Keeps the whole files whose upscale is `scale`x and whose shape matches the first one (a batch needs one
        shape); `skipped` counts the rest."""
        self.scale, self.shape, self.files = scale, None, []
        found = sorted(Path(folder).expanduser().glob("pair-*.bin"))  # .tmp: still being written
        for f in found:
            try:
                shape = self.header(f.open("rb").read(28), f.stat().st_size)
            except OSError:
                continue
            self.shape = self.shape or shape
            if shape and shape == self.shape:
                self.files.append(f)
        self.skipped = len(found) - len(self.files)

    def header(self, head, size):
        """(T, lw, lh, hw, hh) of a whole, valid pair file at this scale, else None."""
        if len(head) < 28 or head[:8] != b"OPPAIR1\0":
            return None
        T, lw, lh, hw, hh = (int(v) for v in np.frombuffer(head, "<i4", 5, 8))
        ok = min(T, lw, lh) > 0 and (hw, hh) == (lw * self.scale, lh * self.scale)
        return (T, lw, lh, hw, hh) if ok and size == 28 + T * 3 * (lw * lh + hw * hh) else None

    def __len__(self):
        return 1 << 30  # random sampling, like Clips

    def __getitem__(self, _):
        for _ in range(100):  # a file deleted or replaced while training runs is skipped
            try:
                raw = random.choice(self.files).read_bytes()
            except OSError:
                continue
            if self.header(raw[:28], len(raw)) != self.shape:
                continue
            T, lw, lh, hw, hh = self.shape
            n = T * 3 * lh * lw
            lr = np.frombuffer(raw, np.uint8, n, 28).reshape(T, 3, lh, lw)
            hr = np.frombuffer(raw, np.uint8, T * 3 * hh * hw, 28 + n).reshape(T, 3, hh, hw)
            return torch.from_numpy(lr.copy()), torch.from_numpy(hr.copy())
        raise RuntimeError("--pairs: no readable pair files left")


class Clips(Dataset):
    """Random runs of `frames` consecutive frames, randomly cropped to crop x crop, as uint8 (T,H,W,3)."""

    def __init__(self, root, frames, crop):
        self.frames, self.crop, self.videos = frames, crop, []
        self.runs, self.turn = [[None, 0] for _ in range(RUNS)], 0  # per worker: [decoded frames, crops left]
        self.root = Path(root).expanduser()
        for p in sorted(self.root.rglob("*")):
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
            if getattr(self, "_weighted", None) is not self.videos:  # the list is replaced (held-out), never edited
                self._weighted, self._weights = self.videos, title_weights([v[0] for v in self.videos], self.root)
            p, w, h, dur = random.choices(self.videos, self._weights)[0]
            # start at the keyframe before t instead of decoding up to t exactly, then skip a few frames so runs
            # don't all open on a keyframe, which is sharper than the frames that follow it
            t, skip = random.uniform(0, dur - 1.5), random.randrange(12)
            raw = subprocess.run(
                # -noautorotate: frames keep the stored size probe() reported. Phone videos tagged as rotated
                # otherwise come out turned, and reshaping them to the stored size scrambles them silently.
                ["ffmpeg", "-v", "error", "-noautorotate", "-noaccurate_seek", "-ss", f"{t:.3f}", "-i", str(p), "-frames:v",
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
    stream, not a clean downscale, so it gets random blur, grain and block compression (one setting per clip,
    shared by its frames like a video's; from generator g). The targets stay clean, so the net learns to remove
    these instead of sharpening them."""
    B, T = hr.shape[:2]
    x = F.interpolate(hr.flatten(0, 1), scale_factor=1 / s, mode="bicubic", antialias=True).clamp(0, 1)
    u = lambda: torch.rand((), generator=g).item()
    clips = []
    for c in x.unflatten(0, (B, T)):
        if u() < 0.5:
            c = blur(c, 0.3 + 1.2 * u())
        if u() < 0.5:
            c = c + torch.randn(c.shape, generator=g, device=c.device, dtype=c.dtype) * (0.02 * u())
        if u() < 0.5:
            c = blocky(c.clamp(0, 1), 30 + 60 * u())
        clips.append(c.clamp(0, 1))
    return torch.stack(clips)


def fft_loss(pred, target):
    """Mean magnitude of the spectrum difference, in float32 even under autocast. Pixel losses reward blur; this
    punishes missing high frequencies."""
    # ortho: the same per-pixel error gives the same loss at any crop size, on the scale of the pixel loss
    return (torch.fft.rfft2(pred.float(), norm="ortho") - torch.fft.rfft2(target.float(), norm="ortho")).abs().mean()


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


@torch.no_grad()
def teacher_outputs(teacher, lr):
    """The teacher's frames for the same input clips, from its own recurrent state."""
    return unroll(teacher, lr)[0]


def pixel_loss(outs, hr, touts=None, w=0.0):
    """Summed L1 of the student's frames to the ground truth, or with a teacher: (1-w) of that plus w of the same
    against the teacher's frames."""
    gt = sum(F.l1_loss(out.float(), hr[:, t]) for t, out in enumerate(outs))
    if touts is None:
        return gt
    return (1 - w) * gt + w * sum(F.l1_loss(out.float(), touts[t].float()) for t, out in enumerate(outs))


VAL_CLIPS, VAL_VIDEOS = 32, 2  # fixed clips scored every 1000 steps, cut from videos held out of training


def make_val(clips, s):
    """VAL_CLIPS clips from `clips`' videos with fixed crops and degradations."""
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


def validation_set(clips, s, path):
    """The validation clips, made once and saved to `path`, then reused, so scores stay comparable across runs:
    picking them afresh changed them whenever a video was added (one model scored +2.59 dB, then +4.11).
    They come from VAL_VIDEOS videos held out of training, so the score measures unseen footage; this removes
    those videos from `clips`. Returns (val_lr, val_hr, made_now)."""
    rel = lambda p: p.relative_to(clips.root).as_posix()
    made = not path.exists()
    if made:
        names = sorted(rel(p) for p, *_ in clips.videos)
        held = random.Random(1234).sample(names, min(VAL_VIDEOS, len(names) - 1))  # keep one to train on
        vc = copy.copy(clips)
        vc.videos = [v for v in clips.videos if rel(v[0]) in held] or clips.videos  # a single video: share it
        lr, hr = make_val(vc, s)
        v = {"lr": lr, "hr": hr, "held": held, "frames": clips.frames, "crop": clips.crop, "scale": s}
        torch.save(v, path)
        print(f"validation set saved to {path}, from held-out videos {held or '(none: only one video)'}")
    else:
        v = torch.load(path)
        if (v["frames"], v["crop"], v["scale"]) != (clips.frames, clips.crop, s):
            raise SystemExit(f"{path} was made for --frames {v['frames']} --crop {v['crop']} and scale {v['scale']}: "
                             "use those, or delete it for a new validation set (scores then stop comparing)")
    clips.videos = [x for x in clips.videos if rel(x[0]) not in v["held"]]
    return v["lr"], v["hr"], made


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


def status_file():
    """training.json in the player's settings folder (next to settings.txt): its Settings > Training tab shows it."""
    import os, sys
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        base = Path(xdg)
    elif sys.platform == "win32" and os.environ.get("APPDATA"):
        base = Path(os.environ["APPDATA"])
    else:
        base = Path.home() / ".config"
    return base / "open-player" / "training.json"


def write_status(**fields):
    """Writes training.json whole (temp file + replace) for the player; never stops training on failure."""
    import os
    try:
        f = status_file()
        f.parent.mkdir(parents=True, exist_ok=True)
        clean = {k: (v if not isinstance(v, float) or math.isfinite(v) else None) for k, v in fields.items()}
        tmp = f.with_name(f.name + ".tmp")
        tmp.write_text(json.dumps({**clean, "updated": int(time.time())}, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, f)
    except OSError:
        pass


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
    ap.add_argument("--size", choices=SIZES, help="standard (up to 1080p30 -> 4K), fast (1080p60 -> 4K) or lean "
                    "(standard's speed, more capacity, no attention gates); default standard. A resumed model keeps its size.")
    ap.add_argument("--frames", type=int, default=8, help="frames per clip the recurrence is trained over")
    ap.add_argument("--crop", type=int, default=256, help="high-res crop size")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, help="total steps (default 100000; a resumed run keeps its own unless given)")
    ap.add_argument("--lr", type=float, help="peak learning rate (default 2e-4; a resumed run keeps its own)")
    ap.add_argument("--pairs", help="folder of the player's --collect files: also learn to match NVIDIA RTX VSR")
    ap.add_argument("--pairs-weight", type=float, metavar="W",
                    help="weight of the --pairs loss next to the usual one (default 0.5)")
    ap.add_argument("--fft-loss", type=float, default=0, metavar="W", help="weight of a frequency-domain loss "
                    "added to the pixel loss, for sharper output; 0 = off, try 0.05. A resumed run keeps its checkpoint's value.")
    ap.add_argument("--teacher", help="trained checkpoint (any size) whose output the student also learns to copy")
    ap.add_argument("--distill", type=float, help="weight 0..1 of the teacher's output in the pixel loss (default 0.5 with --teacher)")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.crop % (a.scale * 8):  # the low-res crop must split into the 8x8 blocks of the compression step
        raise SystemExit(f"--crop must be a multiple of {a.scale * 8}")

    dev = "cuda"
    ck = torch.load(a.resume, map_location=dev) if a.resume else {}
    # a resumed run keeps its schedule: a different --steps or --lr would make the cosine curve jump
    a.steps, a.lr = a.steps or ck.get("steps", 100_000), a.lr or ck.get("lr", 2e-4)
    torch.backends.cudnn.benchmark = True  # crops are a fixed size: time cuDNN's conv algorithms once, keep the fastest
    if a.resume:
        model = load(a.resume, dev).train()
        have = next((k for k, v in SIZES.items() if v == (model.ch, model.blocks, model.gate)),
                    f"{model.ch}x{model.blocks}{'' if model.gate else ' no-gate'}")
        if a.size and a.size != have:
            raise SystemExit(f"{a.resume} is a {have} model; rename or delete it to train a {a.size} one")
    else:
        model = VSR(a.scale, *SIZES[a.size or "standard"]).to(dev)
    s = model.scale
    model = model.to(memory_format=torch.channels_last)  # fp16 convs run 1.8x faster in this layout (0.132 -> 0.074 s/step)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0, fused=True)  # one kernel per step, same math
    start = 0
    if a.resume:
        start = ck.get("step", 0)
        a.fft_loss = a.fft_loss or ck.get("fft_loss", 0)
        a.pairs = a.pairs or ck.get("pairs")
        a.pairs_weight = a.pairs_weight if a.pairs_weight is not None else ck.get("pairs_weight")
        if "opt" in ck:  # checkpoints from before this was saved restart the optimizer
            opt.load_state_dict(ck["opt"])
        if start >= a.steps:
            raise SystemExit(f"{a.resume} already has {start} steps; raise --steps to train further")
        a.teacher, a.distill = a.teacher or ck.get("teacher"), a.distill if a.distill is not None else ck.get("distill")
        print(f"resuming at step {start}")
    if a.distill == 0:  # turns the teacher off, also on a resumed run
        a.teacher, a.distill = None, None
    if a.teacher and a.distill is None:
        a.distill = 0.5
    if a.teacher:
        a.teacher = str(Path(a.teacher).resolve())  # saved with the checkpoint: resuming may start elsewhere
    if a.pairs:
        a.pairs = str(Path(a.pairs).expanduser().resolve())
    if a.pairs_weight == 0:  # turns pairs off, also on a resumed run (like --distill 0)
        a.pairs = None
    if a.pairs_weight is not None and (a.pairs_weight < 0 or (a.pairs_weight and not a.pairs)):
        raise SystemExit("--pairs-weight needs --pairs and a weight of 0 or more")
    if a.pairs_weight is None:
        a.pairs_weight = 0.5
    if a.distill is not None and not (a.teacher and 0 <= a.distill <= 1):
        raise SystemExit("--distill needs --teacher and a weight between 0 and 1")
    teacher = None
    if a.teacher:
        teacher = load(a.teacher, dev).requires_grad_(False).to(memory_format=torch.channels_last)
        if teacher.scale != s:
            raise SystemExit(f"{a.teacher} upscales {teacher.scale}x, this model {s}x")
    clips = Clips(a.videos, a.frames, a.crop)
    out = Path(a.out)
    val_lr, val_hr, new_set = validation_set(clips, s, out.with_suffix(".val.pt"))
    val_lr, val_hr = val_lr.to(dev), val_hr.to(dev)
    # the best model so far is kept apart: vsr.pt is overwritten every 2000 steps. A new validation set can't be
    # compared with an old best score, so that starts over.
    best_path = out.with_name(out.stem + "_best.pt")
    best = -math.inf if new_set or not best_path.exists() else torch.load(best_path, map_location="cpu")["val_db"]

    dg = Diag(out.with_suffix(".diag.jsonl"), fresh=new_set)  # train_diag.py: <out>.diag.jsonl and `suggest:` lines

    status = {"state": "training", "steps": a.steps, "out": str(out.resolve())}  # the player's Training tab

    def check(step):
        nonlocal best
        score = validate(model, val_lr, val_hr, s)
        note = ""
        if score > best:
            best = score
            save(model, best_path, step=step, val_db=score)
            note = f"  (best so far, saved to {best_path.name})"
        print(f"step {step}  validation {score:+.2f} dB vs bilinear{note}", flush=True)
        suggest = dg.validation(model, val_lr, val_hr, s, step, unroll)
        print(suggest, flush=True)
        status.update(step=step, val_db=score, best_db=best, suggest=suggest.removeprefix("suggest: "))
        write_status(**status)

    check(start)
    data = DataLoader(clips, batch_size=a.batch, num_workers=a.workers,
                      persistent_workers=a.workers > 0, pin_memory=True, worker_init_fn=ignore_sigint)

    pair_it = None
    if a.pairs:
        pairs = Pairs(a.pairs, s)
        if not pairs.files:
            raise SystemExit(f"no usable {s}x pair-*.bin files in {a.pairs} (play with open-player --collect {a.pairs})")
        if pairs.skipped:
            print(f"skipping {pairs.skipped} pair files in {a.pairs}: damaged, not {s}x, or another crop size")
        pair_it = iter(DataLoader(pairs, batch_size=a.batch, num_workers=min(a.workers, 2),
                                  persistent_workers=a.workers > 0, pin_memory=True, worker_init_fn=ignore_sigint))
        print(f"{len(pairs.files)} RTX VSR pair files, weight {a.pairs_weight}")

    stop = []  # Ctrl+C / the panel's Stop: finish this step, save, exit
    signal.signal(signal.SIGINT, lambda *_: stop.append(1))
    t0, it = time.time(), iter(data)
    for step in range(start + 1, a.steps + 1):
        hr = dg.next(it).to(dev, non_blocking=True).permute(0, 1, 4, 2, 3).float().div_(255)  # (B,T,3,H,W)
        T = hr.shape[1]
        lr = make_lr(hr, s)
        dg.got_batch(hr, lr)

        with torch.autocast("cuda", torch.bfloat16):
            outs, flows = unroll(model, lr)
            touts = teacher_outputs(teacher, lr) if teacher else None
            loss = pixel_loss(outs, hr, touts, a.distill)
            if a.fft_loss:
                fft = sum(fft_loss(out, hr[:, t]) for t, out in enumerate(outs))
                loss = loss + a.fft_loss * fft
            if pair_it:  # real compressed frames -> NVIDIA RTX VSR's upscale of them
                plr, phr = (x.to(dev, non_blocking=True).float().div_(255) for x in next(pair_it))
                pouts, _ = unroll(model, plr)
                pl = sum(F.l1_loss(o.float(), phr[:, t]) for t, o in enumerate(pouts)) / len(pouts)
                loss = loss + a.pairs_weight * T * pl  # per frame, like the rest after the / T below
            # teach the flow net real motion: the warped previous frame should match the current one
            loss = loss + 0.1 * sum(F.l1_loss(warp(lr[:, t - 1], flows[t].float()), lr[:, t]) for t in range(1, T))
            loss = loss / T

        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for g in opt.param_groups:
            g["lr"] = lr_at(step, a.steps, a.lr)
        if torch.isfinite(gnorm):  # a NaN/inf step would poison the weights and Adam's state for good
            opt.step()
        dg.end(step, loss, gnorm, g["lr"])

        if step % 100 == 0:
            tl = f"  distill {F.l1_loss(outs[-1].float(), touts[-1].float()).item():.4f}" if teacher else ""
            fl = f"  fft {fft.item() / T:.4f}" if a.fft_loss else ""
            sps = (time.time() - t0) / 100
            print(f"step {step}  loss {loss.item():.4f}{tl}{fl}  {sps:.2f}s/step", flush=True)
            status.update(step=step, loss=loss.item(), s_per_step=sps, eta_s=(a.steps - step) * sps)
            write_status(**status)
            t0 = time.time()
        if step % 1000 == 0:
            check(step)
        if step % 2000 == 0 or step == a.steps or stop:
            save(model, a.out, step=step, opt=opt.state_dict(), fft_loss=a.fft_loss, teacher=a.teacher, distill=a.distill,
                 steps=a.steps, lr=a.lr, pairs=a.pairs, pairs_weight=a.pairs_weight)
        if stop:
            break
    status.update(state="stopped" if stop and step < a.steps else "finished", step=step, eta_s=0)
    write_status(**status)
    print(f"saved {a.out} at step {step}")


if __name__ == "__main__":
    main()
