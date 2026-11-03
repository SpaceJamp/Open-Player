"""Training diagnostics: what a run measures, and plain-English suggestions from it (like player/diag.h's verdict).

train.py calls Diag a few times per step; everything lands as JSON lines in <out>.diag.jsonl, one object per
window of steps ("type": "window") and one per validation ("type": "val"). Read it with

    python train_diag.py vsr.diag.jsonl
"""
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# thresholds behind the suggestions
DATA_WAIT_FRAC = 0.25   # share of wall time blocked on the loader (mean of the last 2 windows)
PLATEAU_EVALS = 5       # validations in a row without a new best
OVERFIT_DB = 3.0        # train-crop gain over bilinear minus validation gain
SPIKE_X = 10.0          # a window's max grad norm vs the median of recent windows' mean grad norm
WEAK_CLIP_DB = 2.0      # a validation clip's gain this far below the mean of all clips
FLICKER_MAX = 0.004     # mean |frame change| of output minus ground truth, on a 0..1 scale (about 1/255)
EDGE_LAG_DB = 12.0      # flat-region PSNR minus edge-region PSNR
EDGE_THRESH = 0.2       # Sobel magnitude of the ground truth (a one-pixel 0->1 step gives 4) that counts as edge
RECENT_CLIPS = 8        # training-crop clips kept (2 from each of the last 4 batches) for the overfitting gap


def luma(x):
    """(...,3,H,W) -> (...,1,H,W)"""
    return (x * x.new_tensor([0.299, 0.587, 0.114]).view(3, 1, 1)).sum(-3, keepdim=True)


def ssim(out, gt):
    """Mean SSIM per clip of (n,T,3,H,W) in [0,1]: 7x7 box windows on luma."""
    n = out.shape[0]
    a, b = luma(out).flatten(0, 1), luma(gt).flatten(0, 1)
    mu = lambda x: F.avg_pool2d(x, 7, 1)
    ma, mb = mu(a), mu(b)
    va, vb, cov = mu(a * a) - ma * ma, mu(b * b) - mb * mb, mu(a * b) - ma * mb
    s = ((2 * ma * mb + 0.01 ** 2) * (2 * cov + 0.03 ** 2)) / ((ma * ma + mb * mb + 0.01 ** 2) * (va + vb + 0.03 ** 2))
    return s.reshape(n, -1).mean(1)


def edge_mask(gt):
    """Bool (n,T,1,H,W): where the ground truth has a strong edge (Sobel magnitude on luma)."""
    y = luma(gt).flatten(0, 1)
    p = F.pad(y, (1, 1, 1, 1), mode="replicate")
    gx = p[..., :-2, 2:] + 2 * p[..., 1:-1, 2:] + p[..., 2:, 2:] - p[..., :-2, :-2] - 2 * p[..., 1:-1, :-2] - p[..., 2:, :-2]
    gy = p[..., 2:, :-2] + 2 * p[..., 2:, 1:-1] + p[..., 2:, 2:] - p[..., :-2, :-2] - 2 * p[..., :-2, 1:-1] - p[..., :-2, 2:]
    return (gx.hypot(gy) > EDGE_THRESH).unflatten(0, gt.shape[:2])


def edge_flat_se(out, gt):
    """Per clip: squared error summed over edge pixels, their count, and the same for flat pixels."""
    m = edge_mask(gt).expand_as(gt)
    se = (out - gt) ** 2
    d = (1, 2, 3, 4)
    return (se * m).sum(d), m.sum(d), (se * ~m).sum(d), (~m).sum(d)


def flicker(out, gt):
    """Per clip: mean |change between consecutive output frames| minus the same for the ground truth.
    Positive = the output flickers more than the real footage."""
    ch = lambda x: (x[:, 1:] - x[:, :-1]).abs().mean((1, 2, 3, 4))
    return ch(out) - ch(gt)


def db(se, n):
    """PSNR (peak 1) from a summed squared error over n values."""
    return 10 * torch.log10(1 / (se / n.clamp(min=1) + 1e-10))


@torch.no_grad()
def score(model, lr_all, hr_all, s, unroll, batch=8):
    """Metrics of the net on clips lr_all (n,T,3,h,w) against hr_all (n,T,3,H,W), uint8 or 0..1: the overall
    numbers plus per-clip lists."""
    names = "se_net se_bil ssim flick se_e n_e se_f n_f count".split()
    acc = {k: [] for k in names}
    dev = lr_all.device
    for i in range(0, len(lr_all), batch):
        lr, hr = lr_all[i:i + batch].float(), hr_all[i:i + batch].float()
        if hr_all.dtype == torch.uint8:
            hr = hr / 255
        with torch.autocast(dev.type, torch.bfloat16, enabled=dev.type == "cuda"):
            outs, _ = unroll(model, lr)
        out = torch.stack([o.float() for o in outs], 1).clamp(0, 1)
        bil = F.interpolate(lr.flatten(0, 1), scale_factor=s, mode="bilinear", align_corners=False).unflatten(0, lr.shape[:2])
        vals = [((out - hr) ** 2).sum((1, 2, 3, 4)), ((bil - hr) ** 2).sum((1, 2, 3, 4)), ssim(out, hr),
                flicker(out, hr), *edge_flat_se(out, hr), torch.full((len(out),), float(hr[0].numel()))]
        for k, v in zip(names, vals):
            acc[k].append(v.cpu())
    a = {k: torch.cat(v) for k, v in acc.items()}
    return {"gain": 10 * math.log10(a["se_bil"].sum() / a["se_net"].sum()),
            "clip_gains": (10 * torch.log10(a["se_bil"] / a["se_net"])).tolist(),
            "clip_db": db(a["se_net"], a["count"]).tolist(), "ssim": a["ssim"].mean().item(),
            "flicker": a["flick"].mean().item(), "edge_db": db(a["se_e"].sum(), a["n_e"].sum()).item(),
            "flat_db": db(a["se_f"].sum(), a["n_f"].sum()).item()}


def suggest(recs):
    """Short verdicts from the records so far (window and val dicts, oldest first); ["ok"] when healthy."""
    wins = [r for r in recs if r["type"] == "window"]
    vals = [r for r in recs if r["type"] == "val"]
    out = []
    if len(wins) >= 2:
        wait = statistics.mean(w["wait_frac"] for w in wins[-2:])
        if wait > DATA_WAIT_FRAC:
            out.append(f"data-bound: GPU idle {wait:.0%} waiting for clips - raise --workers or crops per decode")
    if wins:
        base = statistics.median(w["gnorm_mean"] for w in wins[-10:])
        spike = max(w["gnorm_max"] for w in wins[-3:])
        bad = sum(w["nonfinite"] for w in wins[-3:])
        if bad:
            out.append(f"unstable: {bad} non-finite loss/gradient steps - lower LR or clip grads")
        elif base > 0 and spike > SPIKE_X * base:
            out.append(f"unstable: grad-norm spikes {spike / base:.0f}x the median - lower LR or clip grads")
    if vals:
        v = vals[-1]
        if v["since_best"] >= PLATEAU_EVALS:
            out.append(f"plateau: no val gain in {v['since_best']} evals - lower LR or add varied data")
        if v.get("train_gain") is not None and v["train_gain"] - v["gain"] > OVERFIT_DB:
            out.append(f"overfitting: train gain exceeds val by {v['train_gain'] - v['gain']:.1f} dB - add data/augmentation")
        mean = statistics.mean(c["gain"] for c in v["clips"])
        for c in sorted((c for c in v["clips"] if c["gain"] < mean - WEAK_CLIP_DB), key=lambda c: c["gain"])[:2]:
            out.append(f"weak content: {c['name']} is {mean - c['gain']:.1f} dB below mean - add similar footage")
        if v["flicker"] > FLICKER_MAX:
            out.append(f"flicker: output changes {v['flicker']:.4f} more per frame than the source - consider temporal loss")
        if v["flat_db"] - v["edge_db"] > EDGE_LAG_DB:
            out.append(f"edges lag: edge PSNR {v['edge_db']:.1f} dB vs flat {v['flat_db']:.1f} dB - try --fft-loss")
    return out or ["ok"]


def read(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def summary(path):
    """Text for the CLI: latest window, latest validation with clips worst first, gain trend, suggestions."""
    recs = read(path)
    wins = [r for r in recs if r["type"] == "window"]
    vals = [r for r in recs if r["type"] == "val"]
    lines = []
    if wins:
        w = wins[-1]
        lines.append(f"latest window (step {w['step']}): {w['steps_per_s']:.2f} steps/s, data wait {w['wait_frac']:.0%}, "
                     f"gpu busy {w['gpu_frac']:.0%}, mem {w['mem_gb']:.1f} GB, lr {w['lr']:.2e}, "
                     f"grad norm mean {w['gnorm_mean']:.2f} max {w['gnorm_max']:.2f}, loss {w['loss']:.4f}, "
                     f"non-finite {w['nonfinite']}")
    if vals:
        v = vals[-1]
        tg = "n/a" if v["train_gain"] is None else f"{v['train_gain']:+.2f} dB"
        lines.append(f"latest validation (step {v['step']}): gain {v['gain']:+.2f} dB, ssim {v['ssim']:.4f}, "
                     f"flicker {v['flicker']:+.4f}, edge {v['edge_db']:.1f} dB / flat {v['flat_db']:.1f} dB, "
                     f"train gain {tg}, {v['since_best']} evals since best")
        lines.append("per-clip gain, worst first:")
        lines += [f"  {c['name']:<10} {c['gain']:+.2f} dB" for c in sorted(v["clips"], key=lambda c: c["gain"])]
        lines.append("val gain trend: " + " ".join(f"{x['gain']:+.2f}" for x in vals[-8:]))
    lines.append("suggest: " + " | ".join(suggest(recs)))
    return "\n".join(lines)


class Diag:
    """Collects per-step numbers with no GPU sync (events and tensors are read once per window) and writes records."""

    def __init__(self, path, window=100, fresh=False):
        self.path, self.window = Path(path), window
        if fresh:  # a new validation set: old scores no longer compare
            self.path.write_text("")
        self.recs = read(self.path) if self.path.exists() else []
        vals = [r for r in self.recs if r["type"] == "val"]
        self.best = max((r["gain"] for r in vals), default=-math.inf)
        self.since_best = vals[-1]["since_best"] if vals else 0
        self.cuda = torch.cuda.is_available()
        self.recent = []
        self._reset()

    def _reset(self):
        self.losses, self.gnorms, self.ev, self.wait = [], [], [], 0.0
        self.t0 = time.perf_counter()

    def next(self, it):
        """next(it), timing how long the loader kept us waiting."""
        t = time.perf_counter()
        batch = next(it)
        self.wait += time.perf_counter() - t
        return batch

    def got_batch(self, hr, lr):
        """Call once the batch is on the GPU and degraded; keeps a few clips for the train-vs-val comparison."""
        self.recent = (self.recent + [(hr[:2].half().clone(), lr[:2].half().clone())])[-RECENT_CLIPS // 2:]
        if self.cuda:
            self._s = torch.cuda.Event(enable_timing=True)
            self._s.record()
        else:
            self._s = time.perf_counter()

    def end(self, step, loss, gnorm, lr):
        """Call after opt.step(); gnorm is clip_grad_norm_'s return (the norm before clipping)."""
        if self.cuda:
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            self.ev.append((self._s, e))
        else:
            self.ev.append(time.perf_counter() - self._s)
        self.losses.append(loss.detach().float()), self.gnorms.append(gnorm.detach().float())
        if step % self.window:
            return
        if self.cuda:
            torch.cuda.synchronize()
            gpu = sum(s.elapsed_time(e) for s, e in self.ev) / 1000
            mem = torch.cuda.max_memory_allocated() / 2 ** 30
            torch.cuda.reset_peak_memory_stats()
        else:
            gpu, mem = sum(self.ev), 0.0
        wall, n = time.perf_counter() - self.t0, len(self.losses)
        g, l = torch.stack(self.gnorms).cpu(), torch.stack(self.losses).cpu()
        fin = torch.isfinite(g) & torch.isfinite(l)
        gf, lf = g[fin], l[fin]
        nan = float("nan")
        self._write({"type": "window", "step": step, "steps_per_s": n / wall, "wait_frac": self.wait / wall,
                     "gpu_frac": min(1.0, gpu / wall), "mem_gb": mem, "lr": lr,
                     "gnorm_mean": gf.mean().item() if len(gf) else nan, "gnorm_max": gf.max().item() if len(gf) else nan,
                     "loss": lf.mean().item() if len(lf) else nan, "nonfinite": int((~fin).sum())})
        self._reset()

    def validation(self, model, val_lr, val_hr, s, step, unroll):
        """Score the validation set and the recent training crops; record it; return the `suggest:` line."""
        v = score(model, val_lr, val_hr, s, unroll)
        train_gain = None
        if self.recent:
            train_gain = score(model, torch.cat([l for _, l in self.recent]), torch.cat([h for h, _ in self.recent]),
                               s, unroll)["gain"]
        if v["gain"] > self.best:
            self.best, self.since_best = v["gain"], 0
        else:
            self.since_best += 1
        self._write({"type": "val", "step": step, "gain": v["gain"], "ssim": v["ssim"], "flicker": v["flicker"],
                     "edge_db": v["edge_db"], "flat_db": v["flat_db"], "train_gain": train_gain,
                     "since_best": self.since_best,
                     "clips": [{"name": f"clip {i}", "gain": g, "psnr": p}  # the val set doesn't record sources
                               for i, (g, p) in enumerate(zip(v["clip_gains"], v["clip_db"]))]})
        return "suggest: " + " | ".join(suggest(self.recs))

    def _write(self, rec):
        self.recs.append(rec)
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    print(summary(sys.argv[1]))
