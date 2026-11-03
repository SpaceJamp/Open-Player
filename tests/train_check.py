"""Checks train.py's input degradations and validation score. CPU only, no videos needed.

    python tests/train_check.py
"""
import math
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import train  # noqa: E402
from model import VSR  # noqa: E402

torch.manual_seed(0)
x = torch.rand(2, 3, 64, 96)

# the DCT is orthonormal, so compressing and decompressing loses only what quantisation throws away
assert torch.allclose(train.DCT @ train.DCT.T, torch.eye(8), atol=1e-6)
near_lossless = (train.blocky(x, 100) - x).abs().max().item()
assert near_lossless < 0.02, near_lossless
crushed = (train.blocky(x, 30) - x).abs().mean().item()
assert crushed > 0.02, crushed

# blur keeps a flat image flat and keeps the shape
flat = torch.full((1, 3, 32, 32), 0.4)
assert torch.allclose(train.blur(flat, 1.2), flat, atol=1e-6)
assert train.blur(x, 0.7).shape == x.shape

# frequency loss: 0 for identical images, positive for a blurred copy, and gradients reach the prediction
assert train.fft_loss(x, x).item() == 0
soft = train.blur(x, 1.5)
assert train.fft_loss(soft, x).item() > 0
p = soft.clone().requires_grad_()
train.fft_loss(p, x).backward()
assert p.grad is not None and p.grad.abs().sum() > 0
assert train.fft_loss(x.bfloat16(), x).dtype == torch.float32  # computed in fp32 even for autocast outputs

# degraded input: right size and range, and the same from the same seed (the validation set relies on that)
hr = torch.rand(2, 4, 3, 64, 64)
a = train.make_lr(hr, 2, torch.Generator().manual_seed(7))
b = train.make_lr(hr, 2, torch.Generator().manual_seed(7))
assert a.shape == (2, 4, 3, 32, 32) and a.min() >= 0 and a.max() <= 1
assert torch.equal(a, b)
# each clip in a batch gets its own damage, shared by its frames: same clip twice -> different inputs
same_clip = hr[:1].repeat(2, 1, 1, 1, 1)
two = train.make_lr(same_clip, 2, torch.Generator().manual_seed(3))
assert not torch.equal(two[0], two[1])

# validation score: a stand-in that outputs exactly the bilinear upscale must score exactly 0 dB
class Bilinear:
    def init_state(self, lr):
        return lr, None

    def __call__(self, cur, prev, prev_hr):
        return torch.nn.functional.interpolate(cur, scale_factor=2, mode="bilinear", align_corners=False), None


val_hr = (torch.rand(3, 4, 3, 64, 64) * 255).byte()
val_lr = train.make_lr(val_hr.float() / 255, 2, torch.Generator().manual_seed(1)).half()
same = train.validate(Bilinear(), val_lr, val_hr, 2)
assert abs(same) < 1e-6, same
gain = train.validate(VSR(2, 16, 2).to(memory_format=torch.channels_last), val_lr, val_hr, 2)  # real net runs
assert math.isfinite(gain), gain

# validation set: made once from held-out videos, reused unchanged after a video is added, refused if --crop changes
class FakeClips:
    def __init__(self, names, crop=64):
        self.root, self.frames, self.crop = Path("/videos"), 2, crop
        self.videos = [(self.root / n, 128, 128, 60.0) for n in names]

    def decode(self):
        return np.random.randint(0, 256, (self.frames, 128, 128, 3), np.uint8)


with tempfile.TemporaryDirectory() as d:
    path = Path(d) / "vsr.val.pt"
    first = FakeClips(["a.mp4", "b.mp4", "c.mp4", "sub/d.mp4"])
    lr1, hr1, made = train.validation_set(first, 2, path)
    held = torch.load(path)["held"]
    assert made and len(held) == train.VAL_VIDEOS and lr1.shape[0] == train.VAL_CLIPS
    assert not {v[0].relative_to(first.root).as_posix() for v in first.videos} & set(held), "held-out videos left in"
    again = FakeClips(["a.mp4", "b.mp4", "c.mp4", "sub/d.mp4", "new.mp4"])  # a video added between runs
    lr2, hr2, made = train.validation_set(again, 2, path)
    assert not made and torch.equal(lr1, lr2) and torch.equal(hr1, hr2)
    assert len(again.videos) == 5 - train.VAL_VIDEOS and any(v[0].name == "new.mp4" for v in again.videos)
    try:
        train.validation_set(FakeClips(["a.mp4", "b.mp4"], crop=128), 2, path)
        raise AssertionError("a different --crop must be refused")
    except SystemExit:
        pass

# distillation: W=0 is the plain loss; the teacher gets no gradient and does not change; a tiny run with --teacher works
dh, dl = torch.rand(2, 3, 3, 32, 32), torch.rand(2, 3, 3, 16, 16)
student, teacher = VSR(2, 8, 1), VSR(2, 8, 1)
souts, _ = train.unroll(student, dl)
touts = train.teacher_outputs(teacher, dl)
plain = train.pixel_loss(souts, dh)
assert torch.equal(train.pixel_loss(souts, dh, touts, 0.0), plain), "W=0 must equal the plain loss"
assert not torch.equal(train.pixel_loss(souts, dh, touts, 0.5), plain)
before = [p.clone() for p in teacher.parameters()]
opt = torch.optim.SGD(student.parameters(), lr=0.1)
train.pixel_loss(souts, dh, touts, 1.0).backward()
opt.step()
assert all(p.grad is None for p in teacher.parameters()) and not any(t.requires_grad for t in touts)
assert all(torch.equal(a, b) for a, b in zip(before, teacher.parameters())), "teacher changed"

if torch.cuda.is_available() and shutil.which("ffmpeg"):  # train.py trains on cuda only
    with tempfile.TemporaryDirectory() as d:
        for i in range(3):
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=96x96:rate=25", "-t", "3",
                            "-pix_fmt", "yuv420p", f"{d}/v{i}.mp4"], check=True)
        from model import save
        save(VSR(2, 8, 2, False), f"{d}/teacher.pt")
        cmd = [sys.executable, str(Path(train.__file__)), d, "--out", f"{d}/s.pt", "--teacher", f"{d}/teacher.pt",
               "--distill", "0.5", "--size", "fast", "--crop", "32", "--frames", "2", "--batch", "2", "--steps", "100",
               "--workers", "0"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        assert r.returncode == 0 and "distill" in r.stdout, r.stdout + r.stderr
        ck = torch.load(f"{d}/s.pt")
        assert ck["teacher"] == f"{d}/teacher.pt" and ck["distill"] == 0.5
    distilled = "distill run ok"
else:
    distilled = "distill run skipped (no cuda/ffmpeg)"
# diagnostics (train_diag.py): metrics with known answers, suggestions engine, and a diag.jsonl from a tiny run
import json
import train_diag as td  # noqa: E402

gt = torch.zeros(1, 3, 3, 16, 16)
gt[..., 8:] = 1.0  # vertical step edge, constant over time
assert torch.allclose(td.ssim(gt, gt), torch.ones(1))
m = td.edge_mask(gt)
assert m[..., 7:9].all() and not m[..., :5].any() and not m[..., 11:].any(), "edge mask must hug the step"
noisy = gt + 0.1 * m + 0.01 * ~m  # error 0.1 on edges, 0.01 on flat areas
se_e, n_e, se_f, n_f = td.edge_flat_se(noisy, gt)
assert abs(td.db(se_e, n_e).item() - 20) < 1e-3 and abs(td.db(se_f, n_f).item() - 40) < 1e-3
flick = gt[:, :1].repeat(1, 4, 1, 1, 1) * 0 + 0.5
flick_out = flick.clone()
flick_out[:, 1::2] += 0.1  # output alternates 0.5/0.6, ground truth is steady
assert abs(td.flicker(flick_out, flick).item() - 0.1) < 1e-6
assert abs(td.flicker(flick, flick).item()) < 1e-9
assert td.flicker(flick, flick_out).item() < -0.09  # smoother than the truth is negative


def win(**kw):
    return {"type": "window", "step": 100, "steps_per_s": 5.0, "wait_frac": 0.02, "gpu_frac": 0.95, "mem_gb": 4.0,
            "lr": 1e-4, "gnorm_mean": 1.0, "gnorm_max": 1.5, "loss": 0.05, "nonfinite": 0, **kw}


def val(gain=3.0, **kw):
    return {"type": "val", "step": 1000, "gain": gain, "ssim": 0.9, "flicker": 0.0, "edge_db": 25.0, "flat_db": 30.0,
            "train_gain": gain + 0.3, "since_best": 0, "clips": [{"name": f"clip {i}", "gain": gain} for i in range(6)], **kw}


healthy = [win(), win(), val(2.0), val(2.5), val(3.0)]
assert td.suggest(healthy) == ["ok"], td.suggest(healthy)
has = lambda recs, word: any(word in x for x in td.suggest(recs))
assert has([win(wait_frac=0.4), win(wait_frac=0.4)], "data-bound")
assert has(healthy + [val(3.0, since_best=td.PLATEAU_EVALS)], "plateau")
assert has([val(3.0, train_gain=3.0 + td.OVERFIT_DB + 0.5)], "overfitting")
assert has([win(), win(), win(gnorm_max=50.0)], "unstable")
assert has([win(nonfinite=2)], "unstable")
bad = val(3.0)
bad["clips"][2] = {"name": "movie.mp4", "gain": 3.0 - td.WEAK_CLIP_DB - 1}
assert has([bad], "movie.mp4") and has([bad], "weak content")
assert has([val(3.0, flicker=td.FLICKER_MAX * 2)], "flicker")
assert has([val(3.0, edge_db=10.0, flat_db=10.0 + td.EDGE_LAG_DB + 1)], "edges lag")
assert td.suggest([]) == ["ok"]

with tempfile.TemporaryDirectory() as d:
    dpath = Path(d) / "t.diag.jsonl"
    dg = td.Diag(dpath, window=2)
    tiny = VSR(2, 8, 1).to(memory_format=torch.channels_last)
    vhr = (torch.rand(2, 3, 3, 32, 32) * 255).byte()
    vlr = train.make_lr(vhr.float() / 255, 2, torch.Generator().manual_seed(1)).half()
    opt = torch.optim.SGD(tiny.parameters(), lr=1e-4)
    for step in range(1, 5):
        h = dg.next(iter([torch.rand(2, 3, 3, 32, 32)]))
        l = train.make_lr(h, 2)
        dg.got_batch(h, l)
        outs, _ = train.unroll(tiny, l)
        loss = sum(o.float().sub(h[:, t]).abs().mean() for t, o in enumerate(outs))
        opt.zero_grad(), loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(tiny.parameters(), 1.0)
        opt.step()
        dg.end(step, loss, gn, 1e-4)
    line = dg.validation(tiny, vlr, vhr, 2, 4, train.unroll)
    assert line.startswith("suggest: "), line
    recs = [json.loads(x) for x in dpath.read_text().splitlines()]
    assert [r["type"] for r in recs] == ["window", "window", "val"], recs
    assert recs[0]["step"] == 2 and recs[0]["gnorm_max"] >= recs[0]["gnorm_mean"] > 0 and math.isfinite(recs[0]["loss"])
    v = recs[2]
    assert len(v["clips"]) == 2 and v["clips"][0]["name"] == "clip 0" and math.isfinite(v["gain"]) and "train_gain" in v
    assert abs(v["gain"] - train.validate(tiny, vlr, vhr, 2)) < 1e-3, "diag score disagrees with train.validate"
    assert "suggest:" in td.summary(dpath) and "clip 0" in td.summary(dpath)

# a video tagged as rotated (as phones save them) must decode to its stored pixels, not turned and scrambled
if shutil.which("ffmpeg"):
    with tempfile.TemporaryDirectory() as d:
        ff = ["ffmpeg", "-v", "error", "-y"]
        subprocess.run(ff + ["-f", "lavfi", "-i", "testsrc2=size=320x192:rate=25", "-t", "3", "-c:v", "libx264",
                             "-pix_fmt", "yuv420p", f"{d}/plain.mp4"], check=True)
        Path(d, "rot").mkdir(), Path(d, "plain").mkdir()
        subprocess.run(ff + ["-display_rotation", "90", "-i", f"{d}/plain.mp4", "-c", "copy", f"{d}/rot/v.mp4"], check=True)
        shutil.copy(f"{d}/plain.mp4", f"{d}/plain/v.mp4")
        frames = []
        for sub in ("plain", "rot"):
            random.seed(5)  # same start time and skip for both
            frames.append(train.Clips(f"{d}/{sub}", 2, 64).decode())
        assert np.array_equal(*frames), "rotated video decoded differently from its unrotated twin"
    rotation = "rotated video ok"
else:
    rotation = "rotation check skipped (no ffmpeg)"

print(f"train checks passed ({rotation}; {distilled}; compression error q100 {near_lossless:.4f}, q30 {crushed:.4f}; "
      f"bilinear {same:+.2f} dB, untrained net {gain:+.2f} dB)")
