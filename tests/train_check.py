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
