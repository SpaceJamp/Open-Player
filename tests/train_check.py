"""Checks train.py's input degradations and validation score. CPU only, no videos needed.

    python tests/train_check.py
"""
import math
import sys
from pathlib import Path

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

print(f"train checks passed (compression error q100 {near_lossless:.4f}, q30 {crushed:.4f}; "
      f"bilinear {same:+.2f} dB, untrained net {gain:+.2f} dB)")
