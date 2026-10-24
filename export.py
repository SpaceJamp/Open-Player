"""Export a trained checkpoint to ONNX (fp16) for the C++ player, plus a reference file it self-tests against.

    python export.py vsr.pt            # -> vsr.onnx, vsr.ref
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from model import load

REF_H, REF_W = 64, 96  # small fixed size the player's --selftest builds an engine for


class HROnly(torch.nn.Module):
    """The player only needs the upscaled frame, not the flow."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, cur, prev, prev_hr):
        return self.m(cur, prev, prev_hr)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", nargs="?", default="vsr.pt")
    ap.add_argument("--fp32", action="store_true", help="export fp32 instead of fp16 (for checking on CPU)")
    a = ap.parse_args()
    out = Path(a.weights).with_suffix(".onnx")

    m = load(a.weights, "cpu").float()
    s, dt = m.scale, torch.float32 if a.fp32 else torch.float16
    net = HROnly(m).eval()

    g = torch.Generator().manual_seed(0)
    cur, prev = torch.rand(1, 3, REF_H, REF_W, generator=g), torch.rand(1, 3, REF_H, REF_W, generator=g)
    prev_hr = torch.rand(1, 3, REF_H * s, REF_W * s, generator=g)
    with torch.no_grad():
        hr = net(cur, prev, prev_hr)  # fp32 reference, before converting the net
        up = m.init_state(cur)[1]     # how a fresh shot starts; the player's reset kernel must match it
    net.to(dt)

    h, w = torch.export.Dim("h", min=16, max=4320), torch.export.Dim("w", min=16, max=7680)
    torch.onnx.export(
        net, tuple(t.to(dt) for t in (cur, prev, prev_hr)), str(out), dynamo=True, external_data=False,
        input_names=["cur", "prev", "prev_hr"], output_names=["hr"],
        dynamic_shapes={"cur": {2: h, 3: w}, "prev": {2: h, 3: w}, "prev_hr": {2: s * h, 3: s * w}})

    # reference: fp32 torch results on fixed inputs; layout = int32 header [scale, h, w], then as fp32:
    # cur, prev, prev_hr (inputs), hr (network output), up (bilinear start of a new shot)
    ref = Path(a.weights).with_suffix(".ref")
    with open(ref, "wb") as f:
        np.array([s, REF_H, REF_W], np.int32).tofile(f)
        for t in (cur, prev, prev_hr, hr, up):
            t.numpy().astype(np.float32).tofile(f)
    print("wrote", out, ref)


if __name__ == "__main__":
    main()
