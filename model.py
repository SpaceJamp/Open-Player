"""Recurrent video super-resolution net: coarse flow + warp, conv/attention body, pixel-shuffle out.

Each frame sees the current low-res frame plus the previous high-res output, warped onto the
current frame by a learned flow, so detail recovered in earlier frames carries forward.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

FLOW_POOL = 4  # flow runs at 1/4 of low-res size: cheap, and the receptive field covers bigger motion


def warp(x, flow):
    """Sample x at (pixel + flow). flow is (B,2,H,W) in pixels, channel 0 = x, 1 = y."""
    B, _, H, W = x.shape
    gy, gx = torch.meshgrid(torch.arange(H, device=x.device), torch.arange(W, device=x.device), indexing="ij")
    px = gx.to(x.dtype) + flow[:, 0]
    py = gy.to(x.dtype) + flow[:, 1]
    grid = torch.stack((2 * px / max(W - 1, 1) - 1, 2 * py / max(H - 1, 1) - 1), -1)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)


class Block(nn.Module):
    """Residual conv block gated by pixel attention (a per-pixel, per-channel sigmoid mask)."""

    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1), nn.LeakyReLU(0.1, True), nn.Conv2d(ch, ch, 3, padding=1))
        self.attn = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        y = self.conv(x)
        return x + y * torch.sigmoid(self.attn(y))


class VSR(nn.Module):
    def __init__(self, scale=2, ch=48, blocks=6):
        super().__init__()
        self.scale, self.ch, self.blocks = scale, ch, blocks
        act = lambda: nn.LeakyReLU(0.1, True)
        self.flow_net = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1), act(),
            nn.Conv2d(32, 32, 3, padding=1), act(),
            nn.Conv2d(32, 32, 3, padding=1), act(),
            nn.Conv2d(32, 2, 3, padding=1),
        )
        nn.init.zeros_(self.flow_net[-1].weight)  # start as identity warp
        nn.init.zeros_(self.flow_net[-1].bias)
        self.head = nn.Conv2d(3 + 3 * scale * scale, ch, 3, padding=1)
        self.body = nn.Sequential(*[Block(ch) for _ in range(blocks)], nn.Conv2d(ch, ch, 3, padding=1))
        self.tail = nn.Conv2d(ch, 3 * scale * scale, 3, padding=1)

    def coarse_flow(self, cur, prev):
        """Flow (in low-res pixels) that maps each pixel of cur to where it was in prev."""
        x = F.avg_pool2d(torch.cat([cur, prev], 1), FLOW_POOL, ceil_mode=True)
        return F.interpolate(self.flow_net(x), size=cur.shape[-2:], mode="bilinear", align_corners=False) * FLOW_POOL

    def init_state(self, lr):
        """(prev_lr, prev_hr) for the first frame of a shot."""
        return lr, F.interpolate(lr, scale_factor=self.scale, mode="bilinear", align_corners=False)

    def forward(self, cur, prev, prev_hr):
        """cur/prev: (B,3,h,w) in [0,1]; prev_hr: (B,3,sh,sw). Returns (hr, flow)."""
        s = self.scale
        flow = self.coarse_flow(cur, prev)
        hr_flow = F.interpolate(flow, scale_factor=s, mode="bilinear", align_corners=False) * s
        state = F.pixel_unshuffle(warp(prev_hr, hr_flow), s)
        x = self.head(torch.cat([cur, state], 1))
        x = x + self.body(x)
        up = F.interpolate(cur, scale_factor=s, mode="bilinear", align_corners=False)
        return F.pixel_shuffle(self.tail(x), s) + up, flow


def load(path, device="cuda"):
    ck = torch.load(path, map_location=device)
    m = VSR(ck["scale"], ck["ch"], ck["blocks"]).to(device)
    m.load_state_dict(ck["state"])
    return m.eval()


def save(model, path, **extra):
    torch.save({"scale": model.scale, "ch": model.ch, "blocks": model.blocks, "state": model.state_dict(), **extra}, path)


if __name__ == "__main__":  # self-check
    x = torch.rand(2, 3, 17, 23)
    assert torch.allclose(warp(x, torch.zeros(2, 2, 17, 23)), x, atol=1e-5), "zero flow must be identity"
    shifted = warp(x, torch.ones(2, 2, 17, 23))  # sample one pixel right/down
    assert torch.allclose(shifted[..., :-1, :-1], x[..., 1:, 1:], atol=1e-5), "unit flow must shift by one pixel"
    m = VSR(scale=2)
    prev, prev_hr = m.init_state(x)
    hr, flow = m(x, prev, prev_hr)
    assert hr.shape == (2, 3, 34, 46) and flow.shape == (2, 2, 17, 23), (hr.shape, flow.shape)
    print("ok", sum(p.numel() for p in m.parameters()), "params")
