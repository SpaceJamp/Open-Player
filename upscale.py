"""Upscale a video file with a trained VSR checkpoint (to watch live, use the player instead).

    python upscale.py in.mp4 -o out.mp4
"""
import argparse, json, subprocess
from fractions import Fraction

import torch

from model import load

SCENE_CUT = 0.12  # mean abs pixel change that counts as a cut; resets the recurrence so old frames don't ghost in


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,avg_frame_rate",
         "-of", "json", path], capture_output=True, text=True, check=True).stdout
    st = json.loads(out)["streams"][0]
    fps = st["avg_frame_rate"]
    known = not fps.startswith("0/") and Fraction(fps)  # ffprobe says "0/0" when it doesn't know
    return st["width"], st["height"], fps if known else "30"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("-o", "--output", required=True, help="output file")
    ap.add_argument("-w", "--weights", default="vsr.pt")
    ap.add_argument("--crf", type=int, default=18, help="x264 quality for -o (lower = better)")
    a = ap.parse_args()

    model = load(a.weights).half()
    s = model.scale
    w, h, fps = probe(a.input)
    W, H = w * s, h * s

    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", a.input, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                           stdout=subprocess.PIPE)
    enc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", fps, "-i", "-",
         "-i", a.input, "-map", "0:v", "-map", "1:a?", "-c:a", "copy",
         # tag and convert as BT.709 so colors match HD sources instead of ffmpeg's BT.601 default
         "-vf", "scale=out_color_matrix=bt709", "-colorspace", "bt709", "-color_primaries", "bt709",
         "-color_trc", "bt709", "-c:v", "libx264", "-crf", str(a.crf), "-pix_fmt", "yuv420p", a.output],
        stdin=subprocess.PIPE)

    n, frame_bytes, prev = 0, w * h * 3, None
    try:
        with torch.inference_mode():
            while (buf := dec.stdout.read(frame_bytes)) and len(buf) == frame_bytes:
                cur = torch.frombuffer(bytearray(buf), dtype=torch.uint8).cuda().view(h, w, 3)
                cur = cur.permute(2, 0, 1)[None].half().div_(255)
                if prev is None or (cur - prev).abs().mean() > SCENE_CUT:
                    prev, prev_hr = model.init_state(cur)
                out, _ = model(cur, prev, prev_hr)
                prev, prev_hr = cur, out
                enc.stdin.write(out.clamp_(0, 1).mul_(255).round_()[0].permute(1, 2, 0).byte().cpu().numpy().tobytes())
                n += 1
    except BrokenPipeError:  # the encoder quit; its own error message says why
        pass
    finally:
        dec.kill()
        try:
            enc.stdin.close()
        except BrokenPipeError:
            pass
        enc.wait()
    print(f"{n} frames, {w}x{h} -> {W}x{H}")


if __name__ == "__main__":
    main()
