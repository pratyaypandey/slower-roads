"""Upscaled frame-vs-decoded comparison across one or more tokenizer checkpoints.

Loads real sim frames + each checkpoint (rebuilt via the registry so any variant
reloads exactly), decodes them, and writes a big nearest-neighbour-upscaled grid:
rows = [original, recon_ckptA, recon_ckptB, ...], columns = sampled frames. Nearest
upscale keeps the low-poly pixels crisp (matplotlib's default interpolation would
blur the very thing we're judging). CPU is fine — it decodes a handful of frames.

    python -m eval.upscale_compare --data data/seed1 \
        --ckpts checkpoints/fsq.pt:fsq,checkpoints/fsq_v2.pt:fsq_v2 \
        --frames 0,500,1000,1500,2000,2500 --scale 6 --out eval/tokenizer_ab/compare.png
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from PIL import Image, ImageDraw

from model.registry import load_tokenizer


def load_frames(data_dir, idxs):
    manifest = json.load(open(os.path.join(data_dir, "manifest.json")))
    samples = manifest["samples"]
    out = []
    for i in idxs:
        i = min(i, len(samples) - 1)
        arr = np.load(os.path.join(data_dir, samples[i]["frame"]))  # (3,64,64) float [0,1]
        out.append(torch.from_numpy(arr).float())
    return torch.stack(out)  # (N,3,64,64)


def to_img(t, scale):
    """(3,H,W) float [0,1] -> PIL upscaled ×scale, nearest."""
    a = (t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    im = Image.fromarray(a)
    return im.resize((im.width * scale, im.height * scale), Image.NEAREST)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/seed1")
    ap.add_argument("--ckpts", required=True, help="comma list of path:label")
    ap.add_argument("--frames", default="0,500,1000,1500,2000,2500")
    ap.add_argument("--scale", type=int, default=6)
    ap.add_argument("--out", default="eval/tokenizer_ab/compare.png")
    args = ap.parse_args()

    idxs = [int(x) for x in args.frames.split(",")]
    frames = load_frames(args.data, idxs)                       # (N,3,64,64)

    rows = [("original", frames)]
    for spec in args.ckpts.split(","):
        path, _, label = spec.partition(":")
        model, _ = load_tokenizer(path, map_location="cpu")
        model.eval()
        with torch.no_grad():
            recon, idx, _ = model(frames)
        usage = int(torch.unique(idx).numel())
        rows.append((label or os.path.basename(path), recon, usage, model.codebook_size))

    N, S = len(idxs), args.scale
    cell = 64 * S
    pad, lab_w, top = 6, 130, 22
    W = lab_w + N * (cell + pad) + pad
    H = top + len(rows) * (cell + pad) + pad
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for c, fi in enumerate(idxs):
        draw.text((lab_w + c * (cell + pad) + 4, 4), f"frame {fi}", fill=(0, 0, 0))
    for r, row in enumerate(rows):
        label, imgs = row[0], row[1]
        y = top + r * (cell + pad) + pad
        tag = label if len(row) == 2 else f"{label}\n{row[2]}/{row[3]} codes\n({100 * row[2] / row[3]:.0f}%)"
        draw.text((6, y + cell // 2 - 16), tag, fill=(0, 0, 0))
        for c in range(N):
            canvas.paste(to_img(imgs[c], S), (lab_w + c * (cell + pad) + pad, y))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    canvas.save(args.out)
    print(f"saved {args.out}  ({W}x{H})")


if __name__ == "__main__":
    main()
