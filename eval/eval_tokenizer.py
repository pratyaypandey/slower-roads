"""Show original vs. FSQ-reconstructed frames side by side, so a low recon loss
can be trusted (or not) by eye — a tokenizer can score well by nailing the easy
sky/ground split while smearing the road, which the loss alone won't reveal.

    CUDA_VISIBLE_DEVICES=7 python -m eval.eval_tokenizer \
        --data data/seed1_drive --ckpt checkpoints/tokenizer.pt

Writes eval/plots/tokenizer_recon.png (a grid: top row originals, bottom row
reconstructions) if matplotlib/PIL is available; always prints per-frame L1 and
a codebook-usage number (how many of the 12800 codes the encoder actually uses —
low usage means the tokenizer collapsed to a few codes).
"""

import argparse
import json
import os

import numpy as np
import torch

from model.registry import load_tokenizer


def load_frames(data_dir, idxs):
    manifest = json.load(open(os.path.join(data_dir, "manifest.json")))
    samples = manifest["samples"]
    frames = []
    for i in idxs:
        arr = np.load(os.path.join(data_dir, samples[i]["frame"]))
        frames.append(arr)
    return np.stack(frames)  # (N,3,H,W)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/seed1")
    p.add_argument("--ckpt", default="checkpoints/tokenizer.pt")
    p.add_argument("--n", type=int, default=6, help="frames to show")
    p.add_argument("--out", default="eval/plots")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_tokenizer(args.ckpt, default_cfg={"hidden": 64}, map_location=device)
    model = model.to(device).eval()

    manifest = json.load(open(os.path.join(args.data, "manifest.json")))
    total = len(manifest["samples"])
    idxs = np.linspace(0, total - 1, args.n).astype(int)
    frames = torch.from_numpy(load_frames(args.data, idxs)).float().to(device)

    with torch.no_grad():
        recon, indices, _ = model(frames)

    l1 = torch.abs(recon - frames).mean(dim=(1, 2, 3))
    print("per-frame L1:", "  ".join(f"{v:.4f}" for v in l1.tolist()))

    # Codebook usage over a big sample: how many distinct codes appear? Low usage
    # (a handful of the 12800) means the tokenizer collapsed and isn't really
    # using its capacity — a failure the recon loss can hide on easy images.
    big = torch.from_numpy(load_frames(args.data,
        np.linspace(0, total - 1, min(200, total)).astype(int))).float().to(device)
    with torch.no_grad():
        _, big_idx, _ = model(big)
    used = torch.unique(big_idx).numel()
    print(f"codebook usage: {used} / {model.fsq.codebook_size} distinct codes "
          f"({100 * used / model.fsq.codebook_size:.1f}%)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(args.out, exist_ok=True)
        n = len(idxs)
        fig, ax = plt.subplots(2, n, figsize=(2 * n, 4.2))
        for j in range(n):
            ax[0, j].imshow(frames[j].cpu().permute(1, 2, 0).clamp(0, 1))
            ax[0, j].set_title(f"orig {idxs[j]}", fontsize=8)
            ax[1, j].imshow(recon[j].cpu().permute(1, 2, 0).clamp(0, 1))
            ax[1, j].set_title(f"recon L1={l1[j]:.3f}", fontsize=8)
            for a in (ax[0, j], ax[1, j]):
                a.set_xticks([]); a.set_yticks([])
        ax[0, 0].set_ylabel("original"); ax[1, 0].set_ylabel("FSQ recon")
        path = os.path.join(args.out, "tokenizer_recon.png")
        fig.tight_layout(); fig.savefig(path, dpi=110)
        print(f"saved {path}")
    except ImportError:
        print("(matplotlib not installed — numbers above; pip install matplotlib for the grid)")


if __name__ == "__main__":
    main()
