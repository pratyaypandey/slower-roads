"""Render every sample of a state manifest to a frame with the numpy software
renderer, writing .npy frames + a frame path into each sample. Output is a
drop-in 'rgb' dataset for model/data/dataset.py (which reads .npy directly, no
PIL needed) and thus for train_tokenizer.py / train_dynamics.py.

    node sim/headless/generate_drive.js --seed 1 --steps 3000
    python -m sim.render.render_manifest --data data/seed1_drive --size 64

Rewrites the manifest in place with representation 'rgb' + per-sample frame
paths, and drops the bulky road geometry (only the renderer needed it).
"""

import argparse
import json
import os

import numpy as np

from sim.render.software_renderer import render_frame


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="dir with a state manifest.json")
    p.add_argument("--size", type=int, default=64)
    args = p.parse_args()

    manifest_path = os.path.join(args.data, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
    road = manifest["road"]
    frames_dir = os.path.join(args.data, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    for i, sample in enumerate(manifest["samples"]):
        img = render_frame(sample["state"], road, size=args.size)
        # (H,W,3) uint8 -> (3,H,W) float32 [0,1], the dataset's expected layout.
        arr = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)
        rel = os.path.join("frames", f"{i:06d}.npy")
        np.save(os.path.join(args.data, rel), arr)
        sample["frame"] = rel

    manifest["representation"] = "rgb"
    manifest["resolution"] = [args.size, args.size]
    manifest.pop("road", None)  # renderer consumed it; keep the manifest lean
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    print(f"rendered {len(manifest['samples'])} frames at {args.size}px -> {frames_dir}")


if __name__ == "__main__":
    main()
