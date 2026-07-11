"""Export the state model's free-running dreamed trajectory (and the true one)
as car poses, for rendering through the real sim renderer (Option A).

The state dynamics model predicts {x, z, heading, speed}; this rolls it out
free-running from a mid-drive seed (feeding its own output back), alongside the
ground-truth poses, and writes both to JSON. sim/headless/render_dream.mjs then
drives the real WebGL renderer along each pose sequence to make a side-by-side
video — true drive vs the model's dream, both fully rendered.

    python -m eval.export_dream_poses --data data/seed1 \
        --ckpt checkpoints/state_dynamics.pt --start 100 --steps 200
"""

import argparse
import json
import os

import numpy as np
import torch

from model.train_state_dynamics import StateDynamics
from eval.eval_state_dynamics import load_drive, free_run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/seed1")
    p.add_argument("--ckpt", default="checkpoints/state_dynamics.pt")
    p.add_argument("--start", type=int, default=100)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--out", default="data/dream_poses.json")
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = StateDynamics(hidden=ckpt.get("hidden", 128))
    model.load_state_dict(ckpt["model"])
    model.eval()
    mean, std = ckpt["state_mean"], ckpt["state_std"]

    states, actions = load_drive(args.data)
    start = min(args.start, len(actions) - 2)
    steps = min(args.steps, len(actions) - start)
    pred = free_run(model, mean, std, states[start], actions[start:start + steps], steps)
    true = states[start + 1:start + steps + 1]

    # STATE_KEYS order is (x, z, heading, speed) — pose is what the renderer needs.
    def poses(arr):
        return [{"x": float(x), "z": float(z), "heading": float(h), "speed": float(s)}
                for x, z, h, s in arr]

    out = {
        "seed": json.load(open(os.path.join(args.data, "manifest.json")))["seed"],
        "start": start, "steps": steps,
        "true": poses(true), "dream": poses(pred),
    }
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"wrote {steps} true+dream poses to {args.out} (seed {out['seed']}, start {start})")


if __name__ == "__main__":
    main()
