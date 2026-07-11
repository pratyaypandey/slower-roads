"""Roll the trained state-dynamics model forward and compare its dreamed drive
to the sim's ground truth — the visual, honest test that a low training loss
actually means the model *drives*.

Given a start state + the true action sequence, the model predicts every step
feeding its OWN output back (free-running autoregression, the hard case that
compounds error), then we overlay the dreamed (x,z) path on the true path and
plot per-step drift. Answers "does it stay on the road, and where does it
diverge?" — not just "did the loss go down?".

    python -m model.train_state_dynamics --data ~/data/seed1_state --epochs 40
    python -m eval.eval_state_dynamics --data ~/data/seed1_state \
        --ckpt checkpoints/state_dynamics.pt

Writes eval/plots/state_rollout.png if matplotlib is available; always prints an
ASCII path overlay + drift table so it works headless.
"""

import argparse
import json
import os

import numpy as np
import torch

from model.train_state_dynamics import StateDynamics, STATE_DIM
from model.data.dataset import STATE_KEYS
from model.dynamics.config import tokenize_action
from eval.drift import latent_drift


def load_drive(data_dir):
    manifest = json.load(open(os.path.join(data_dir, "manifest.json")))
    samples = manifest["samples"]
    states = np.array(
        [[s["state"][k] for k in STATE_KEYS] for s in samples], dtype=np.float32
    )
    # samples[0].action is null; action i drives state i-1 -> state i.
    actions = np.array(
        [tokenize_action(s["action"]["throttle"], s["action"]["brake"], s["action"]["steer"])
         for s in samples[1:]],
        dtype=np.int64,
    )
    return states, actions


@torch.no_grad()
def free_run(model, mean, std, init_state, actions, steps):
    # Free-running rollout: start at init_state, feed the model its own output.
    # Work in standardized space (as trained), de-standardize for the path.
    s = ((torch.from_numpy(init_state) - mean) / std).unsqueeze(0)
    out = []
    for k in range(steps):
        a = torch.tensor([actions[k]])
        s = model(s, a)
        out.append((s.squeeze(0) * std + mean).numpy())
    return np.array(out)  # (steps, STATE_DIM), real units


def ascii_paths(true_xz, pred_xz, width=64, height=24):
    # Overlay two (x,z) paths in a terminal grid. '#' true, 'o' pred, '*' both.
    pts = np.concatenate([true_xz, pred_xz], axis=0)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    grid = [[" "] * width for _ in range(height)]

    def place(path, ch):
        for x, z in path:
            cx = int((x - lo[0]) / span[0] * (width - 1))
            cy = int((z - lo[1]) / span[1] * (height - 1))
            cy = height - 1 - cy  # z up
            cur = grid[cy][cx]
            grid[cy][cx] = "*" if cur in ("#", "o", "*") and cur != ch else ch

    place(true_xz, "#")
    place(pred_xz, "o")
    print("\n  true '#'   predicted 'o'   overlap '*'")
    print("  +" + "-" * width + "+")
    for row in grid:
        print("  |" + "".join(row) + "|")
    print("  +" + "-" * width + "+")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/seed1_state")
    p.add_argument("--ckpt", default="checkpoints/state_dynamics.pt")
    p.add_argument("--steps", type=int, default=200, help="rollout length to evaluate")
    p.add_argument("--start", type=int, default=100,
                   help="index into the drive to start the rollout from; default 100 "
                        "skips the standing-start acceleration ramp so the eval tests "
                        "steady-state driving, not just the hardest first seconds")
    p.add_argument("--out", default="eval/plots")
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = StateDynamics(hidden=ckpt.get("hidden", 128))
    model.load_state_dict(ckpt["model"])
    model.eval()
    mean, std = ckpt["state_mean"], ckpt["state_std"]

    states, actions = load_drive(args.data)
    start = min(args.start, len(actions) - 2)
    steps = min(args.steps, len(actions) - start)
    # Roll from `start` using the true state there as the seed, then free-run.
    pred = free_run(model, mean, std, states[start], actions[start:start + steps], steps)
    true = states[start + 1 : start + steps + 1]

    # Per-step drift over the whole state vector (standardized so dims compare).
    pred_z = (pred - mean.numpy()) / std.numpy()
    true_z = (true - mean.numpy()) / std.numpy()
    drift = latent_drift(pred_z, true_z)

    # Report position error relative to distance actually driven — an absolute
    # metre figure is meaningless without knowing how far the car went.
    final_err = np.linalg.norm(pred[-1, :2] - true[-1, :2])
    dist = np.sum(np.linalg.norm(np.diff(true[:, :2], axis=0), axis=1))
    print(f"\nfree-running rollout: {steps} steps from drive index {start} "
          f"(speed there = {states[start, 3]:.1f} m/s)")
    print(f"  final position error: {final_err:.1f} m  over {dist:.0f} m driven "
          f"= {100 * final_err / max(1, dist):.1f}% of path length")
    print(f"  drift (standardized L2)  step 1: {drift[0]:.3f}   "
          f"step {steps//2}: {drift[steps//2]:.3f}   step {steps}: {drift[-1]:.3f}")
    per_dim = np.abs(pred - true).mean(axis=0)
    print("  mean abs error per dim: " +
          "  ".join(f"{k}={e:.3f}" for k, e in zip(STATE_KEYS, per_dim)))

    ascii_paths(true[:, :2], pred[:, :2])

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(args.out, exist_ok=True)
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].plot(true[:, 0], true[:, 1], label="ground truth", lw=2)
        ax[0].plot(pred[:, 0], pred[:, 1], "--", label="model (free-run)", lw=2)
        ax[0].set_title("driven path (x, z)")
        ax[0].set_aspect("equal"); ax[0].legend(); ax[0].set_xlabel("x"); ax[0].set_ylabel("z")
        ax[1].plot(drift, lw=2)
        ax[1].set_title("per-step drift vs oracle (standardized L2)")
        ax[1].set_xlabel("rollout step"); ax[1].set_ylabel("drift")
        # Tag the plot by the checkpoint's parent dir so A/B/C runs don't
        # overwrite each other (checkpoints/B/state_dynamics.pt -> rollout_B.png).
        tag = os.path.basename(os.path.dirname(os.path.abspath(args.ckpt))) or "run"
        path = os.path.join(args.out, f"rollout_{tag}.png")
        fig.tight_layout(); fig.savefig(path, dpi=110)
        print(f"\nsaved plot: {path}")
    except ImportError:
        print("\n(matplotlib not installed — ASCII overlay above; "
              "pip install matplotlib for the PNG)")


if __name__ == "__main__":
    main()
