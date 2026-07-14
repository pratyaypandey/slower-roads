"""M2 claim #2: does the world model respond *correctly* to steering input?

The sim is the oracle, so we anchor the test to ground truth without re-rendering
anything. From data/seed1 we pick two real stretches: one where the car is
steering left (L) and one steering right (R). Each gives a context (the T frames
before it), a driving action sequence (its logged actions), and an oracle (its
real H frames).

Then we roll the dynamics core from each context under each action sequence and
score against the two oracles — a 2x2 matrix M[context][action]:

    M[c][a] = pixel_L1( dream(context_c, actions_a),  oracle_c )

The diagonal (a == c: true context + its own true steering) should beat the
off-diagonal (a != c: same context, the *wrong* steering) against the same
oracle_c. If steering were ignored, dream(ctx_L, act_L) == dream(ctx_L, act_R)
and the margin would be ~0. A positive, growing

    margin = mean(off-diagonal) - mean(diagonal)

means turning the wheel bends the generated world in the direction that matches
reality — exactly "responds correctly to steering input". We also write a labeled
GIF (same context, left-steer dream | right-steer dream) as the figure-1 artifact.

    python -m eval.eval_steering --data data/seed1 \
        --tokenizer checkpoints/tokenizer.pt --dynamics checkpoints/dynamics.pt \
        --context 4 --steps 8
"""

import argparse
import json
import os

import numpy as np
import torch

from model.registry import load_tokenizer, load_dynamics
from model.dynamics.sequence import build_context, action_to_vocab
from model.dynamics.config import STEER_EDGES, THROTTLE_BUCKETS, FRAME_STRIDE
from eval.drift import pixel_drift, latent_drift
from eval.eval_dream import action_id


def steer_of(a):
    """Signed steer in [-1,1] of a logged action ({...} or None)."""
    return 0.0 if a is None else float(a["steer"])


def find_turn_windows(samples, context, steps):
    """Return (left_start, right_start): starts of the most-left and most-right
    stretches whose `steps` driving actions have room for `context` frames before
    them. A window at start s dreams frames s..s+steps-1; the action driving frame
    s+k is samples[s+k-1].action, so its mean steer summarises the stretch."""
    best_l = best_r = None
    lo, hi = context, len(samples) - steps
    for s in range(lo, hi):
        mean_steer = np.mean([steer_of(samples[s + k - 1]["action"]) for k in range(steps)])
        if best_l is None or mean_steer < best_l[1]:
            best_l = (s, mean_steer)
        if best_r is None or mean_steer > best_r[1]:
            best_r = (s, mean_steer)
    return best_l, best_r


def load_window(data_dir, samples, start, context, steps, device):
    """Context frames + context action ids + the H driving action ids + oracle
    frames, for the window at `start`. Frame s+k is driven by action s+k-1."""
    ctx_idx = range(start - context, start)
    frames = [np.load(os.path.join(data_dir, samples[i]["frame"])) for i in ctx_idx]
    ctx_frames = torch.from_numpy(np.stack(frames)).float().to(device)      # (T,3,64,64)
    ctx_acts = torch.tensor([action_id(samples[i]["action"]) for i in ctx_idx], device=device)
    # H actions driving frames start..start+steps-1 (action at s+k-1 drives s+k).
    step_acts = torch.tensor(
        [action_id(samples[start + k - 1]["action"]) for k in range(steps)], device=device)
    oracle = np.stack([np.load(os.path.join(data_dir, samples[start + k]["frame"]))
                       for k in range(steps)])                              # (H,3,64,64)
    return ctx_frames, ctx_acts, step_acts, oracle


@torch.no_grad()
def rollout(dyn, tok, ctx_frames, ctx_acts, step_acts, window):
    """Free-run `len(step_acts)` frames from the context, feeding the given action
    each step (the model generates the visual tokens — that is what steering must
    drive). Keeps a bounded `window`-frame prefix (matching the trained context;
    an unbounded prefix runs RoPE out of distribution and melts). Returns dream
    frames (H,3,64,64)."""
    _, ctx_tokens, _ = tok(ctx_frames)                       # (T, TOKENS_PER_FRAME)
    prefix = build_context(ctx_acts.unsqueeze(0), ctx_tokens.unsqueeze(0))
    dreamed = []
    for a in step_acts:
        a = a.view(1)
        vis = dyn.generate_frame(prefix, a)
        dreamed.append(tok.decode_indices(vis).squeeze(0).cpu())
        prefix = torch.cat([prefix, action_to_vocab(a).view(1, 1), vis], dim=1)
        if window > 0 and prefix.shape[1] > window * FRAME_STRIDE:
            prefix = prefix[:, -window * FRAME_STRIDE:]
    return torch.stack(dreamed).clamp(0, 1).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/seed1")
    p.add_argument("--tokenizer", default="checkpoints/tokenizer.pt")
    p.add_argument("--dynamics", default="checkpoints/dynamics.pt")
    p.add_argument("--context", type=int, default=4)
    p.add_argument("--steps", type=int, default=8, help="frames to dream per branch")
    p.add_argument("--window", type=int, default=-1,
                   help="bounded prefix frames (-1 = match --context; 0 = unbounded)")
    p.add_argument("--out", default="eval/plots")
    args = p.parse_args()
    if args.window < 0:
        args.window = args.context

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok, _ = load_tokenizer(args.tokenizer, default_cfg={"hidden": 64}, map_location=device)
    tok = tok.to(device).eval()
    dyn, _ = load_dynamics(args.dynamics, map_location=device)
    dyn = dyn.to(device).eval()

    manifest = json.load(open(os.path.join(args.data, "manifest.json")))
    samples = manifest["samples"]
    (l_start, l_steer), (r_start, r_steer) = find_turn_windows(samples, args.context, args.steps)
    print(f"left stretch  @ {l_start}  mean steer {l_steer:+.3f}")
    print(f"right stretch @ {r_start}  mean steer {r_steer:+.3f}")
    if l_start == r_start:
        print("!! degenerate: no distinct left/right stretches in this trajectory")

    windows = {
        "L": load_window(args.data, samples, l_start, args.context, args.steps, device),
        "R": load_window(args.data, samples, r_start, args.context, args.steps, device),
    }

    # dreams[c][a] = free-run from context c under action sequence a.
    dreams, oracles = {}, {}
    for c, (ctx_frames, ctx_acts, _, oracle) in windows.items():
        oracles[c] = oracle
        dreams[c] = {}
        for a, (_, _, step_acts, _) in windows.items():
            dreams[c][a] = rollout(dyn, tok, ctx_frames, ctx_acts, step_acts, args.window)

    # 2x2 per-step pixel-L1 vs each context's own oracle.
    def curve(c, a):
        return pixel_drift(dreams[c][a], oracles[c], metric="l1")
    diag = 0.5 * (curve("L", "L") + curve("R", "R"))          # true steering
    off = 0.5 * (curve("L", "R") + curve("R", "L"))           # swapped steering
    margin = off - diag

    print("\n2x2 pixel-L1 vs oracle (rows=context, cols=action):")
    print(f"           act_L    act_R")
    for c in ("L", "R"):
        print(f"  ctx_{c}   {curve(c,'L').mean():.4f}   {curve(c,'R').mean():.4f}"
              f"   (oracle_{c})")
    print(f"\ndiagonal (true steer):  {diag.mean():.4f}")
    print(f"off-diag (wrong steer): {off.mean():.4f}")
    print(f"steering margin (off - diag): mean {margin.mean():+.4f}   "
          f"step1 {margin[0]:+.4f}   last {margin[-1]:+.4f}")
    print("PASS-ish" if margin.mean() > 0 else "FAIL: steering not on-target",
          "(margin > 0 means the correct action reconstructs reality better)")

    # Latent-L2 corroboration (encode dream + oracle to continuous codes).
    def latent(c, a):
        with torch.no_grad():
            zd = tok.encode(torch.from_numpy(dreams[c][a]).float().to(device)).cpu().numpy()
            zo = tok.encode(torch.from_numpy(oracles[c]).float().to(device)).cpu().numpy()
        return latent_drift(zd, zo)
    ldiag = 0.5 * (latent("L", "L") + latent("R", "R"))
    loff = 0.5 * (latent("L", "R") + latent("R", "L"))
    print(f"latent-L2 margin (off - diag): {(loff - ldiag).mean():+.4f}")

    # --- momentum-robust directional test -----------------------------------
    # The 2x2-vs-oracle metric is confounded when a context has strong turn
    # momentum. This measures the action's effect DIRECTLY: from each context,
    # how much do the left-action and right-action dreams diverge, and does the
    # road shift the correct way? (Steering left rotates the world right in view,
    # so the road centroid should move right — centroid_x(left) > centroid_x(right).)
    def road_centroid_x(frame):  # (3,H,W) -> mean column of dark road pixels, lower half
        dark = 1.0 - frame.mean(0)
        lower = dark[frame.shape[1] // 2:]
        w = lower.sum(0)
        return float((np.arange(frame.shape[2]) * w).sum() / (w.sum() + 1e-8))
    sens, dxs = [], []
    for c in ("L", "R"):
        dl, dr = dreams[c]["L"], dreams[c]["R"]        # same context, left vs right actions
        sens.append(np.mean([_ := np.abs(dl[k] - dr[k]).mean() for k in range(len(dl))]))
        dxs.append(np.mean([road_centroid_x(dl[k]) - road_centroid_x(dr[k]) for k in range(len(dl))]))
    print(f"\naction sensitivity (mean |dream_left - dream_right|): {np.mean(sens):.4f}   "
          f"(higher = action changes the world more; near 0 = ignores the dial)")
    print(f"road-centroid shift left-vs-right: {np.mean(dxs):+.2f} px   "
          f"(>0 = steering-left pushes road right, the correct direction)")

    # Figure-1 GIF: same context (L), left-steer dream | right-steer dream.
    try:
        from PIL import Image
        os.makedirs(args.out, exist_ok=True)
        imgs, scale = [], 4
        dl, dr = dreams["L"]["L"], dreams["L"]["R"]
        for k in range(args.steps):
            a = (dl[k].transpose(1, 2, 0) * 255).astype(np.uint8)
            b = (dr[k].transpose(1, 2, 0) * 255).astype(np.uint8)
            gap = np.full((a.shape[0], 2, 3), 255, np.uint8)
            pair = np.concatenate([a, gap, b], axis=1)
            imgs.append(Image.fromarray(np.repeat(np.repeat(pair, scale, 0), scale, 1)))
        path = os.path.join(args.out, "steering.gif")
        imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=120, loop=0)
        print(f"\nsaved {path}  (same context; left = steer-left dream, right = steer-right dream)")
    except ImportError:
        print("(pillow not installed — pip install pillow for the GIF)")


if __name__ == "__main__":
    main()
