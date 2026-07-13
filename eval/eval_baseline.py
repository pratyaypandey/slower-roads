"""The honest M2 bar: does the dynamics model actually predict the next frame
better than trivially copying the current one?

Absolute pixel drift is misleading — consecutive driving frames are ~99% identical,
so a broken model still scores a low L1. The only meaningful question is whether the
model beats the **persistence baseline** (predict next tokens = current tokens),
measured in the *same* decode path, ideally approaching the tokenizer floor.

Reports, per seed, in pixel-L1 and token-accuracy:
  * tokenizer floor    — decode(encode(true_{t+1})) vs true_{t+1}: best any latent
                         model could do (pure reconstruction error).
  * persistence        — decode(true tokens_t) vs true_{t+1}: copy the current frame.
  * MODEL (teacher-forced, bounded context) — the dynamics core.

Also reports tokenizer **temporal stability** (tokens-changed/frame, tokens flipped
by 1% pixel noise) — the root cause when the model can't beat persistence.

    python -m eval.eval_baseline --data data/seed2 --tokenizer checkpoints/tokenizer.pt \
        --dynamics checkpoints/dynamics_best.pt --start 100 --context 4 --steps 40
"""

import argparse
import json
import os

import numpy as np
import torch

from model.registry import load_tokenizer, load_dynamics
from model.dynamics.sequence import build_context, action_to_vocab
from model.dynamics.config import FRAME_STRIDE
from eval.eval_dream import action_id


def _l1(a, b):
    return float((a - b).abs().mean())


def load_seq(data_dir, start, n, device):
    m = json.load(open(os.path.join(data_dir, "manifest.json")))["samples"]
    fr = np.stack([np.load(os.path.join(data_dir, m[i]["frame"])) for i in range(start, start + n)])
    acts = [action_id(m[i]["action"]) for i in range(start, start + n)]
    return torch.from_numpy(fr).float().to(device), torch.tensor(acts, device=device)


@torch.no_grad()
def temporal_stability(tok, frames):
    """tokens-changed between consecutive frames, and tokens flipped by 1% noise."""
    _, toks, _ = tok(frames)
    changed = np.mean([float((toks[k + 1] != toks[k]).float().mean()) for k in range(len(frames) - 1)])
    noisy = (frames[:1] + 0.01 * torch.randn_like(frames[:1])).clamp(0, 1)
    _, t0, _ = tok(frames[:1])
    _, t1, _ = tok(noisy)
    flip = float((t0 != t1).float().mean())
    return changed, flip


@torch.no_grad()
def evaluate(tok, dyn, frames, acts, T, H, window):
    _, toks, _ = tok(frames)                                   # (T+H, tok) true tokens
    floor = np.mean([_l1(tok.decode_indices(toks[T + k:T + k + 1]), frames[T + k:T + k + 1])
                     for k in range(H)])
    persist = np.mean([_l1(tok.decode_indices(toks[T + k - 1:T + k]), frames[T + k:T + k + 1])
                       for k in range(H)])
    persist_acc = np.mean([float((toks[T + k] == toks[T + k - 1]).float().mean()) for k in range(H)])

    prefix = build_context(acts[:T].unsqueeze(0), toks[:T].unsqueeze(0))
    md, macc = [], []
    for k in range(H):
        a = acts[T + k - 1].view(1)
        vis = dyn.generate_frame(prefix, a)
        md.append(_l1(tok.decode_indices(vis), frames[T + k:T + k + 1]))
        macc.append(float((vis == toks[T + k].unsqueeze(0)).float().mean()))
        prefix = torch.cat([prefix, action_to_vocab(a).view(1, 1), toks[T + k].unsqueeze(0)], dim=1)
        if window > 0 and prefix.shape[1] > window * FRAME_STRIDE:
            prefix = prefix[:, -window * FRAME_STRIDE:]
    return dict(floor=floor, persist=persist, persist_acc=persist_acc,
                model=float(np.mean(md)), model_acc=float(np.mean(macc)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", default=["data/seed1"], help="seed dir(s) to evaluate on")
    p.add_argument("--tokenizer", default="checkpoints/tokenizer.pt")
    p.add_argument("--dynamics", default="checkpoints/dynamics_best.pt")
    p.add_argument("--start", type=int, default=100)
    p.add_argument("--context", type=int, default=4)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--window", type=int, default=-1, help="-1 = match --context")
    args = p.parse_args()
    if args.window < 0:
        args.window = args.context

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok, _ = load_tokenizer(args.tokenizer, default_cfg={"hidden": 64}, map_location=device)
    tok = tok.to(device).eval()
    dyn, _ = load_dynamics(args.dynamics, map_location=device)
    dyn = dyn.to(device).eval()

    all_pass = True
    for seed in args.data:
        frames, acts = load_seq(seed, args.start, args.context + args.steps, device)
        r = evaluate(tok, dyn, frames, acts, args.context, args.steps, args.window)
        changed, flip = temporal_stability(tok, frames)
        beats = r["model"] < r["persist"]
        all_pass &= beats
        name = os.path.basename(seed.rstrip("/"))
        print(f"\n[{name}]  (context {args.context}, {args.steps} steps, window {args.window})")
        print(f"  tokenizer temporal stability: tokens-changed/frame {changed*100:.0f}%   "
              f"noise-flip {flip*100:.0f}%   (lower = better)")
        print(f"  {'metric':<24}{'pixel-L1':>10}{'token-acc':>11}")
        print(f"  {'tokenizer floor':<24}{r['floor']:>10.4f}{'—':>11}")
        print(f"  {'persistence (copy)':<24}{r['persist']:>10.4f}{r['persist_acc']:>11.3f}")
        print(f"  {'MODEL (teacher-forced)':<24}{r['model']:>10.4f}{r['model_acc']:>11.3f}")
        print(f"  -> beats persistence: {'YES ✅' if beats else 'NO ❌'}   "
              f"(pixel {r['persist']-r['model']:+.4f}, token-acc {r['model_acc']-r['persist_acc']:+.3f})")
    print(f"\n{'='*40}\nM2 bar (beat persistence on all seeds): "
          f"{'PASS ✅' if all_pass else 'FAIL ❌'}")


if __name__ == "__main__":
    main()
