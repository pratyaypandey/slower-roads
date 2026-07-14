"""Watch the world model dream. Seeds the AR dynamics core with a few real
frames, then generates the rest, decodes every frame with the tokenizer, and
writes a side-by-side GIF: ground truth (left) vs. the model's dream (right).

Two modes:
  * free-run (default) — feed the model's own predicted tokens back each step.
    The honest drift test (M3's bar): melting after a few steps *is* drift.
  * --teacher-forced   — feed the ground-truth tokens back each step, so the
    model always predicts frame t+1 from *true* context. This is per-step
    next-frame prediction and is exactly the M2 "done-when" bar: high token
    accuracy + low, flat pixel drift over a few seconds = coherent under TF.

This is the payoff artifact — if the dreamed road curves with the action and
stays coherent, the world model works.

    CUDA_VISIBLE_DEVICES=7 python -m eval.eval_dream \
        --data data/seed1 --tokenizer checkpoints/tokenizer.pt \
        --dynamics checkpoints/dynamics.pt --context 4 --steps 60 [--teacher-forced]

Writes eval/plots/dream.gif (free-run) or dream_tf.gif (--teacher-forced); needs
pillow. Prints per-step pixel drift + mean token accuracy.
"""

import argparse
import json
import os

import numpy as np
import torch

from model.registry import load_tokenizer, load_dynamics
from model.dynamics.config import NUM_VISUAL_TOKENS, TOKENS_PER_FRAME, FRAME_STRIDE
from model.dynamics.sequence import build_context, action_to_vocab
from eval.drift import pixel_drift


def load_frames(data_dir, start, n):
    manifest = json.load(open(os.path.join(data_dir, "manifest.json")))
    samples = manifest["samples"]
    frames, actions = [], []
    for i in range(start, start + n):
        frames.append(np.load(os.path.join(data_dir, samples[i]["frame"])))
        a = samples[i]["action"]
        actions.append(a)  # may be None for i==0
    return np.stack(frames), actions, manifest


def action_id(a):
    from model.dynamics.config import tokenize_action, NUM_ACTION_TOKENS
    if a is None:
        return NUM_ACTION_TOKENS // 2  # neutral (straight + coast, center bucket)
    return tokenize_action(a["steer"], a["throttle"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/seed1")
    p.add_argument("--tokenizer", default="checkpoints/tokenizer.pt")
    p.add_argument("--dynamics", default="checkpoints/dynamics.pt")
    p.add_argument("--start", type=int, default=100)
    p.add_argument("--context", type=int, default=4)
    p.add_argument("--steps", type=int, default=60, help="frames to dream")
    p.add_argument("--teacher-forced", action="store_true", dest="teacher_forced",
                   help="feed ground-truth tokens back each step (the M2 bar)")
    p.add_argument("--window", type=int, default=-1,
                   help="keep only the last N frames in the prefix (bounded context, "
                        "how the core was trained + how real-time runs). "
                        "-1 = match --context (default), 0 = unbounded (melts; OOD)")
    p.add_argument("--out", default="eval/plots")
    args = p.parse_args()
    if args.window < 0:
        args.window = args.context  # bounded to the trained context by default

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Registry loaders rebuild each model from its saved builder+cfg, so a
    # non-default tokenizer/dynamics (a variant, or a differently-sized core)
    # reloads correctly instead of silently constructing the default arch.
    tok, _ = load_tokenizer(args.tokenizer, default_cfg={"hidden": 64}, map_location=device)
    tok = tok.to(device).eval()
    dyn, _ = load_dynamics(args.dynamics, map_location=device)
    dyn = dyn.to(device).eval()

    T, H = args.context, args.steps
    frames_np, actions, _ = load_frames(args.data, args.start, T + H)
    frames = torch.from_numpy(frames_np).float().to(device)
    act_ids = torch.tensor([action_id(a) for a in actions], device=device)

    # Encode every frame (context + targets) to tokens up front: the context
    # seeds the prefix, and in teacher-forced mode the target tokens are what we
    # feed back each step + score the prediction against.
    with torch.no_grad():
        _, all_tokens, _ = tok(frames[:T + H])                # (T+H, TOKENS_PER_FRAME)
        ctx_tokens = all_tokens[:T].unsqueeze(0)              # (1, T, TOKENS_PER_FRAME)
        prefix = build_context(act_ids[:T].unsqueeze(0), ctx_tokens)  # (1, T*FRAME_STRIDE)

        dreamed, token_acc = [], []
        for k in range(H):
            a = act_ids[T + k - 1].view(1)                    # action driving this step
            vis = dyn.generate_frame(prefix, a)               # (1, TOKENS_PER_FRAME) predicted
            frame_hat = tok.decode_indices(vis)               # (1,3,64,64)
            dreamed.append(frame_hat.squeeze(0).cpu())
            gt_vis = all_tokens[T + k].unsqueeze(0)           # (1, TOKENS_PER_FRAME) ground truth
            token_acc.append((vis == gt_vis).float().mean().item())
            # Autoregress: append (action, tokens) to the prefix. Free-run feeds
            # the model's own prediction; teacher-forced feeds ground truth.
            u = action_to_vocab(a).view(1, 1)
            feed = gt_vis if args.teacher_forced else vis
            prefix = torch.cat([prefix, u, feed], dim=1)
            # Sliding window: the core was trained on `context`-frame sequences, so
            # an unbounded prefix runs RoPE positions far out of distribution and
            # the frame melts. --window N keeps only the last N frames (a bounded
            # context, exactly how a real-time deployment runs). 0 = unbounded.
            if args.window > 0 and prefix.shape[1] > args.window * FRAME_STRIDE:
                prefix = prefix[:, -args.window * FRAME_STRIDE:]
    dream = torch.stack(dreamed).clamp(0, 1).numpy()          # (H,3,64,64)
    truth = frames_np[T:T + H]                                # (H,3,64,64)

    drift = pixel_drift(dream, truth, metric="l1")
    acc = np.array(token_acc)
    mode = "teacher-forced" if args.teacher_forced else "free-run"
    print(f"dreamed {H} frames from {T} context frames  [{mode}]")
    print(f"pixel drift (L1)  step 1: {drift[0]:.4f}   "
          f"step {H//2}: {drift[H//2]:.4f}   step {H}: {drift[-1]:.4f}")
    print(f"token accuracy    step 1: {acc[0]:.3f}   "
          f"step {H//2}: {acc[H//2]:.3f}   step {H}: {acc[-1]:.3f}   mean: {acc.mean():.3f}")

    try:
        from PIL import Image
        os.makedirs(args.out, exist_ok=True)
        imgs = []
        scale = 4
        for k in range(H):
            t = (truth[k].transpose(1, 2, 0) * 255).astype(np.uint8)
            d = (dream[k].transpose(1, 2, 0) * 255).astype(np.uint8)
            gap = np.full((t.shape[0], 2, 3), 255, np.uint8)
            pair = np.concatenate([t, gap, d], axis=1)  # truth | dream
            pair = np.repeat(np.repeat(pair, scale, 0), scale, 1)
            imgs.append(Image.fromarray(pair))
        path = os.path.join(args.out, "dream_tf.gif" if args.teacher_forced else "dream.gif")
        imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=100, loop=0)
        print(f"saved {path}  (left = ground truth, right = model's dream)")
    except ImportError:
        print("(pillow not installed — pip install pillow for the GIF)")


if __name__ == "__main__":
    main()
