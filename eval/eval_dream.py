"""Watch the world model dream. Seeds the AR dynamics core with a few real
frames, then generates the rest autoregressively (feeding its own predictions
back — the honest free-running test), decodes every frame with the tokenizer,
and writes a side-by-side GIF: ground truth (left) vs. the model's dream (right).

This is the payoff artifact — if the dreamed road curves with the action and
stays coherent, the world model works; if it melts into noise after a few steps,
that's drift, shown directly.

    CUDA_VISIBLE_DEVICES=7 python -m eval.eval_dream \
        --data data/seed1_drive --tokenizer checkpoints/tokenizer.pt \
        --dynamics checkpoints/dynamics.pt --context 4 --steps 60

Writes eval/plots/dream.gif (needs pillow). Also prints per-step pixel drift.
"""

import argparse
import json
import os

import numpy as np
import torch

from model.tokenizer.fsq_autoencoder import FSQAutoencoder
from model.dynamics.ar_core import ARDynamics
from model.dynamics.config import NUM_VISUAL_TOKENS, TOKENS_PER_FRAME
from model.train_dynamics import build_context
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
    from model.dynamics.config import tokenize_action
    if a is None:
        return 4  # neutral (coast+straight)
    return tokenize_action(a["throttle"], a["brake"], a["steer"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/seed1_drive")
    p.add_argument("--tokenizer", default="checkpoints/tokenizer.pt")
    p.add_argument("--dynamics", default="checkpoints/dynamics.pt")
    p.add_argument("--start", type=int, default=100)
    p.add_argument("--context", type=int, default=4)
    p.add_argument("--steps", type=int, default=60, help="frames to dream")
    p.add_argument("--out", default="eval/plots")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok_ck = torch.load(args.tokenizer, map_location=device)
    tok = FSQAutoencoder(hidden=tok_ck.get("hidden", 64)).to(device).eval()
    tok.load_state_dict(tok_ck["model"])
    dyn_ck = torch.load(args.dynamics, map_location=device)
    dyn = ARDynamics().to(device).eval()
    dyn.load_state_dict(dyn_ck["model"])

    T, H = args.context, args.steps
    frames_np, actions, _ = load_frames(args.data, args.start, T + H)
    frames = torch.from_numpy(frames_np).float().to(device)
    act_ids = torch.tensor([action_id(a) for a in actions], device=device)

    # Encode the context frames to tokens, build the interleaved prefix.
    with torch.no_grad():
        _, ctx_tokens, _ = tok(frames[:T])                    # (T, 64)
        ctx_tokens = ctx_tokens.unsqueeze(0)                  # (1, T, 64)
        prefix = build_context(act_ids[:T].unsqueeze(0), ctx_tokens)  # (1, T*65)

        dreamed = []
        for k in range(H):
            a = act_ids[T + k - 1].view(1)                    # action driving this step
            vis = dyn.generate_frame(prefix, a)               # (1, 64) predicted visual ids
            frame_hat = tok.decode_indices(vis)               # (1,3,64,64)
            dreamed.append(frame_hat.squeeze(0).cpu())
            # Autoregress: append (action, predicted visual) to the prefix.
            u = (a + NUM_VISUAL_TOKENS).view(1, 1)
            prefix = torch.cat([prefix, u, vis], dim=1)
    dream = torch.stack(dreamed).clamp(0, 1).numpy()          # (H,3,64,64)
    truth = frames_np[T:T + H]                                # (H,3,64,64)

    drift = pixel_drift(dream, truth, metric="l1")
    print(f"dreamed {H} frames from {T} context frames")
    print(f"pixel drift (L1)  step 1: {drift[0]:.4f}   "
          f"step {H//2}: {drift[H//2]:.4f}   step {H}: {drift[-1]:.4f}")

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
        path = os.path.join(args.out, "dream.gif")
        imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=100, loop=0)
        print(f"saved {path}  (left = ground truth, right = model's dream)")
    except ImportError:
        print("(pillow not installed — pip install pillow for the GIF)")


if __name__ == "__main__":
    main()
