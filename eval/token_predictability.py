"""Token predictability + temporal stability of a tokenizer's latents.

The tokenizer serves the AR dynamics model, so the decisive question isn't
reconstruction sharpness but: are its tokens *predictable*? Two cheap proxies,
both per tokenizer checkpoint:

  1. temporal change-rate: fraction of the 256 tokens that flip between
     consecutive (near-identical) frames. Lower = more stable = easier to predict.
  2. AR next-token accuracy: train a small causal transformer on the tokenizer's
     visual-token sequences (no actions, to isolate the tokens) for a few hundred
     steps and report held-out teacher-forced accuracy. Higher = more predictable.

A tokenizer with slightly worse rFID but far more predictable tokens is better for
a world model — so this is the tie-breaker (docs/tokenizer_research.md).

    python -m eval.token_predictability --data data/seed1 --ckpt checkpoints/fsq_v2.pt --steps 400
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from model.dynamics.ar_core import ARDynamics
from model.registry import load_tokenizer


@torch.no_grad()
def encode_drive(model, data_dir, n, device, bs=64):
    manifest = json.load(open(os.path.join(data_dir, "manifest.json")))
    samples = manifest["samples"][:n]
    toks = []
    for i in range(0, len(samples), bs):
        arrs = [np.load(os.path.join(data_dir, s["frame"])) for s in samples[i:i + bs]]
        frames = torch.from_numpy(np.stack(arrs)).float().to(device)
        _, idx, _ = model(frames)                      # (b, tok)
        toks.append(idx.cpu())
    return torch.cat(toks)                             # (n, tok) int64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/seed1")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--frames", type=int, default=1200)
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(0)

    tok_model, _ = load_tokenizer(args.ckpt, map_location=device)
    tok_model = tok_model.to(device).eval()
    for p in tok_model.parameters():
        p.requires_grad_(False)
    z = encode_drive(tok_model, args.data, args.frames, device)   # (N, TOK)
    N, TOK = z.shape
    V = tok_model.codebook_size

    # (1) temporal change-rate between consecutive frames.
    change = (z[1:] != z[:-1]).float().mean().item()

    # (2) AR next-token accuracy on flat visual-token windows (no actions).
    W = args.window
    seq_len = W * TOK
    split = int(N * 0.85)
    train_z, test_z = z[:split], z[split:]

    ar = ARDynamics(d_model=128, n_heads=4, n_layers=3,
                    max_seq_len=seq_len + 8, vocab_size=V).to(device)
    opt = torch.optim.Adam(ar.parameters(), lr=3e-4)

    def sample(zsrc, b):
        starts = torch.randint(0, zsrc.shape[0] - W, (b,))
        return torch.stack([zsrc[s:s + W].reshape(-1) for s in starts]).to(device)  # (b, W*TOK)

    ar.train()
    for step in range(args.steps):
        loss = ar.training_step(sample(train_z, 16))
        opt.zero_grad(); loss.backward(); opt.step()

    ar.eval()
    with torch.no_grad():
        seq = sample(test_z, 128)
        logits = ar(seq)
        pred = logits[:, :-1].argmax(-1)
        acc = (pred == seq[:, 1:]).float().mean().item()

    print(json.dumps({
        "ckpt": args.ckpt, "vocab": V, "tokens_per_frame": TOK,
        "temporal_change_rate": round(change, 4),
        "ar_next_token_acc": round(acc, 4),
        "chance_acc": round(1.0 / V, 5),
    }, indent=1))


if __name__ == "__main__":
    main()
