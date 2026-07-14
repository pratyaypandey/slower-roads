"""Precompute frozen-tokenizer token indices for a seed's frames, once.

Dynamics training re-encodes every frame through the tokenizer every epoch — pure
waste, since the tokenizer is frozen. This encodes each seed's frames a single time
and writes `<seed>/latents.npy` (N, tokens) int32, aligned 1:1 with the manifest
samples. The dataset's representation='latent' then serves token windows directly,
so training never loads a frame or runs the tokenizer — ~10x faster, and what makes
many-seed training affordable.

Re-run whenever the tokenizer changes (the latents are tokenizer-specific).

    python -m model.precompute_latents --tokenizer checkpoints/tokenizer_tc.pt \
        --data data/seed1 data/seed3 data/seed4 data/seed5
"""

import argparse
import json
import os

import numpy as np
import torch

from model.registry import load_tokenizer


def encode_seed(tok, data_dir, batch_size, device):
    manifest = json.load(open(os.path.join(data_dir, "manifest.json")))
    samples = manifest["samples"]
    idxs = []
    for i in range(0, len(samples), batch_size):
        chunk = samples[i:i + batch_size]
        frames = np.stack([np.load(os.path.join(data_dir, s["frame"])) for s in chunk])
        x = torch.from_numpy(frames).float().to(device)
        with torch.no_grad():
            _, tok_idx, _ = tok(x)
        idxs.append(tok_idx.cpu().numpy().astype(np.int32))
    latents = np.concatenate(idxs, axis=0)                 # (N, tokens)
    out = os.path.join(data_dir, "latents.npy")
    np.save(out, latents)
    return out, latents.shape


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", required=True, help="seed dir(s) to encode")
    p.add_argument("--tokenizer", default="checkpoints/tokenizer.pt")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    device = torch.device(args.device)
    tok, _ = load_tokenizer(args.tokenizer, default_cfg={"hidden": 64}, map_location=device)
    tok = tok.to(device).eval()
    for d in args.data:
        out, shape = encode_seed(tok, d, args.batch_size, device)
        print(f"wrote {out}  {shape}")


if __name__ == "__main__":
    main()
