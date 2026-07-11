"""Decode-latency + parameter profile for tokenizer checkpoints.

The decoder is the per-frame render path at inference (the encoder runs only at
train/conditioning time), so decode latency is the real-time constraint. Reports
encoder/decoder param split and decode ms/frame at batch 1 (the on-device case)
and a throughput batch.

    python -m eval.profile_decode --ckpts checkpoints/fsq.pt:fsq,checkpoints/fsq_v2.pt:fsq_v2
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from model.registry import load_tokenizer


def params_m(module):
    return sum(p.numel() for p in module.parameters()) / 1e6


@torch.no_grad()
def time_decode(model, tok, device, batch, iters=50):
    idx = torch.randint(0, model.codebook_size, (batch, tok), device=device)
    for _ in range(5):
        model.decode_indices(idx)                    # warmup
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        model.decode_indices(idx)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    return dt * 1000.0 / batch                        # ms per frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", required=True, help="comma list path:label")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    print(f"device={device}   (budget: <33 ms/frame for 30fps real-time)")
    print(f"{'model':>10} {'params':>8} {'enc':>7} {'dec':>7} {'tok':>5} "
          f"{'ms/frame b1':>12} {'ms/frame b32':>13}")
    for spec in args.ckpts.split(","):
        path, _, label = spec.partition(":")
        model, _ = load_tokenizer(path, map_location=device)
        model = model.to(device).eval()
        tok = model.tokens_per_frame
        b1 = time_decode(model, tok, device, 1)
        b32 = time_decode(model, tok, device, 32)
        print(f"{label or os.path.basename(path):>10} {params_m(model):7.2f}M "
              f"{params_m(model.encoder):6.2f}M {params_m(model.decoder):6.2f}M {tok:5d} "
              f"{b1:11.2f} {b32:12.2f}")


if __name__ == "__main__":
    main()
