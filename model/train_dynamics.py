"""M2: train the AR dynamics core on latent tokens from a frozen tokenizer.

Encodes each frame to visual tokens with the M1 tokenizer (frozen), builds the
interleaved [action, visual...] context per §4, and optimizes the multi-step
rollout loss (§5) — token CE plus decoded-pixel loss over an H-step rollout.
The tokenizer's decode_indices is passed to rollout_loss as the decoder.

    python -m model.train_tokenizer --data data/seed1 --epochs 20
    python -m model.train_dynamics --data data/seed1 --tokenizer checkpoints/tokenizer.pt

CPU-shape-testable with --smoke (random tensors, no data or checkpoint needed).
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader

from model.tokenizer.fsq_autoencoder import FSQAutoencoder
from model.dynamics.ar_core import ARDynamics
from model.dynamics.rollout_loss import rollout_loss
from model.dynamics.config import (
    NUM_VISUAL_TOKENS,
    TOKENS_PER_FRAME,
)
from model.dynamics.sequence import build_context
from model.data.dataset import SimSequenceDataset


def load_frozen_tokenizer(path, device):
    ckpt = torch.load(path, map_location=device)
    tok = FSQAutoencoder(hidden=ckpt.get("hidden", 64)).to(device)
    tok.load_state_dict(ckpt["model"])
    tok.eval()
    for p in tok.parameters():
        p.requires_grad_(False)
    return tok


@torch.no_grad()
def encode_frames(tokenizer, frames):
    # frames (B, N, 3, 64, 64) -> visual token ids (B, N, TOKENS_PER_FRAME).
    b, n = frames.shape[:2]
    flat = frames.reshape(b * n, *frames.shape[2:])
    _, indices, _ = tokenizer(flat)
    return indices.reshape(b, n, TOKENS_PER_FRAME)


def train(args):
    device = torch.device(args.device)
    model = ARDynamics(
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers
    ).to(device)
    print(f"dynamics parameters: {model.param_count() / 1e6:.2f}M")

    if args.smoke:
        tok = FSQAutoencoder().to(device).eval()
        B, T, H = 2, args.context, args.horizon
        ctx_frames = torch.rand(B, T, 3, 64, 64, device=device)
        tgt_frames = torch.rand(B, H, 3, 64, 64, device=device)
        ctx_actions = torch.randint(0, 9, (B, T), device=device)
        tgt_actions = torch.randint(0, 9, (B, H), device=device)
        z_ctx = build_context(ctx_actions, encode_frames(tok, ctx_frames))
        target_tokens = encode_frames(tok, tgt_frames)
        total, parts = rollout_loss(
            model, tok.decode_indices, z_ctx, tgt_actions, target_tokens,
            tgt_frames, H,
        )
        total.backward()
        print(f"[smoke] rollout loss {total.item():.4f} "
              f"(ce {parts['ce'].item():.4f}, pixel {parts['pixel'].item():.4f}) — OK")
        return

    tokenizer = load_frozen_tokenizer(args.tokenizer, device)
    dataset = SimSequenceDataset(
        os.path.join(args.data, "manifest.json"),
        context=args.context,
        horizon=args.horizon,
        representation="rgb",
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Resume: reload dynamics model + optimizer + epoch (the tokenizer is always
    # loaded fresh and frozen, so it's not part of the resume state).
    start_epoch = 0
    if args.resume:
        prev = torch.load(args.resume, map_location=device)
        model.load_state_dict(prev["model"])
        if "opt" in prev:
            opt.load_state_dict(prev["opt"])
        start_epoch = prev.get("epoch", 0)
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    ckpt = os.path.join(args.out, "dynamics.pt")
    os.makedirs(args.out, exist_ok=True)
    for epoch in range(start_epoch, args.epochs):
        run_ce = run_px = 0.0
        for item in loader:
            ctx_frames = item["context_frames"].float().to(device)
            tgt_frames = item["target_frames"].float().to(device)
            ctx_actions = item["context_actions"].to(device)
            tgt_actions = item["target_actions"].to(device)

            z_ctx = build_context(ctx_actions, encode_frames(tokenizer, ctx_frames))
            target_tokens = encode_frames(tokenizer, tgt_frames)

            total, parts = rollout_loss(
                model, tokenizer.decode_indices, z_ctx, tgt_actions,
                target_tokens, tgt_frames, args.horizon,
                ce_weight=args.ce_weight, pixel_weight=args.pixel_weight,
            )
            opt.zero_grad()
            total.backward()
            opt.step()
            run_ce += parts["ce"].item()
            run_px += parts["pixel"].item()
        n = max(1, len(loader))
        print(f"epoch {epoch + 1}/{args.epochs}  ce {run_ce / n:.4f}  pixel {run_px / n:.4f}")
        # Checkpoint every epoch so long runs survive interruption + resume.
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "epoch": epoch + 1}, ckpt)
    print(f"saved {ckpt}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/seed1")
    p.add_argument("--tokenizer", default="checkpoints/tokenizer.pt")
    p.add_argument("--out", default="checkpoints")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--context", type=int, default=4)
    p.add_argument("--horizon", type=int, default=6)
    p.add_argument("--d-model", type=int, default=256, dest="d_model")
    p.add_argument("--n-heads", type=int, default=4, dest="n_heads")
    p.add_argument("--n-layers", type=int, default=4, dest="n_layers")
    p.add_argument("--ce-weight", type=float, default=1.0, dest="ce_weight")
    p.add_argument("--pixel-weight", type=float, default=1.0, dest="pixel_weight")
    p.add_argument("--resume", default=None, help="checkpoint to continue training from")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--smoke", action="store_true", help="tiny random-tensor pass, no data")
    train(p.parse_args())


if __name__ == "__main__":
    main()
