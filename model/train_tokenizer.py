"""M1: train the FSQ autoencoder to clean reconstruction on sim frames.

Reconstruction is per-frame, so this flattens the dataset's sequence windows
into individual frames. Produces a checkpoint the dynamics stage (M2) loads and
freezes. Run on a GPU box after generating a pixel dataset with the sim's WebGL
capture exporter (needs a browser/GPU):

    node sim/headless/generate_pixels.mjs --seed 1 --steps 2000 --size 64
    python -m model.train_tokenizer --data data/seed1 --epochs 20

CPU-shape-testable with --smoke (few random frames, no data needed).
"""

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from model.tokenizer.fsq_autoencoder import (
    reconstruction_loss,
    count_parameters,
)
from model.tokenizer.losses import tokenizer_loss, make_lpips
from model.registry import build_tokenizer
from model.data.dataset import SimSequenceDataset


def frames_from_batch(item):
    # Dataset yields (B, T, 3, 64, 64) context + (B, H, 3, 64, 64) target windows;
    # for reconstruction we only need frames, so merge both into one (N,3,64,64).
    frames = torch.cat([item["context_frames"], item["target_frames"]], dim=1)
    return frames.reshape(-1, *frames.shape[2:]).float()


def load_all_frames(data_dir, device):
    """Load every unique frame once into a single (N,3,64,64) device tensor.

    Reconstruction training only needs individual frames, and the whole 2.5k-frame
    seed set is ~137MB — so caching it on the GPU makes training compute-bound
    instead of re-reading ~390 overlapping-window frames per step from disk. It
    also fixes the 10x redundancy of flattening sequence windows.
    """
    import numpy as np
    manifest = json.load(open(os.path.join(data_dir, "manifest.json")))
    arrs = [np.load(os.path.join(data_dir, s["frame"])) for s in manifest["samples"]]
    return torch.from_numpy(np.stack(arrs)).float().to(device)


def train(args):
    device = torch.device(args.device)
    # Build via the registry so --arch selects the tokenizer; the cfg is saved
    # with the checkpoint so it reloads exactly (any size/variant).
    tok_cfg = {"hidden": args.hidden}
    if args.levels:
        tok_cfg["levels"] = tuple(int(x) for x in args.levels.split(","))
    model = build_tokenizer(args.arch, **tok_cfg).to(device)
    print(f"tokenizer '{args.arch}' parameters: {count_parameters(model) / 1e6:.2f}M"
          f"  codebook={model.codebook_size}  loss={'stack' if args.loss_stack else args.loss}")

    # The research loss stack (saliency-L1 + edge + FFL + optional LPIPS) vs the
    # baseline pixel(+edge) loss, selected by --loss-stack.
    lpips_fn = make_lpips(device) if (args.loss_stack and args.lpips_weight > 0) else None

    def loss_fn(recon, frames):
        if args.loss_stack:
            total, _ = tokenizer_loss(recon, frames, saliency_alpha=args.saliency_alpha,
                                      w_edge=args.grad_weight, w_ffl=args.ffl_weight,
                                      w_lpips=args.lpips_weight, lpips_fn=lpips_fn)
            return total
        return reconstruction_loss(recon, frames, kind=args.loss, grad_weight=args.grad_weight)

    if args.smoke:
        frames = torch.rand(4, 3, 64, 64, device=device)
        recon, indices, _ = model(frames)
        loss = loss_fn(recon, frames)
        loss.backward()
        assert recon.shape == frames.shape and indices.shape == (4, model.tokens_per_frame)
        print(f"[smoke] recon {recon.shape}, loss {loss.item():.4f} — OK")
        return

    # Two data paths: a fast in-RAM frame cache (unique frames on the GPU, no
    # DataLoader) or the sequence-window dataset. epoch_batches() yields (B,3,64,64)
    # frame batches either way so the training loop below is identical.
    if args.frame_cache:
        all_frames = load_all_frames(args.data, device)
        n = all_frames.shape[0]
        steps_per_epoch = max(1, n // args.batch_size)
        print(f"frame cache: {n} frames on {device}, {steps_per_epoch} steps/epoch")

        def epoch_batches():
            perm = torch.randperm(n, device=device)
            for i in range(steps_per_epoch):
                yield all_frames[perm[i * args.batch_size:(i + 1) * args.batch_size]]
    else:
        dataset = SimSequenceDataset(
            os.path.join(args.data, "manifest.json"),
            context=args.context,
            horizon=args.horizon,
            representation="rgb",
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        steps_per_epoch = len(loader)

        def epoch_batches():
            for item in loader:
                yield frames_from_batch(item).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.95))

    # Cosine LR with linear warmup, and EMA of the weights (the research's cheapest
    # quality bump). Eval/checkpoints use the EMA weights. FSQ has no learnable
    # codebook, so EMA here means EMA of the *network* weights.
    import math
    total_steps = args.epochs * max(1, steps_per_epoch)
    sched = None
    if args.cosine:
        warm = max(1, int(args.warmup_frac * total_steps))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (
            s / warm if s < warm else
            0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total_steps - warm)))))
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()} if args.ema > 0 else None

    # Resume: reload model + optimizer + epoch so --epochs is a total, and
    # training picks up where the checkpoint left off.
    start_epoch = 0
    if args.resume:
        prev = torch.load(args.resume, map_location=device)
        model.load_state_dict(prev.get("model_raw", prev["model"]))
        if ema is not None and "model" in prev:
            ema = {k: v.detach().clone().to(device) for k, v in prev["model"].items()}
        if "opt" in prev:
            opt.load_state_dict(prev["opt"])
        start_epoch = prev.get("epoch", 0)
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    ckpt = os.path.join(args.out, "tokenizer.pt")
    os.makedirs(args.out, exist_ok=True)
    for epoch in range(start_epoch, args.epochs):
        running = 0.0
        for frames in epoch_batches():
            recon, _, _ = model(frames)
            loss = loss_fn(recon, frames)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if sched is not None:
                sched.step()
            if ema is not None:
                with torch.no_grad():
                    for k, v in model.state_dict().items():
                        if v.dtype.is_floating_point:
                            ema[k].mul_(args.ema).add_(v.detach(), alpha=1 - args.ema)
                        else:
                            ema[k].copy_(v)
            running += loss.item()
        avg = running / max(1, steps_per_epoch)
        label = "stack" if args.loss_stack else args.loss
        lr_now = opt.param_groups[0]["lr"]
        print(f"epoch {epoch + 1}/{args.epochs}  loss_{label} {avg:.4f}  lr {lr_now:.2e}")
        # Checkpoint every epoch so a long run is resumable if interrupted.
        # builder + cfg let registry.load_tokenizer rebuild any variant exactly.
        # Save the EMA weights as "model" when EMA is on (they eval better); keep
        # the raw weights under "model_raw" so a resume continues the live model.
        save_model = ema if ema is not None else model.state_dict()
        torch.save({"builder": args.arch, "cfg": tok_cfg, "hidden": args.hidden,
                    "model": save_model, "model_raw": model.state_dict(),
                    "opt": opt.state_dict(), "epoch": epoch + 1}, ckpt)
    print(f"saved {ckpt}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/seed1")
    p.add_argument("--arch", default="fsq", help="registered tokenizer name (default: fsq)")
    p.add_argument("--out", default="checkpoints")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--loss", choices=["l1", "mse"], default="l1")
    p.add_argument("--grad-weight", type=float, default=0.0, dest="grad_weight",
                   help="weight on the gradient/edge loss term; >0 preserves small objects (the car)")
    p.add_argument("--loss-stack", action="store_true", dest="loss_stack",
                   help="use the research loss stack: saliency-weighted L1 + edge + FFL + LPIPS")
    p.add_argument("--ffl-weight", type=float, default=0.1, dest="ffl_weight")
    p.add_argument("--lpips-weight", type=float, default=0.1, dest="lpips_weight")
    p.add_argument("--saliency-alpha", type=float, default=2.0, dest="saliency_alpha")
    p.add_argument("--levels", default=None, help="override FSQ levels, e.g. '8,5,5,5' (vocab 1024)")
    p.add_argument("--cosine", action="store_true", help="cosine LR decay (to 5%) with linear warmup")
    p.add_argument("--warmup-frac", type=float, default=0.03, dest="warmup_frac")
    p.add_argument("--ema", type=float, default=0.0, help="weight-EMA decay (e.g. 0.999); 0 disables")
    p.add_argument("--frame-cache", action="store_true", dest="frame_cache",
                   help="load all unique frames into a GPU tensor (fast; skips the sequence dataset)")
    p.add_argument("--context", type=int, default=4)
    p.add_argument("--horizon", type=int, default=6)
    p.add_argument("--resume", default=None, help="checkpoint to continue training from")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--smoke", action="store_true", help="tiny random-tensor pass, no data")
    train(p.parse_args())


if __name__ == "__main__":
    main()
