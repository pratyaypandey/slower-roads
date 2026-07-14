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


def load_all_frames(data_dirs, device):
    """Load every frame from one or more seed dirs into a (N,3,64,64) device tensor,
    plus a boolean `has_next` mask (True where frame i+1 is the *same seed*'s
    temporal successor). The mask is what the temporal-consistency loss consumes —
    it must never pair the last frame of one seed with the first of the next.

    Frames are cached on-device (the whole multi-seed set is a few hundred MB) so
    training is compute-bound, not disk-bound.
    """
    import numpy as np
    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]
    arrs, has_next = [], []
    for d in data_dirs:
        manifest = json.load(open(os.path.join(d, "manifest.json")))
        samples = manifest["samples"]
        for j, s in enumerate(samples):
            arrs.append(np.load(os.path.join(d, s["frame"])))
            has_next.append(j < len(samples) - 1)  # last frame of this seed has no successor
    frames = torch.from_numpy(np.stack(arrs)).float().to(device)
    return frames, torch.tensor(has_next, device=device)


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

    # The temporal + noise consistency terms need consecutive-frame pairs, which
    # only the in-RAM frame cache (frames in temporal order) provides — so the
    # consistency objective requires --frame-cache. Each batch yields (x, x_next)
    # where x_next is x's same-seed successor (x_next is None on the DataLoader path).
    temporal_on = args.temporal_weight > 0
    if args.frame_cache:
        all_frames, has_next = load_all_frames(args.data, device)
        n = all_frames.shape[0]
        pair_idx = torch.nonzero(has_next, as_tuple=False).squeeze(1)  # i's with a successor
        steps_per_epoch = max(1, n // args.batch_size)
        print(f"frame cache: {n} frames on {device} ({len(pair_idx)} consecutive pairs), "
              f"{steps_per_epoch} steps/epoch")

        def epoch_batches():
            # Sample from frames that have a successor so (x, x_next) is always valid.
            src = pair_idx if temporal_on else torch.arange(n, device=device)
            perm = src[torch.randperm(len(src), device=device)]
            for i in range(steps_per_epoch):
                idx = perm[i * args.batch_size:(i + 1) * args.batch_size]
                if len(idx) == 0:
                    continue
                yield all_frames[idx], (all_frames[idx + 1] if temporal_on else None)
    else:
        if temporal_on:
            raise SystemExit("--temporal-weight needs --frame-cache (temporal frame order)")
        dataset = SimSequenceDataset(
            os.path.join(args.data[0], "manifest.json"),
            context=args.context,
            horizon=args.horizon,
            representation="rgb",
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        steps_per_epoch = len(loader)

        def epoch_batches():
            for item in loader:
                yield frames_from_batch(item).to(device), None

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
    # training picks up where the checkpoint left off. --reset-epoch loads the
    # *weights* but restarts the epoch counter at 0 — for fine-tuning a converged
    # tokenizer under a NEW objective (the temporal/noise consistency retrain),
    # where --epochs means "this many fresh epochs", not a global total.
    start_epoch = 0
    if args.resume:
        prev = torch.load(args.resume, map_location=device)
        model.load_state_dict(prev.get("model_raw", prev["model"]))
        if ema is not None and "model" in prev:
            ema = {k: v.detach().clone().to(device) for k, v in prev["model"].items()}
        if "opt" in prev and not args.reset_epoch:
            opt.load_state_dict(prev["opt"])
        start_epoch = 0 if args.reset_epoch else prev.get("epoch", 0)
        print(f"resumed from {args.resume} "
              f"({'weights only, fresh epochs' if args.reset_epoch else f'at epoch {start_epoch}'})")

    ckpt = os.path.join(args.out, "tokenizer.pt")
    os.makedirs(args.out, exist_ok=True)
    for epoch in range(start_epoch, args.epochs):
        running = {}
        for frames, frames_next in epoch_batches():
            recon, _, z = model(frames)                       # z = continuous pre-quant code
            rec = loss_fn(recon, frames)
            parts = {"rec": rec}
            # Noise robustness: encoding must not flip codes under a tiny pixel
            # perturbation (the measured failure: 1% noise flipped ~57% of tokens).
            # Penalizing the pre-quant code change pushes the encoder off the FSQ
            # quantization boundaries where flips happen.
            if args.noise_weight > 0:
                x_noisy = (frames + args.noise_std * torch.randn_like(frames)).clamp(0, 1)
                z_noisy = model.encode(x_noisy)
                parts["noise"] = args.noise_weight * (z - z_noisy).abs().mean()
            # Temporal consistency: near-identical consecutive frames should get
            # near-identical codes (the measured failure: 84% of tokens changed
            # between frames that were 99.3% identical). Penalize the code change
            # between successive frames; reconstruction still forces codes to move
            # where content actually changes.
            if frames_next is not None:
                z_next = model.encode(frames_next)
                parts["temporal"] = args.temporal_weight * (z - z_next).abs().mean()
            loss = sum(parts.values())
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
            for k, v in parts.items():
                running[k] = running.get(k, 0.0) + v.item()
        sp = max(1, steps_per_epoch)
        avg = {k: running[k] / sp for k in running}
        label = "stack" if args.loss_stack else args.loss
        lr_now = opt.param_groups[0]["lr"]
        parts_str = "  ".join(f"{k} {avg[k]:.4f}" for k in avg)
        print(f"epoch {epoch + 1}/{args.epochs}  {parts_str}  (rec={label})  lr {lr_now:.2e}")
        # Checkpoint every epoch so a long run is resumable if interrupted.
        # builder + cfg let registry.load_tokenizer rebuild any variant exactly.
        # Save the EMA weights as "model" when EMA is on (they eval better); keep
        # the raw weights under "model_raw" so a resume continues the live model.
        save_model = ema if ema is not None else model.state_dict()
        torch.save({"builder": args.arch, "cfg": tok_cfg, "hidden": args.hidden,
                    "model": save_model, "model_raw": model.state_dict(),
                    "opt": opt.state_dict(), "epoch": epoch + 1}, ckpt)
    print(f"saved {ckpt}")


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", default=["data/seed1"],
                   help="one or more seed dirs (frames concatenated; temporal pairs stay in-seed)")
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
    # Temporal + noise consistency (fix the tokenizer's temporal instability that
    # breaks M2 — see docs/M2_RESULTS.md). Weights are on the pre-quant code L1.
    p.add_argument("--temporal-weight", type=float, default=0.0, dest="temporal_weight",
                   help="penalize code change between consecutive frames (needs --frame-cache)")
    p.add_argument("--noise-weight", type=float, default=0.0, dest="noise_weight",
                   help="penalize code change under a small pixel perturbation")
    p.add_argument("--noise-std", type=float, default=0.02, dest="noise_std",
                   help="std of the pixel noise for the robustness term")
    p.add_argument("--context", type=int, default=4)
    p.add_argument("--horizon", type=int, default=6)
    p.add_argument("--resume", default=None, help="checkpoint to continue training from")
    p.add_argument("--reset-epoch", action="store_true", dest="reset_epoch",
                   help="on resume, load weights but restart epoch/opt at 0 (fine-tune a new objective)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--smoke", action="store_true", help="tiny random-tensor pass, no data")
    return p


def main(argv=None):
    train(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
