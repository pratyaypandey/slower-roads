"""M2: train a dynamics core on latents from a frozen tokenizer.

Arch-agnostic: the selected core (--arch, default ar_transformer) owns how it
turns a dataset item into its training batch (`prepare_batch`) and its loss
(`loss`), so this trainer just calls those two protocol methods. The AR core uses
interleaved [action, visual] token sequences with a multi-step rollout loss (§5,
token CE + decoded pixel); the flow bridge uses continuous FSQ-code transitions.
The frozen tokenizer's decode_indices is passed as the decoder for pixel terms.

    python -m model.train_tokenizer --data data/seed1 --epochs 20
    python -m model.train_dynamics --data data/seed1 --tokenizer checkpoints/tokenizer.pt
    python -m model.train_dynamics --arch flow_bridge --data data/seed1 --tokenizer checkpoints/tokenizer.pt

CPU-shape-testable with --smoke (random tensors, no data or checkpoint needed).
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader

from model.tokenizer.fsq_autoencoder import FSQAutoencoder
from model.registry import build_dynamics, load_tokenizer
from model.data.dataset import SimSequenceDataset


def load_frozen_tokenizer(path, device):
    # Rebuild via the registry so a non-default tokenizer (variant/size) reloads
    # from its saved cfg. Back-compat default cfg for old checkpoints.
    tok, _ = load_tokenizer(path, default_cfg={"hidden": 64}, map_location=device)
    tok = tok.to(device).eval()
    for p in tok.parameters():
        p.requires_grad_(False)
    return tok


def train(args):
    device = torch.device(args.device)
    # Build via the registry so --arch selects the dynamics core; cfg is saved
    # with the checkpoint so it reloads exactly (default = ar_transformer).
    dyn_cfg = {"d_model": args.d_model, "n_heads": args.n_heads, "n_layers": args.n_layers}
    model = build_dynamics(args.arch, **dyn_cfg).to(device)
    print(f"dynamics '{args.arch}' parameters: {model.param_count() / 1e6:.2f}M")

    if args.smoke:
        tok = FSQAutoencoder().to(device).eval()
        B, T, H = 2, args.context, args.horizon
        item = {
            "context_frames": torch.rand(B, T, 3, 64, 64),
            "target_frames": torch.rand(B, H, 3, 64, 64),
            "context_actions": torch.randint(0, 9, (B, T)),
            "target_actions": torch.randint(0, 9, (B, H)),
        }
        batch = model.prepare_batch(tok, item, H, device)
        total, parts = model.loss(batch, tok.decode_indices)
        total.backward()
        parts_str = " ".join(f"{k} {v.item():.4f}" for k, v in parts.items())
        print(f"[smoke] {args.arch} loss {total.item():.4f} ({parts_str}) — OK")
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
    # Scheduled sampling: anneal teacher forcing tf_start -> 0 across training, so
    # early epochs are grounded (stable) and later ones free-run (learn to
    # self-correct). tf_start=0 (default) is pure free-running throughout.
    import inspect
    accepts_tf = "teacher_forcing" in inspect.signature(model.prepare_batch).parameters

    for epoch in range(start_epoch, args.epochs):
        frac = epoch / max(1, args.epochs - 1)
        tf = args.tf_start * (1 - frac)
        # Accumulate whatever loss parts this arch reports (AR: ce/pixel; flow
        # bridge: flow/pixel) so the trainer doesn't assume a fixed set.
        run = {}
        for item in loader:
            kw = dict(ce_weight=args.ce_weight, pixel_weight=args.pixel_weight)
            if accepts_tf:
                kw["teacher_forcing"] = tf
            batch = model.prepare_batch(tokenizer, item, args.horizon, device, **kw)
            total, parts = model.loss(batch, tokenizer.decode_indices)
            opt.zero_grad()
            total.backward()
            opt.step()
            for k, v in parts.items():
                run[k] = run.get(k, 0.0) + v.item()
        n = max(1, len(loader))
        parts_str = "  ".join(f"{k} {run[k] / n:.4f}" for k in run)
        tf_str = f"  tf {tf:.2f}" if accepts_tf and args.tf_start > 0 else ""
        print(f"epoch {epoch + 1}/{args.epochs}  {parts_str}{tf_str}")
        # Checkpoint every epoch so long runs survive interruption + resume.
        # builder + cfg let registry.load_dynamics rebuild any arch/size exactly
        # (fixes the old bug where eval assumed default d_model/n_heads/n_layers).
        torch.save({"builder": args.arch, "cfg": dyn_cfg,
                    "model": model.state_dict(), "opt": opt.state_dict(),
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
    p.add_argument("--arch", default="ar_transformer", help="registered dynamics name")
    p.add_argument("--d-model", type=int, default=256, dest="d_model")
    p.add_argument("--n-heads", type=int, default=4, dest="n_heads")
    p.add_argument("--n-layers", type=int, default=4, dest="n_layers")
    p.add_argument("--ce-weight", type=float, default=1.0, dest="ce_weight")
    p.add_argument("--pixel-weight", type=float, default=1.0, dest="pixel_weight")
    p.add_argument("--tf-start", type=float, default=0.0, dest="tf_start",
                   help="initial teacher-forcing prob, annealed to 0 (0 = pure free-run)")
    p.add_argument("--resume", default=None, help="checkpoint to continue training from")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--smoke", action="store_true", help="tiny random-tensor pass, no data")
    train(p.parse_args())


if __name__ == "__main__":
    main()
