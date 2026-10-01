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

Real runs get: a held-out tail split for validation, cosine LR + warmup, grad
clipping, CUDA AMP, a best-on-val checkpoint (`dynamics_best.pt`) alongside the
latest (`dynamics.pt`, for resume), and one JSON line per epoch to
`dynamics_metrics.jsonl`. CPU-shape-testable with --smoke (random tensors).
"""

import argparse
import inspect
import json
import math
import os
import random

import torch
from torch.utils.data import DataLoader, ConcatDataset

from model.tokenizer.fsq_autoencoder import FSQAutoencoder
from model.registry import build_dynamics, load_tokenizer
from model.data.dataset import SimSequenceDataset, load_manifest
from model.dynamics.sequence import action_to_vocab
from model.dynamics.config import FRAME_STRIDE


def load_frozen_tokenizer(path, device):
    # Rebuild via the registry so a non-default tokenizer (variant/size) reloads
    # from its saved cfg. Back-compat default cfg for old checkpoints.
    tok, _ = load_tokenizer(path, default_cfg={"hidden": 64}, map_location=device)
    tok = tok.to(device).eval()
    for p in tok.parameters():
        p.requires_grad_(False)
    return tok


def lr_at(step, total_steps, base_lr, warmup_steps, min_frac=0.1):
    """Linear warmup for `warmup_steps`, then cosine decay to base_lr*min_frac."""
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    prog = min(1.0, prog)
    return base_lr * (min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * prog)))


def _batch_kw(model, args, tf, accepts_tf):
    kw = dict(ce_weight=args.ce_weight, pixel_weight=args.pixel_weight)
    if accepts_tf:
        kw["teacher_forcing"] = tf
    if "self_rollout" in inspect.signature(model.prepare_batch).parameters:
        kw["self_rollout"] = args.self_rollout
        kw["context_window"] = args.context
        kw["anchor_lambda_min"] = args.anchor_lambda_min
        kw["anchor_lambda_max"] = args.anchor_lambda_max
        kw["corruption_min"] = args.corruption_min
        kw["corruption_max"] = args.corruption_max
    if "state_weight" in inspect.signature(model.prepare_batch).parameters:
        kw["state_weight"] = args.state_weight
    return kw


@torch.no_grad()
def evaluate(model, tokenizer, loader, args, device):
    """Mean loss parts over the val split (no teacher forcing — measure the model
    on its own terms). Returns {} if the split has no windows."""
    if loader is None or len(loader) == 0:
        return {}
    model.eval()
    run, n = {}, 0
    for item in loader:
        kw = dict(ce_weight=args.ce_weight, pixel_weight=args.pixel_weight)
        if "anchor_lambda_min" in inspect.signature(model.prepare_batch).parameters:
            kw.update(anchor_lambda_min=args.anchor_val_lambda,
                      anchor_lambda_max=args.anchor_val_lambda)
        batch = model.prepare_batch(tokenizer, item, args.horizon, device, **kw)
        _, parts = model.loss(batch, tokenizer.decode_indices)
        for k, v in parts.items():
            run[k] = run.get(k, 0.0) + v.item()
        n += 1
    model.train()
    return {k: run[k] / max(1, n) for k in run}


@torch.no_grad()
def evaluate_free_run(model, tokenizer, loader, args, device):
    """Exact bounded-context rollout error for honest checkpoint selection."""
    if loader is None or args.free_run_val_batches <= 0:
        return {}
    model.eval()
    correct = total = 0
    for batch_idx, item in enumerate(loader):
        if batch_idx >= args.free_run_val_batches:
            break
        kw = dict(ce_weight=1.0, pixel_weight=0.0)
        if "anchor_lambda_min" in inspect.signature(model.prepare_batch).parameters:
            kw.update(anchor_lambda_min=args.anchor_val_lambda,
                      anchor_lambda_max=args.anchor_val_lambda)
        batch = model.prepare_batch(tokenizer, item, args.horizon, device, **kw)
        if not {"z_ctx", "action_ids", "target_tokens"}.issubset(batch):
            model.train()
            return {}
        prefix = batch["z_ctx"]
        anchor_lambda = batch.get("anchor_lambda", 0.0)
        anchor_prefix = None
        if getattr(model, "anchor_encoder", None) is not None:
            anchor_prefix = model.anchor_sequence(batch["context_anchor"], anchor_lambda)
        for k in range(args.horizon):
            action = batch["action_ids"][:, k]
            current_anchor = batch.get("target_anchor")
            current_anchor = None if current_anchor is None else current_anchor[:, k]
            pred = model.generate_frame(
                prefix, action, context_anchor_emb=anchor_prefix,
                anchor=current_anchor, anchor_lambda=anchor_lambda,
            )
            target = batch["target_tokens"][:, k]
            correct += int((pred == target).sum())
            total += target.numel()
            prefix = torch.cat(
                [prefix, action_to_vocab(action).unsqueeze(1), pred], dim=1
            )
            max_tokens = args.context * FRAME_STRIDE
            if prefix.shape[1] > max_tokens:
                prefix = prefix[:, -max_tokens:]
            if anchor_prefix is not None:
                current_frame = model.anchor_frames(
                    current_anchor[:, None], anchor_lambda
                )[:, 0]
                anchor_prefix = torch.cat([anchor_prefix, current_frame], dim=1)
                if anchor_prefix.shape[1] > max_tokens:
                    anchor_prefix = anchor_prefix[:, -max_tokens:]
    model.train()
    if total == 0:
        return {}
    acc = correct / total
    return {"free_run_error": 1.0 - acc, "free_run_acc": acc}


def train(args, on_epoch_end=None):
    # on_epoch_end(epoch) is called after each epoch's checkpoints are written —
    # the Modal wrapper passes vol.commit so partial runs persist to the volume
    # (durability + mid-run `modal volume get`), and it's a no-op locally.
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    use_amp = args.amp and device.type == "cuda"
    # Build via the registry so --arch selects the dynamics core; cfg is saved
    # with the checkpoint so it reloads exactly (default = ar_transformer).
    dyn_cfg = {"d_model": args.d_model, "n_heads": args.n_heads,
               "n_layers": args.n_layers, "dropout": args.dropout,
               "action_cond": args.action_cond,
               "anchor_cond": args.anchor_cond,
               "anchor_injection": args.anchor_injection,
               "corruption_cond": args.corruption_cond,
               "attention_recency_bias": args.attention_recency_bias,
               "mem_cross_attn": args.mem_cross_attn,
               "mem_tokens": args.mem_tokens,
               "state_head": args.state_head}
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
        if args.anchor_cond or args.mem_cross_attn:
            item["context_anchor"] = torch.rand(B, T, 16, 16, 16)
            item["target_anchor"] = torch.rand(B, H, 16, 16, 16)
        if args.state_head:
            item["context_state"] = torch.rand(B, T, 4)
            item["target_state"] = torch.rand(B, H, 4)
        smoke_kw = {}
        if "self_rollout" in inspect.signature(model.prepare_batch).parameters:
            smoke_kw.update(
                self_rollout=args.self_rollout,
                context_window=args.context,
                anchor_lambda_min=args.anchor_val_lambda,
                anchor_lambda_max=args.anchor_val_lambda,
                corruption_min=args.corruption_min,
                corruption_max=args.corruption_max,
            )
        if "state_weight" in inspect.signature(model.prepare_batch).parameters:
            smoke_kw["state_weight"] = args.state_weight
        batch = model.prepare_batch(tok, item, H, device, **smoke_kw)
        total, parts = model.loss(batch, tok.decode_indices)
        total.backward()
        parts_str = " ".join(f"{k} {v.item():.4f}" for k, v in parts.items())
        print(f"[smoke] {args.arch} loss {total.item():.4f} ({parts_str}) — OK")
        return

    tokenizer = load_frozen_tokenizer(args.tokenizer, device)
    rep = "latent" if args.latent else "rgb"

    def seed_ds(data_dir, sample_range=None):
        return SimSequenceDataset(os.path.join(data_dir, "manifest.json"),
                                  context=args.context, horizon=args.horizon,
                                  representation=rep, sample_range=sample_range,
                                  include_anchor=args.anchor_cond or args.mem_cross_attn,
                                  include_state=args.state_head)

    # Two validation modes:
    #  * --val-data SEED  -> validate on a whole held-out trajectory (the honest
    #    generalization test: an entirely unseen seed). Train on all --data seeds.
    #  * else             -> tail-split the (single) --data trajectory (leakage-free
    #    via window_indices' sample_range), for a quick same-seed sanity val.
    if args.val_data:
        dataset = ConcatDataset([seed_ds(d) for d in args.data])
        val_ds = seed_ds(args.val_data)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                drop_last=False) if len(val_ds) else None
        print(f"train seeds: {args.data}  ({len(dataset)} windows)  |  "
              f"val seed: {args.val_data}  ({len(val_ds)} windows)")
    else:
        d0 = args.data[0]
        n_samples = len(load_manifest(os.path.join(d0, "manifest.json"))[0]["samples"])
        gap = args.context + args.horizon
        split = int(n_samples * (1 - args.val_frac))
        dataset = seed_ds(d0, (0, split) if args.val_frac > 0 else None)
        val_loader = None
        if args.val_frac > 0:
            val_ds = seed_ds(d0, (split + gap, n_samples))
            if len(val_ds) > 0:
                val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                        drop_last=False)
        print(f"train windows: {len(dataset)}  val windows: "
              f"{0 if val_loader is None else len(val_loader.dataset)}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # AdamW so --weight-decay actually regularizes (decoupled). The 11M core
    # overfits ~10 seeds by epoch 4; weight decay lets it train longer on the
    # generalizable-motion signal before memorizing.
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Resume: reload dynamics model + optimizer + epoch (the tokenizer is always
    # loaded fresh and frozen, so it's not part of the resume state).
    start_epoch = 0
    if args.init_from:
        prev = torch.load(args.init_from, map_location=device)
        missing, unexpected = model.load_state_dict(prev["model"], strict=False)
        print(
            f"initialized weights from {args.init_from}; "
            f"new params={len(missing)} unexpected={len(unexpected)}"
        )
    if args.resume:
        prev = torch.load(args.resume, map_location=device)
        model.load_state_dict(prev["model"])
        if "opt" in prev:
            opt.load_state_dict(prev["opt"])
        start_epoch = prev.get("epoch", 0)
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    os.makedirs(args.out, exist_ok=True)
    ckpt = os.path.join(args.out, "dynamics.pt")
    best_ckpt = os.path.join(args.out, "dynamics_best.pt")
    metrics_path = os.path.join(args.out, "dynamics_metrics.jsonl")

    # Scheduled sampling: anneal teacher forcing tf_start -> 0 across training, so
    # early epochs are grounded (stable) and later ones free-run (learn to
    # self-correct). tf_start=0 (default) is pure free-running throughout.
    accepts_tf = "teacher_forcing" in inspect.signature(model.prepare_batch).parameters

    steps_per_epoch = max(
        1,
        min(len(loader), args.max_train_batches)
        if args.max_train_batches > 0 else len(loader),
    )
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_steps if args.warmup_steps is not None \
        else int(args.warmup_frac * total_steps)
    step = start_epoch * steps_per_epoch
    # Best-checkpoint selection keys on the *validation* metric whenever a val
    # split exists — never on train loss. Mixing the two clobbers the val optimum
    # (train CE keeps falling past the point val starts rising = overfitting), so
    # the "best" would just track the most-overfit epoch. Only with no val split
    # do we fall back to train. stale_evals counts consecutive validations with no
    # improvement, for early stopping.
    have_val = val_loader is not None
    best_sel = math.inf
    stale_evals = 0

    # A resumed run must retain the historical selection threshold. Otherwise
    # its first validation overwrites dynamics_best.pt even when it is worse.
    if start_epoch > 0 and os.path.exists(metrics_path):
        with open(metrics_path) as f:
            for line in f:
                row = json.loads(line)
                if "val_free_run_error" in row:
                    best_sel = min(best_sel, row["val_free_run_error"])
                elif "val_ce" in row:
                    best_sel = min(best_sel, row["val_ce"])
        if math.isfinite(best_sel):
            print(f"restored historical best selection metric: {best_sel:.4f}")

    def _save(path, epoch):
        torch.save({"builder": args.arch, "cfg": dyn_cfg,
                    "model": model.state_dict(), "opt": opt.state_dict(),
                    "epoch": epoch}, path)

    def _sel(parts):
        if not parts:
            return math.inf
        return parts.get("free_run_error", parts.get("ce", sum(parts.values())))

    model.train()
    for epoch in range(start_epoch, args.epochs):
        frac = epoch / max(1, args.epochs - 1)
        tf = args.tf_start * (1 - frac)
        run = {}
        last_lr = args.lr
        for batch_idx, item in enumerate(loader):
            if args.max_train_batches > 0 and batch_idx >= args.max_train_batches:
                break
            last_lr = lr_at(step, total_steps, args.lr, warmup_steps, args.min_lr_frac)
            for g in opt.param_groups:
                g["lr"] = last_lr
            kw = _batch_kw(model, args, tf, accepts_tf)
            batch = model.prepare_batch(tokenizer, item, args.horizon, device, **kw)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                total, parts = model.loss(batch, tokenizer.decode_indices)
            scaler.scale(total).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            for k, v in parts.items():
                run[k] = run.get(k, 0.0) + v.item()
            step += 1

        n = steps_per_epoch
        train_parts = {k: run[k] / n for k in run}
        do_val = val_loader is not None and ((epoch + 1) % args.eval_every == 0
                                             or epoch + 1 == args.epochs)
        val_parts = evaluate(model, tokenizer, val_loader, args, device) if do_val else {}
        if do_val:
            val_parts.update(evaluate_free_run(model, tokenizer, val_loader, args, device))

        train_str = "  ".join(f"{k} {v:.4f}" for k, v in train_parts.items())
        val_str = ("  ".join(f"val_{k} {v:.4f}" for k, v in val_parts.items())) or "—"
        tf_str = f"  tf {tf:.2f}" if accepts_tf and args.tf_start > 0 else ""
        print(f"epoch {epoch + 1}/{args.epochs}  lr {last_lr:.2e}  {train_str}"
              f"{tf_str}  |  {val_str}")

        with open(metrics_path, "a") as f:
            row = {"epoch": epoch + 1, "lr": last_lr}
            row.update({f"train_{k}": v for k, v in train_parts.items()})
            row.update({f"val_{k}": v for k, v in val_parts.items()})
            f.write(json.dumps(row) + "\n")

        # Latest every epoch (for resume). Best-on-val: only reconsider on epochs
        # that actually validated (or, with no val split, every epoch on train).
        _save(ckpt, epoch + 1)
        considered = val_parts if have_val else train_parts
        if considered:
            sel = _sel(considered)
            if sel < best_sel:
                best_sel = sel
                stale_evals = 0
                _save(best_ckpt, epoch + 1)
                print(f"  new best ({'val' if have_val else 'train'} {sel:.4f}) -> {best_ckpt}")
            else:
                stale_evals += 1

        if on_epoch_end is not None:
            on_epoch_end(epoch + 1)

        # Early stop: val hasn't improved for `patience` validations — past the
        # optimum, only overfitting from here. Disabled with --patience 0.
        if have_val and args.patience > 0 and stale_evals >= args.patience:
            print(f"early stop: no val improvement for {stale_evals} validations "
                  f"(best {best_sel:.4f})")
            break

    print(f"saved {ckpt}  (best {best_ckpt}, metrics {metrics_path})")


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", default=["data/seed1"],
                   help="one or more seed dirs to train on (concatenated)")
    p.add_argument("--val-data", default=None, dest="val_data",
                   help="held-out seed dir to validate on (whole trajectory); "
                        "the honest generalization test. Overrides --val-frac tail split")
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
    p.add_argument("--dropout", type=float, default=0.0, help="residual + embedding dropout (regularization)")
    p.add_argument("--action-cond", action="store_true", dest="action_cond",
                   help="strong action conditioning: add the action to every frame position")
    p.add_argument("--anchor-cond", action="store_true", dest="anchor_cond",
                   help="condition every visual token on the sim skeleton anchor")
    p.add_argument("--anchor-injection", choices=["input", "output"], default="input",
                   dest="anchor_injection",
                   help="inject anchor before attention (input) or only before logits (output)")
    p.add_argument("--anchor-lambda-min", type=float, default=0.0,
                   dest="anchor_lambda_min")
    p.add_argument("--anchor-lambda-max", type=float, default=1.0,
                   dest="anchor_lambda_max")
    p.add_argument("--anchor-val-lambda", type=float, default=1.0,
                   dest="anchor_val_lambda")
    p.add_argument("--corruption-cond", action="store_true", dest="corruption_cond",
                   help="train with FSQ-local context corruption and per-frame severity embeddings")
    p.add_argument("--corruption-min", type=float, default=0.0,
                   dest="corruption_min")
    p.add_argument("--corruption-max", type=float, default=0.25,
                   dest="corruption_max")
    p.add_argument("--attention-recency-bias", type=float, default=0.0,
                   dest="attention_recency_bias",
                   help="initial learned per-head attention penalty per frame of age")
    p.add_argument("--mem-cross-attn", action="store_true", dest="mem_cross_attn",
                   help="cross-attend visual tokens into dedicated skeleton memory "
                        "tokens (decoupled from the causal visual KV cache)")
    p.add_argument("--mem-tokens", type=int, default=16, dest="mem_tokens",
                   help="number of skeleton memory tokens (rounded to a square grid)")
    p.add_argument("--state-head", action="store_true", dest="state_head",
                   help="auxiliary head regressing the per-frame sim-state delta "
                        "(continuity objective on the cache)")
    p.add_argument("--state-weight", type=float, default=0.5, dest="state_weight",
                   help="weight of the state-continuity loss term (0 = off)")
    p.add_argument("--ce-weight", type=float, default=1.0, dest="ce_weight")
    p.add_argument("--pixel-weight", type=float, default=1.0, dest="pixel_weight")
    p.add_argument("--tf-start", type=float, default=0.0, dest="tf_start",
                   help="initial teacher-forcing prob, annealed to 0 (0 = pure free-run)")
    p.add_argument("--self-rollout", action="store_true", dest="self_rollout",
                   help="condition future training steps on exact KV-cached generated frames")
    # Real-run hardening knobs (all defaulted so the documented invocation works).
    p.add_argument("--val-frac", type=float, default=0.15, dest="val_frac",
                   help="held-out tail fraction for validation (0 = no val split)")
    p.add_argument("--eval-every", type=int, default=2, dest="eval_every",
                   help="run validation every N epochs")
    p.add_argument("--free-run-val-batches", type=int, default=0,
                   dest="free_run_val_batches",
                   help="exact KV-cached val batches for checkpoint selection (0 = off)")
    p.add_argument("--max-train-batches", type=int, default=0,
                   dest="max_train_batches",
                   help="cap batches per epoch for smoke runs (0 = full epoch)")
    p.add_argument("--patience", type=int, default=5, dest="patience",
                   help="early-stop after this many validations with no val improvement (0 = off)")
    p.add_argument("--grad-clip", type=float, default=1.0, dest="grad_clip")
    p.add_argument("--weight-decay", type=float, default=0.0, dest="weight_decay",
                   help="AdamW decoupled weight decay (regularization; 0 = off)")
    p.add_argument("--warmup-frac", type=float, default=0.03, dest="warmup_frac",
                   help="fraction of total steps spent in linear LR warmup")
    p.add_argument("--warmup-steps", type=int, default=None, dest="warmup_steps",
                   help="explicit warmup steps (overrides --warmup-frac)")
    p.add_argument("--min-lr-frac", type=float, default=0.1, dest="min_lr_frac",
                   help="cosine floor as a fraction of --lr")
    p.add_argument("--amp", action="store_true", help="CUDA mixed precision (no-op on CPU)")
    p.add_argument("--latent", action="store_true",
                   help="train from precomputed latents.npy (no tokenizer/frames at "
                        "train time; ~10x faster). Requires model.precompute_latents first")
    p.add_argument("--resume", default=None, help="checkpoint to continue training from")
    p.add_argument("--init-from", default=None, dest="init_from",
                   help="warm-start model weights only (allows new anchor/corruption params)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0,
                   help="training RNG seed (model init, shuffling, corruption, lambda sampling)")
    p.add_argument("--smoke", action="store_true", help="tiny random-tensor pass, no data")
    return p


def main(argv=None, on_epoch_end=None):
    train(build_parser().parse_args(argv), on_epoch_end=on_epoch_end)


if __name__ == "__main__":
    main()
