"""Long-horizon, baseline-relative M3 drift evaluation.

Runs the exact bounded-context KV-cached generator once per start/lambda and
records per-step pixel drift, token accuracy, normalized drift, and survival
time.  The normalization removes the tokenizer floor and scales frozen-frame
drift to 1.0, making curves comparable across seeds and tokenizers.
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from model.data.dataset import action_driving_frame, anchor_grid
from model.dynamics.config import FRAME_STRIDE
from model.dynamics.sequence import action_to_vocab, build_context
from model.registry import load_dynamics, load_tokenizer
from eval.eval_dream import action_id


def parse_numbers(value, cast):
    return [cast(x) for x in value.split(",") if x.strip()]


def load_window(data_dir, start, context, steps):
    manifest = json.load(open(os.path.join(data_dir, "manifest.json")))
    samples = manifest["samples"]
    end = start + context + steps
    if end > len(samples):
        raise ValueError(
            f"window [{start},{end}) exceeds {len(samples)} samples in {data_dir}"
        )
    indices = list(range(start, end))
    frames = np.stack([
        np.load(os.path.join(data_dir, samples[i]["frame"])) for i in indices
    ]).astype(np.float32)
    actions = np.array([
        action_id(action_driving_frame(samples, i)) for i in indices
    ], dtype=np.int64)
    anchors = None
    if all("skeleton" in samples[i] for i in indices):
        anchors = np.stack([anchor_grid(samples[i]["skeleton"]) for i in indices])
    return frames, actions, anchors


@torch.no_grad()
def encode_batches(tok, frames, device, batch_size):
    chunks = []
    for lo in range(0, len(frames), batch_size):
        x = torch.from_numpy(frames[lo:lo + batch_size]).to(device)
        _, idx, _ = tok(x)
        chunks.append(idx.cpu())
    return torch.cat(chunks)


@torch.no_grad()
def baseline_curves(tok, tokens, frames, context, device, batch_size):
    target = tokens[context:]
    previous = tokens[context - 1:-1]
    frozen_tok = tokens[context - 1:context].to(device)
    frozen_frame = tok.decode_indices(frozen_tok).cpu().numpy()[0]
    floor, persistence = [], []
    for lo in range(0, len(target), batch_size):
        hi = min(len(target), lo + batch_size)
        gt = frames[context + lo:context + hi]
        floor_hat = tok.decode_indices(target[lo:hi].to(device)).cpu().numpy()
        persist_hat = tok.decode_indices(previous[lo:hi].to(device)).cpu().numpy()
        floor.extend(np.abs(floor_hat - gt).mean(axis=(1, 2, 3)).tolist())
        persistence.extend(np.abs(persist_hat - gt).mean(axis=(1, 2, 3)).tolist())
    truth = frames[context:]
    frozen = np.abs(truth - frozen_frame[None]).mean(axis=(1, 2, 3))
    return np.asarray(floor), np.asarray(persistence), frozen


def first_sustained_failure(curve, width=30, threshold=1.0):
    bad = curve >= threshold
    if len(bad) < width:
        return len(bad)
    run = np.convolve(bad.astype(np.int32), np.ones(width, dtype=np.int32), mode="valid")
    hits = np.flatnonzero(run == width)
    return int(hits[0]) if len(hits) else len(curve)


@torch.no_grad()
def rollout(tok, dyn, tokens, frames, actions, anchors, context, window,
            lam, device, attention_every=0, progress_every=0):
    ctx_actions = torch.from_numpy(actions[:context]).to(device).unsqueeze(0)
    prefix = build_context(ctx_actions, tokens[:context].to(device).unsqueeze(0))
    anchor_prefix = None
    if getattr(dyn, "anchor_encoder", None) is not None:
        if anchors is None:
            raise ValueError("anchor-conditioned checkpoint needs skeletons in the manifest")
        raw_ctx = torch.from_numpy(anchors[:context]).float().to(device).unsqueeze(0)
        anchor_prefix = dyn.anchor_sequence(raw_ctx, lam)

    l1, acc, attention = [], [], []
    started = time.perf_counter()
    for k in range(len(frames) - context):
        frame_idx = context + k
        action = torch.tensor([int(actions[frame_idx])], device=device)
        current_anchor = None
        if anchor_prefix is not None:
            current_anchor = torch.from_numpy(anchors[frame_idx:frame_idx + 1]).float().to(device)
        pred = dyn.generate_frame(
            prefix, action, context_anchor_emb=anchor_prefix,
            anchor=current_anchor, anchor_lambda=lam,
            capture_final_attention=(attention_every > 0 and k % attention_every == 0),
        )
        if attention_every > 0 and k % attention_every == 0:
            attention.append({"step": k + 1, "layers": dyn.attention_profile()})
        hat = tok.decode_indices(pred).cpu().numpy()[0]
        l1.append(float(np.abs(hat - frames[frame_idx]).mean()))
        target = tokens[frame_idx:frame_idx + 1].to(device)
        acc.append(float((pred == target).float().mean()))
        if progress_every > 0 and (k + 1) % progress_every == 0:
            elapsed = time.perf_counter() - started
            print(
                f"  rollout {k + 1}/{len(frames) - context}: "
                f"{(k + 1) / max(elapsed, 1e-9):.2f} fps",
                flush=True,
            )
        prefix = torch.cat([prefix, action_to_vocab(action).unsqueeze(1), pred], dim=1)
        max_tokens = window * FRAME_STRIDE
        if prefix.shape[1] > max_tokens:
            prefix = prefix[:, -max_tokens:]
        if anchor_prefix is not None:
            current_frame = dyn.anchor_frames(current_anchor[:, None], lam)[:, 0]
            anchor_prefix = torch.cat([anchor_prefix, current_frame], dim=1)
            if anchor_prefix.shape[1] > max_tokens:
                anchor_prefix = anchor_prefix[:, -max_tokens:]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return np.asarray(l1), np.asarray(acc), time.perf_counter() - started, attention


def summarize(curve, acc, floor, frozen, horizons, seconds, lam, start):
    norm = (curve - floor) / np.maximum(frozen - floor, 1e-6)
    points = {}
    for h in horizons:
        if h <= len(curve):
            points[str(h)] = {
                "model_l1": float(curve[h - 1]),
                "normalized_drift": float(norm[h - 1]),
                "mean_normalized_drift": float(norm[:h].mean()),
                "token_acc": float(acc[h - 1]),
            }
    return {
        "start": start,
        "lambda": lam,
        "steps": len(curve),
        "seconds": seconds,
        "fps": len(curve) / max(seconds, 1e-9),
        "survival_frames": first_sustained_failure(norm),
        "mean_normalized_drift": float(norm.mean()),
        "mean_token_acc": float(acc.mean()),
        "horizons": points,
        "curves": {
            "model_l1": curve.tolist(),
            "token_acc": acc.tolist(),
            "normalized_drift": norm.tolist(),
        },
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--dynamics", required=True)
    p.add_argument("--starts", default="100")
    p.add_argument("--horizons", default="30,150,300,900,1800,3600")
    p.add_argument("--lambdas", default="0,0.25,0.5,0.75,1")
    p.add_argument("--context", type=int, default=8)
    p.add_argument("--window", type=int, default=8)
    p.add_argument("--encode-batch", type=int, default=64)
    p.add_argument("--attention-every", type=int, default=0,
                   help="capture final-token attention every N rollout frames (0=off)")
    p.add_argument("--progress-every", type=int, default=300,
                   help="print rollout progress every N frames (0=off)")
    p.add_argument("--out", default="eval/m3_metrics.json")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    starts = parse_numbers(args.starts, int)
    horizons = sorted(set(parse_numbers(args.horizons, int)))
    lambdas = parse_numbers(args.lambdas, float)
    max_steps = max(horizons)
    device = torch.device(args.device)
    tok, _ = load_tokenizer(args.tokenizer, default_cfg={"hidden": 128}, map_location=device)
    dyn, checkpoint = load_dynamics(args.dynamics, map_location=device)
    tok, dyn = tok.to(device).eval(), dyn.to(device).eval()
    if getattr(dyn, "anchor_encoder", None) is None:
        lambdas = [0.0]

    checkpoint_meta = {
        "builder": checkpoint.get("builder"),
        "cfg": checkpoint.get("cfg", {}),
        "epoch": checkpoint.get("epoch"),
    }
    results = {"config": vars(args), "checkpoint": checkpoint_meta, "runs": []}
    baseline_saved = False
    for start in starts:
        frames, actions, anchors = load_window(
            args.data, start, args.context, max_steps
        )
        tokens = encode_batches(tok, frames, device, args.encode_batch)
        floor, persistence, frozen = baseline_curves(
            tok, tokens, frames, args.context, device, args.encode_batch
        )
        if not baseline_saved:
            results["baselines"] = {
                "floor_l1": floor.tolist(),
                "persistence_l1": persistence.tolist(),
                "frozen_l1": frozen.tolist(),
            }
            baseline_saved = True
        for lam in lambdas:
            curve, acc, elapsed, attention = rollout(
                tok, dyn, tokens, frames, actions, anchors, args.context,
                args.window, lam, device, args.attention_every, args.progress_every,
            )
            run = summarize(curve, acc, floor, frozen, horizons, elapsed, lam, start)
            if attention:
                run["attention"] = attention
            results["runs"].append(run)
            print(
                f"start {start} lambda {lam:.2f}: norm-AUC "
                f"{run['mean_normalized_drift']:.3f}, survival "
                f"{run['survival_frames']} frames, {run['fps']:.2f} fps"
            )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
