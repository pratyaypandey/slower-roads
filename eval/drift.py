"""Drift metric: divergence between a model rollout and the oracle's ground truth,
indexed by rollout step (ROADMAP §7, architecture.md §5).

Because the sim is deterministic, given a seed + action sequence we have the exact
future the model should have produced. Drift is how far the model's rollout has
wandered from that future at each step k — the curve that other world-model papers
can't draw. Returned per-step so it plots directly against rollout length, and the
same function re-run under different model sizes / decoding schemes / λ overlays
those curves.

Two spaces (both wanted by §5 / §7):
  - latent L2   : divergence of the dynamics core's own predictions z_hat vs z_gt
  - pixel L1/MSE: divergence of the decoded frames vs the oracle frames

Core is numpy so it's verifiable by passing arrays directly; a torch tensor is
accepted and detached to numpy, so eval code can hand rollouts straight in.
"""

import numpy as np


def _to_numpy(a):
    if hasattr(a, "detach"):  # torch.Tensor
        a = a.detach().cpu().numpy()
    return np.asarray(a, dtype=np.float64)


def _per_step_reduce(pred, gt, reduce):
    """pred, gt: (steps, ...) aligned by rollout step. Returns (steps,)."""
    pred, gt = _to_numpy(pred), _to_numpy(gt)
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs gt {gt.shape}")
    if pred.ndim < 1 or pred.shape[0] == 0:
        raise ValueError("need at least a rollout-step axis with >=1 step")
    diff = pred - gt
    flat = diff.reshape(diff.shape[0], -1)
    if reduce == "l2":       # per-step Euclidean distance
        return np.sqrt(np.sum(flat * flat, axis=1))
    if reduce == "mse":      # per-step mean squared error
        return np.mean(flat * flat, axis=1)
    if reduce == "l1":       # per-step mean absolute error
        return np.mean(np.abs(flat), axis=1)
    raise ValueError(f"unknown reduce {reduce!r}")


def latent_drift(pred_latents, gt_latents):
    """Per-step latent L2 between predicted and ground-truth latent rollouts.

    Arrays shaped (H, ...) — anything after the step axis (tokens, channels) is
    flattened. Returns (H,).
    """
    return _per_step_reduce(pred_latents, gt_latents, "l2")


def pixel_drift(pred_frames, gt_frames, metric="l1"):
    """Per-step pixel divergence of decoded vs oracle frames.

    Frames shaped (H, C, Hpx, Wpx) in [0,1]. metric: 'l1' (MAE) or 'mse'.
    Returns (H,).
    """
    if metric not in ("l1", "mse"):
        raise ValueError(f"pixel metric must be 'l1' or 'mse', got {metric!r}")
    return _per_step_reduce(pred_frames, gt_frames, metric)


def drift_curves(pred_latents=None, gt_latents=None,
                 pred_frames=None, gt_frames=None, pixel_metric="l1"):
    """Bundle the per-step curves that are available into one dict, so a caller
    can sweep model size / decoding / λ and stack the results. Each value is a
    (steps,) array indexed by rollout step."""
    out = {}
    if pred_latents is not None and gt_latents is not None:
        out["latent_l2"] = latent_drift(pred_latents, gt_latents)
    if pred_frames is not None and gt_frames is not None:
        out[f"pixel_{pixel_metric}"] = pixel_drift(pred_frames, gt_frames, pixel_metric)
    if not out:
        raise ValueError("provide at least one of the latent or frame pairs")
    return out
