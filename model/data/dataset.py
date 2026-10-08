"""Dataset over a sim manifest: context window + H future frames as targets.

Implements the model-side data contract in docs/architecture.md (§1 shapes,
§5 rollout loss, §6 scene representation). One item is a length-T context of
(frames, actions[, state]) plus the H future frames the rollout loss regresses
onto, so `target_frames[k]` is `samples[ctx_end + 1 + k]` — the k-th prediction
in the autoregressive rollout.

torch is imported behind a guard: the tensor-producing Dataset needs it, but the
manifest parsing, window indexing, and item assembly are pure-numpy so they can
be verified without torch or PIL (see test_dataset.py).
"""

import json
import os

import numpy as np

# Single source of truth for the §3 9-bucket action scheme lives in the
# dynamics config (torch-free). Import it so the tokens the dataset feeds in and
# the tokens the dynamics core interprets can never silently drift apart.
from model.dynamics.config import (
    G,
    NUM_ACTION_TOKENS as ACTION_VOCAB,
    tokenize_action as _tokenize_action,
)

try:
    import torch
except ImportError:  # torch is optional here; the numpy assembly path stands alone.
    torch = None


# --- action tokenization -----------------------------------------------------
# The sim's action is {steer, throttle}, both in [-1,1] (throttle < 0 = brake).
NEUTRAL_ACTION_TOKEN = _tokenize_action(0.0, 0.0)  # straight + coast (center bucket)


def tokenize_action(action):
    """{steer, throttle} -> int in [0, NUM_ACTION_TOKENS). None -> neutral."""
    if action is None:
        return NEUTRAL_ACTION_TOKEN
    return _tokenize_action(action["steer"], action["throttle"])


def action_driving_frame(samples, frame_idx):
    """Return the logged action that produced ``samples[frame_idx]``.

    The sim records sample i *before* applying ``samples[i]["action"]``; that
    action therefore drives frame/state i -> i+1.  A frame-aligned sequence must
    pair frame i with action i-1.  Frame zero has no incoming action and uses the
    neutral bucket.
    """
    if frame_idx <= 0:
        return None
    return samples[frame_idx - 1]["action"]


# --- manifest + windowing (torch-free) --------------------------------------
# The sim's state nests the car pose under state.car; pull the same 4-vector the
# state-space dynamics model uses. heading/speed live there alongside x/z.
STATE_KEYS = ("x", "z", "heading", "speed")
ANCHOR_CHANNELS = 16


def anchor_grid(skeleton, grid_size=G):
    """Rasterize the cheap sim skeleton into a token-aligned conditioning grid.

    Channels 0..4 are spatial road geometry; 5..15 broadcast global car and
    environment state.  It intentionally contains no rendered pixels.
    """
    out = np.zeros((ANCHOR_CHANNELS, grid_size, grid_size), dtype=np.float32)
    road = skeleton.get("road") or []
    width = float(skeleton.get("width", 8.0))
    max_forward = max([float(p.get("forward", 0.0)) for p in road] + [1.0])
    for p in road:
        forward = max(0.0, float(p.get("forward", 0.0)))
        lateral = float(p.get("lateral", 0.0))
        # Near road is at the bottom. Perspective narrows the corridor with depth.
        y = int(round((1.0 - min(1.0, forward / max_forward)) * (grid_size - 1)))
        lateral_scale = max(width, forward * 0.28)
        x_mid = (0.5 + lateral / (2.0 * lateral_scale)) * (grid_size - 1)
        half = max(0.5, width / (2.0 * lateral_scale) * (grid_size - 1) * 0.5)
        x0 = max(0, int(round(x_mid - half)))
        x1 = min(grid_size - 1, int(round(x_mid + half)))
        x_center = min(grid_size - 1, max(0, int(round(x_mid))))
        out[0, y, x0:x1 + 1] = 1.0
        out[1, y, x_center] = 1.0
        out[2, y, x0:x1 + 1] = np.clip(float(p.get("headingDelta", 0.0)) / np.pi, -1, 1)
        out[3, y, x0:x1 + 1] = np.clip(float(p.get("curvature", 0.0)) * 20.0, -1, 1)
        out[4, y, x0:x1 + 1] = np.clip(float(p.get("grade", 0.0)) * 10.0, -1, 1)

    car = skeleton.get("car") or {}
    env = skeleton.get("env") or {}
    tod = float(env.get("timeOfDay", 0.0))
    globals_ = [
        np.clip(width / 20.0, 0, 1),
        np.clip(float(car.get("speed", 0.0)) / 40.0, 0, 1),
        np.clip(float(car.get("slip", 0.0)), -1, 1),
        float(bool(car.get("grounded", True))),
        np.clip(float(env.get("fog", 0.0)), 0, 1),
        np.clip(float(env.get("rain", 0.0)), 0, 1),
        np.clip(float(env.get("snow", 0.0)), 0, 1),
        np.sin(tod), np.cos(tod),
        np.clip(float(env.get("biomeX", 0.5)), 0, 1),
        np.clip(float(env.get("biomeY", 0.5)), 0, 1),
    ]
    for channel, value in enumerate(globals_, start=5):
        out[channel].fill(value)
    return out


def load_manifest(manifest_path):
    with open(manifest_path) as f:
        manifest = json.load(f)
    return manifest, os.path.dirname(os.path.abspath(manifest_path))


def window_indices(n_samples, context, horizon, sample_range=None):
    """Start indices whose [context | horizon] window fits inside the sequence.

    A window at start s covers context samples s..s+T-1 and target samples
    s+T..s+T+H-1, so targets are exactly the H samples after the context.

    sample_range=(lo, hi) restricts windows to lie *entirely* within samples
    [lo, hi): a start s is kept only if s >= lo and s+context+horizon-1 <= hi-1.
    This is how a train/val split holds out a contiguous slice of one trajectory
    without any window straddling the boundary (leakage-free — see the trainer's
    val split). Default None = the whole sequence.
    """
    lo, hi = (0, n_samples) if sample_range is None else sample_range
    lo = max(0, lo)
    hi = min(n_samples, hi)
    last_start = hi - context - horizon  # inclusive last start within [lo, hi)
    return [s for s in range(lo, last_start + 1)] if last_start >= lo else []


# --- frame / state loading (torch-free) -------------------------------------
def _resize_chw(arr, size):
    _, h, w = arr.shape
    if (h, w) == (size, size):
        return arr
    ys = (np.arange(size) * h // size).clip(0, h - 1)
    xs = (np.arange(size) * w // size).clip(0, w - 1)
    return arr[:, ys][:, :, xs]


def _load_frame_array(path, size):
    """Load a frame as (3, size, size) float32 in [0, 1].

    .npy is read directly (numpy-only, used by the tests). .png needs PIL, which
    the real training box has; here it fails loudly rather than silently.
    """
    if path.endswith(".npy"):
        arr = np.load(path)
    else:
        try:
            from PIL import Image
        except ImportError as e:
            raise RuntimeError(
                f"loading {path} needs PIL; use .npy frames for torch/PIL-free tests"
            ) from e
        arr = np.asarray(Image.open(path).convert("RGB"))

    if arr.ndim == 3 and arr.shape[-1] in (3, 4):  # HWC(A) -> CHW, drop alpha
        arr = np.transpose(arr[..., :3], (2, 0, 1))
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.float32)
    return _resize_chw(arr, size)


def _state_vector(state):
    # New sim nests the car pose under state["car"].
    car = state["car"]
    return np.array([car[k] for k in STATE_KEYS], dtype=np.float32)


def assemble_item(manifest, manifest_dir, ctx_start, context, horizon,
                  representation, frame_size, latents=None, include_anchor=False,
                  include_state=False, packed=None):
    """Build one training item as plain numpy arrays (see module docstring).

    representation='latent' yields precomputed token windows (context_tokens,
    target_tokens) from `latents` (N, tokens) instead of frames — the cache path
    that skips the tokenizer at train time. include_state adds the sim-state
    vectors regardless of representation (the state-continuity head reads them even
    on the latent path)."""
    want_rgb = representation in ("rgb", "both")
    want_state = representation in ("state", "both") or include_state
    want_latent = representation == "latent"
    samples = manifest["samples"]
    ctx = range(ctx_start, ctx_start + context)
    tgt = range(ctx_start + context, ctx_start + context + horizon)

    item = {
        "context_actions": np.array(
            [tokenize_action(action_driving_frame(samples, i)) for i in ctx], dtype=np.int64
        ),
        "target_actions": np.array(
            [tokenize_action(action_driving_frame(samples, i)) for i in tgt], dtype=np.int64
        ),
        "meta": {
            "seed": manifest.get("seed"),
            "ctx_start": ctx_start,
            "ctx_end": ctx_start + context - 1,
            "target_start": ctx_start + context,
        },
    }
    if want_rgb and packed is not None:   # (N,3,H,W) uint8 episode array (model/data/frames.py)
        load = lambda i: _resize_chw(packed[i].astype(np.float32) / 255.0, frame_size)
        item["context_frames"] = np.stack([load(i) for i in ctx])
        item["target_frames"] = np.stack([load(i) for i in tgt])
    elif want_rgb:
        item["context_frames"] = np.stack(
            [_load_frame_array(os.path.join(manifest_dir, samples[i]["frame"]), frame_size) for i in ctx]
        )
        item["target_frames"] = np.stack(
            [_load_frame_array(os.path.join(manifest_dir, samples[i]["frame"]), frame_size) for i in tgt]
        )
    if want_state:
        item["context_state"] = np.stack([_state_vector(samples[i]["state"]) for i in ctx])
        item["target_state"] = np.stack([_state_vector(samples[i]["state"]) for i in tgt])
    if want_latent:
        item["context_tokens"] = latents[list(ctx)].astype(np.int64)   # (T, tokens)
        item["target_tokens"] = latents[list(tgt)].astype(np.int64)    # (H, tokens)
    if include_anchor:
        if any("skeleton" not in samples[i] for i in [*ctx, *tgt]):
            raise ValueError("anchor conditioning needs skeleton entries in every sample")
        item["context_anchor"] = np.stack([anchor_grid(samples[i]["skeleton"]) for i in ctx])
        item["target_anchor"] = np.stack([anchor_grid(samples[i]["skeleton"]) for i in tgt])
    return item


_DatasetBase = torch.utils.data.Dataset if torch is not None else object


class SimSequenceDataset(_DatasetBase):
    """Sliding (context, horizon) windows over one sim manifest.

    representation: 'rgb' | 'state' | 'both' (architecture.md §6). 'rgb'/'both'
    require frames in the manifest; a state-only manifest supports only 'state'.
    """

    def __init__(self, manifest_path, context, horizon, representation="rgb",
                 frame_size=64, sample_range=None, latents_path=None,
                 include_anchor=False, include_state=False):
        if representation not in ("rgb", "state", "both", "latent"):
            raise ValueError(f"unknown representation {representation!r}")
        self.manifest, self.manifest_dir = load_manifest(manifest_path)
        self.context = context
        self.horizon = horizon
        self.representation = representation
        self.frame_size = frame_size
        self.include_anchor = include_anchor
        self.include_state = include_state
        # sample_range=(lo, hi) restricts to a contiguous slice of the trajectory
        # (train/val split); None = all samples.
        self.sample_range = sample_range
        self._starts = window_indices(
            len(self.manifest["samples"]), context, horizon, sample_range
        )

        if representation in ("rgb", "both"):
            if any("frame" not in s for s in self.manifest["samples"]):
                raise ValueError(
                    f"representation {representation!r} needs frames, but this "
                    "manifest has none (state-only export)"
                )
        # Latent cache: precomputed token indices (N, tokens) aligned 1:1 with the
        # manifest samples. Default path is <manifest_dir>/latents.npy.
        self.latents = None
        if representation == "latent":
            path = latents_path or os.path.join(self.manifest_dir, "latents.npy")
            if not os.path.exists(path):
                raise ValueError(
                    f"representation 'latent' needs precomputed {path} "
                    "(run model.precompute_latents)")
            self.latents = np.load(path)
            if len(self.latents) != len(self.manifest["samples"]):
                raise ValueError(
                    f"latents ({len(self.latents)}) and manifest samples "
                    f"({len(self.manifest['samples'])}) length mismatch at {path}")

        # Packed uint8 frames (frames_u8.npy), memory-mapped, if the episode has them.
        self.packed = None
        packed_path = os.path.join(self.manifest_dir, "frames_u8.npy")
        if representation in ("rgb", "both") and os.path.exists(packed_path):
            self.packed = np.load(packed_path, mmap_mode="r")

    def __len__(self):
        return len(self._starts)

    def __getitem__(self, idx):
        item = assemble_item(
            self.manifest, self.manifest_dir, self._starts[idx],
            self.context, self.horizon, self.representation, self.frame_size,
            latents=self.latents, include_anchor=self.include_anchor,
            include_state=self.include_state, packed=self.packed,
        )
        if torch is None:
            return item
        out = {"meta": item.pop("meta")}
        for k, v in item.items():
            out[k] = torch.from_numpy(v)
        return out
