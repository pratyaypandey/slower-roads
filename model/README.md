# model/

The model side of Slower Roads: turn sim frames into latent tokens, learn the
action-conditioned dynamics in that latent space, and measure how far a rollout
drifts from the oracle. The shapes and formats every piece agrees on are the
contract in [`docs/architecture.md`](../docs/architecture.md) — read it first;
this README is just the wiring.

## How it fits together

```
frames + actions ─▶ tokenizer ─▶ dynamics ─▶ rollout loss ─▶ drift eval
   (data/)          (tokenizer/)  (dynamics/)   (§5)          (eval/drift.py)
```

- **tokenizer/** (§2) — FSQ autoencoder: `frame (3,64,64) -> z (G*G, C)` on the
  integer grid, and the mirror decoder back to pixels. No codebook, no
  commitment loss. `fsq_variants_study.py` measures Jai's non-uniform-grid idea.
- **dynamics/** (§4) — action-conditioned core predicting `z_{t+1}` from context
  tokens + action token. Branch A (AR transformer, default) and Branch B (flow /
  Schrödinger bridge) share one interface: `predict_next(z_t, u_t) -> z_{t+1}`.
- **data/** — `dataset.py` (this deliverable) reads a sim manifest and yields
  `(context frames+actions, H future frames)` training items, per §5/§6.
- **rollout loss** (§5) — roll the dynamics H steps autoregressively, decode each
  step, compare decoded frames to the H targets. Token CE + decoded-pixel loss.
  This multi-step decoded comparison is the anti-drift signal.
- **eval/drift.py** — per-step divergence of a rollout vs the oracle's ground
  truth (latent L2 + pixel L1/MSE), indexed by rollout step for the drift curve.

## `data/dataset.py`

`SimSequenceDataset(manifest_path, context, horizon, representation, frame_size)`
yields sliding `[context | horizon]` windows. Item `k`'s `target_frames[j]` is
`samples[ctx_end + 1 + j]`, so targets are exactly the H frames the rollout must
predict.

- `context` (T) and `horizon` (H) are configurable (§1).
- Frames load as `(3, frame_size, frame_size)` float32 in `[0,1]`; `frame_size`
  defaults to 64 (§1). The sparse `state` vector `{x,z,heading,speed}` (§6) is
  exposed alongside.
- `representation`: `'rgb'` | `'state'` | `'both'` (§6) — the scene-representation
  knob. A state-only manifest (no frames) supports `'state'`; `'rgb'` is rejected.
- Actions use the 9-bucket tokenization from §3, imported from
  `model/dynamics/config.py` so the dataset and the dynamics core share one
  source and cannot drift apart.

`torch` is a guarded import: the `Dataset` returns tensors when torch is present,
plain numpy dicts otherwise. `test_dataset.py` fabricates a tiny manifest + `.npy`
frames and checks tuple alignment with no torch/PIL:

```
python3 model/data/test_dataset.py
```

Real training frames are `.png` (needs PIL on the GPU box); the tests use `.npy`
so they run in the torch/PIL-free environment.

## `eval/drift.py`

Pass aligned `(H, ...)` rollout arrays; get back per-step curves:

- `latent_drift(pred_latents, gt_latents)` — per-step L2.
- `pixel_drift(pred_frames, gt_frames, metric='l1'|'mse')` — per-step pixel error.
- `drift_curves(...)` — bundles whichever pairs you pass, so a caller sweeps
  model size / decoding scheme / λ and stacks the curves (§7).

Accepts numpy arrays or torch tensors (tensors are detached to numpy).

```
python3 eval/test_drift.py
```

## Generate data

There is no data on disk by default (headless `gl` isn't available everywhere).
Produce a dataset from the sim:

```
cd sim && npm install
npm run gen -- --seed 1 --steps 300 --res 128x128
# writes ../data/seed1/frames/*.png + manifest.json
```

Point the dataset at `data/seed1/manifest.json`. The manifest schema and the
`(frame, action, next_frame)` tuple definition are in
[`sim/README.md`](../sim/README.md).

## Run training (on Jai's GPU box, once torch is available)

The environment here has numpy but no torch/PIL, so only the numpy-verifiable
paths run locally. On the GPU machine with torch + PIL installed:

1. Generate data (above).
2. `SimSequenceDataset('data/seed1/manifest.json', context=T, horizon=H,
   representation='rgb')` → `torch.utils.data.DataLoader`.
3. Train tokenizer to clean reconstruction (M1), then the dynamics core with the
   §5 rollout loss (M2 → M3).
4. Evaluate with `eval/drift.py` against oracle rollouts; plot the drift curve
   vs rollout length, model size, and λ (§7).
