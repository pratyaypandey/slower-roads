# Training the world model

End-to-end runbook: from an empty repo to a world model you can watch dream.
Everything runs without the native `gl` module — frames are rendered in pure
numpy. See `docs/architecture.md` for the design contract.

## The pipeline at a glance

Two networks, trained in sequence, that behave differently at train vs.
inference time — the two diagrams below make that split explicit.

### Train time

The tokenizer is frozen. Every real frame in a window is encoded to tokens once;
the dynamics core predicts each next-frame's tokens, and the loss compares them
to the *real* next frame — in token space (cross-entropy) and in pixel space
(decode the prediction, compare to the true frame). Ground-truth frames are
always available, so each step is scored against truth (the rollout can feed
predictions or truth back — see `--tf-start` in train_dynamics).

```
  real frames  x_0 ... x_T (a window of ground truth)
       │
       ▼  (FSQ encoder + quantize, FROZEN)
  tokens       z_0 ... z_T          actions a_0 ... a_T (each -> 1 of 9 tokens)
       │                                   │
       └──────────────┬────────────────────┘
                      ▼
        interleave [a_0,z_0(64), a_1,z_1(64), ...]
                      │
                      ▼
        ┌─────────────────────────────┐
        │  AR dynamics (causal xformer)│  predicts ẑ_{t+1} for every t
        └─────────────────────────────┘
                      │
          ┌───────────┴────────────┐
          ▼                         ▼
   token CE loss             decode ẑ_{t+1} (FROZEN decoder) -> x̂_{t+1}
   ẑ_{t+1} vs z_{t+1}                        │
          │                                  ▼
          │                          pixel loss  x̂_{t+1} vs REAL x_{t+1}
          └──────────────┬───────────────────┘
                         ▼
              total loss = CE + pixel   ── backprop ──▶ dynamics weights only
```

### Inference time (the dream)

No ground-truth future exists — the model **generates it**. A few real frames
seed the context; from then on the dynamics core samples the next frame's tokens
autoregressively (KV-cached), the frozen decoder turns them into pixels, and
those predicted tokens are appended to the context to drive the next step. The
model drives on its own imagination. This free-running feedback is where drift
appears — small errors compound because nothing pulls the state back to truth.

```
  seed: real frames x_0..x_{T-1} ──▶ encode ──▶ context tokens [a,z,a,z,...]
                                                        │
        ┌───────────────────────────────────────────────┘
        ▼
  ┌─────────────────────────────┐   ẑ_{t+1}      ┌──────────────────┐  x̂_{t+1}
  │  AR dynamics (KV-cached)     │──64 tokens────▶│ FSQ decoder      │────────▶ screen
  │  + action a_t               │                │ (FROZEN)         │
  └─────────────────────────────┘                └──────────────────┘
        ▲                                                │
        │            append (a_t, ẑ_{t+1}) to context    │
        └────────────────────────────────────────────────┘
                     autoregressive: dream t+1, t+2, t+3, ...
                     (no ground truth — errors compound = drift)
```

Two networks, trained in sequence:

- **Tokenizer (M1)** — a convolutional autoencoder with an FSQ bottleneck
  (~0.86M params). Compresses each 64×64 frame to 64 discrete tokens and decodes
  them back. Trained first, then **frozen**.
- **Dynamics core (M2)** — a causal Transformer (~9.85M params) over the
  interleaved sequence `[a_0, z_0(64 tokens), a_1, z_1, ...]`. Predicts the next
  frame's tokens from the past. Trained on the frozen tokenizer's latents with a
  multi-step rollout loss (token cross-entropy + decoded-pixel loss).

At inference the dynamics core generates tokens autoregressively and the
tokenizer's decoder turns them into pixels — the model "dreams" frame by frame.

```
        MODULE               PARAMS   INPUT              OUTPUT
  ┌────────────────────┐
  │ FSQ encoder        │    ~0.40M   (B,3,64,64) frame   (B,64,5) continuous
  │ FSQ quantize       │      0       (B,64,5)           (B,64) code indices
  │ FSQ decoder        │    ~0.46M   (B,64) indices      (B,3,64,64) frame
  ├────────────────────┤
  │ AR dynamics        │    ~9.85M   (B, T*65) tokens    (B, T*65, 12809) logits
  │  (embed + N blocks │            [a,z,a,z,...]        next-token distribution
  │   + head, KV-cache)│
  └────────────────────┘
  vocab = 12800 visual codes (8·8·8·5·5) + 9 action tokens = 12809
```

## Prerequisites

Python with PyTorch matching your CUDA driver. On the H100 box (driver CUDA 12.8):

```bash
conda create -n slowroads python=3.12 -y && conda activate slowroads
pip install torch numpy pillow matplotlib --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print('sees GPU:', torch.cuda.is_available())"   # want True
```

On a shared box, pin to a free GPU (check `nvidia-smi` for one at low util) and
prefix every training/eval command with `CUDA_VISIBLE_DEVICES=<n>`.

## Step 0 — generate + render data (CPU, no gl)

```bash
# road-following drive: the car steers to stay on the road (renderable frames)
node sim/headless/generate_drive.js --seed 1 --steps 3000
# render each state to a (3,64,64) .npy frame -> a drop-in 'rgb' dataset
python -m sim.render.render_manifest --data data/seed1_drive --size 64
```

For a bigger/varied dataset, generate several seeds into separate dirs and
render each; train on whichever `--data` dir you want.

## Step 1 — tokenizer (M1)

```bash
CUDA_VISIBLE_DEVICES=7 python -m model.train_tokenizer \
    --data data/seed1_drive --epochs 20 --out checkpoints
# -> checkpoints/tokenizer.pt   (watch recon_l1 fall)

# eval: original vs FSQ reconstruction grid + codebook usage
CUDA_VISIBLE_DEVICES=7 python -m eval.eval_tokenizer \
    --data data/seed1_drive --ckpt checkpoints/tokenizer.pt
# -> eval/plots/tokenizer_recon.png
```

Judge by the recon grid, not just the loss: reconstructions should preserve the
road, and codebook usage should be more than a handful of the 12800 codes (low
usage = collapse, which a low loss on easy frames can hide).

## Step 2 — dynamics core (M2)

```bash
CUDA_VISIBLE_DEVICES=7 python -m model.train_dynamics \
    --data data/seed1_drive --tokenizer checkpoints/tokenizer.pt \
    --epochs 20 --out checkpoints
# -> checkpoints/dynamics.pt   (watch ce + pixel fall)

# eval: THE payoff — a GIF of the model dreaming vs ground truth
CUDA_VISIBLE_DEVICES=7 python -m eval.eval_dream \
    --data data/seed1_drive --tokenizer checkpoints/tokenizer.pt \
    --dynamics checkpoints/dynamics.pt --steps 60
# -> eval/plots/dream.gif   (left = ground truth, right = dreamed)
```

The dreamed road should advance/curve like the real one for a stretch, then
gradually degrade — that gradual divergence is drift, the thing later milestones
(the λ sim-anchor, anti-drift training) exist to fight.

## Continuing from a checkpoint

Both trainers checkpoint every epoch and take `--resume`. `--epochs` is the
**total** epoch count, so training continues from the saved epoch up to it:

```bash
# tokenizer: trained 20, want 20 more (i.e. up to epoch 40)
CUDA_VISIBLE_DEVICES=7 python -m model.train_tokenizer \
    --data data/seed1_drive --resume checkpoints/tokenizer.pt --epochs 40

# dynamics: same pattern (the frozen tokenizer is always loaded fresh via --tokenizer)
CUDA_VISIBLE_DEVICES=7 python -m model.train_dynamics \
    --data data/seed1_drive --tokenizer checkpoints/tokenizer.pt \
    --resume checkpoints/dynamics.pt --epochs 40
```

A resume restores model + optimizer + epoch, so it's identical to never having
stopped (modulo dataloader shuffle order). If a run is interrupted, just
`--resume` the same checkpoint.

## Full clean run, copy-paste

```bash
conda activate slowroads
node sim/headless/generate_drive.js --seed 1 --steps 3000
python -m sim.render.render_manifest --data data/seed1_drive --size 64
CUDA_VISIBLE_DEVICES=7 python -m model.train_tokenizer --data data/seed1_drive --epochs 20
CUDA_VISIBLE_DEVICES=7 python -m eval.eval_tokenizer   --data data/seed1_drive --ckpt checkpoints/tokenizer.pt
CUDA_VISIBLE_DEVICES=7 python -m model.train_dynamics  --data data/seed1_drive --tokenizer checkpoints/tokenizer.pt --epochs 20
CUDA_VISIBLE_DEVICES=7 python -m eval.eval_dream       --data data/seed1_drive --tokenizer checkpoints/tokenizer.pt --dynamics checkpoints/dynamics.pt --steps 60
```

## Knobs worth touching

| flag | where | why |
|------|-------|-----|
| `--epochs` | both | more passes; use with `--resume` to extend |
| `--batch-size` | both | H100 has headroom — raise it (e.g. 64/32) to train faster |
| `--d-model / --n-layers` | dynamics | bigger dynamics core = more capacity, slower |
| `--horizon` | dynamics | rollout length the loss covers; longer = harder, less drift |
| `--size` | render | frame resolution; 64 default, drop to 32 for speed |
| `--steps` | eval_dream | how many frames to dream before stopping |
