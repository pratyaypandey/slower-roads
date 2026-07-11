# Training the world model

End-to-end runbook: from an empty repo to a world model you can watch dream.
Everything runs without the native `gl` module — frames are rendered in pure
numpy. See `docs/architecture.md` for the design contract.

## The pipeline at a glance

```
                         ACTION a_t (throttle/brake/steer -> 1 of 9 tokens)
                             │
                             ▼
  frame x_t ──▶ ┌───────────────────────┐   z_t (8x8=64 FSQ tokens)
  (3,64,64)     │  FSQ tokenizer         │──────────────┐
                │  (encoder + FSQ + dec) │              │
                └───────────────────────┘              ▼
                        ▲                     ┌───────────────────────┐
                        │ decode              │  AR dynamics core      │
                        │                     │  (causal transformer   │
                        │                     │   over [a_t, z_t...])   │
                  x̂_{t+1} ◀── decode ẑ_{t+1} ─┤  predicts next-frame    │
                  (dreamed frame)             │  tokens ẑ_{t+1}         │
                                              └───────────────────────┘
                                                        │ feed ẑ back
                                                        └──▶ (autoregressive:
                                                             dream t+2, t+3, ...)
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
