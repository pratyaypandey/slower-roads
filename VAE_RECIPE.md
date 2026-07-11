# VAE / Tokenizer Recipe — Slower Roads M1

The world model tokenizes each 64×64 driving frame into a small grid of discrete
codes; the M2 dynamics model then learns to roll those codes forward under steering.
This doc is the **recipe** for the tokenizer we shipped, plus an **addendum** of the
A/B experiments that got us there (kept for a future blog post / side-by-side).

TL;DR: an **FSQ convolutional autoencoder** (`fsq_v2`), 256 tokens/frame over a
12800-code vocabulary, reaches **~0.0009 mean pixel-L1** reconstruction — visually
near-lossless on the stylized sim frames — at 3ms decode, well inside real-time.

---

## The recipe (what to reproduce)

### Data
- **Source:** the in-repo Three.js driving sim, captured headless via WebGL.
  `node sim/headless/generate_pixels.mjs --seed 1 --steps 2500 --size 64`
- **Set:** `data/seed1` — 2501 frames, `(3,64,64)` float32 in `[0,1]`, one autopilot
  drive (seed 1). Deterministic sim ⇒ unlimited more data by changing `--seed`.
- Reconstruction only needs individual frames; the tokenizer trains on the flat set
  of frames (not sequence windows).

### Architecture — `fsq_v2` (`model/tokenizer/fsq_v2.py`)
- Conv encoder/decoder, **hidden width 128**, GroupNorm + SiLU **residual blocks**.
- **Bottleneck self-attention** (one `AttnBlock`, `F.scaled_dot_product_attention`).
- **PixelShuffle + ICNR** upsampling in the decoder (replaces ConvTranspose ⇒ no
  checkerboarding).
- **Sigmoid** output (frames live in `[0,1]`).
- **FSQ quantizer** (Finite Scalar Quantization), levels **[8,8,8,5,5] = 12800 codes**,
  on a **16×16 grid ⇒ 256 tokens/frame**. No learnable codebook, no
  commitment/collapse — the AR head just predicts one of 12800 ids per position.
- ~12M params. Decoder ~2ms/frame on a discrete GPU (≪ 33ms budget ⇒ no distillation).

### Loss stack (`model/tokenizer/losses.py`)
Sharpen without a GAN's hallucination risk:
- **Saliency-weighted L1** — weight the loss by the image's own gradient magnitude,
  so the car / road edges / lane dashes (a few % of pixels) aren't drowned out by the
  flat sky/ground that dominates a plain L1 average.
- **Focal Frequency Loss** — penalize the missing high-frequency energy directly in
  the 2D DFT, focally weighting the frequencies the recon gets most wrong.
- **Edge/gradient** term (weight 0.5).
- LPIPS is wired in but **off** (its VGG forward crashed on the training pods;
  the three terms above are the workhorses).

### Training (`model/train_tokenizer.py`)
- **EMA of weights** (decay 0.999) — eval/checkpoint use the EMA copy. Single biggest
  "free" quality bump.
- **Cosine LR** decay to 5% with linear **warmup** (`--cosine`, `--warmup-frac 0.03`).
- **In-RAM frame cache** (`--frame-cache`) — load all unique frames into a GPU tensor
  and sample minibatches directly. ~12× faster than the disk-backed dataset (which was
  I/O-bound re-reading ~10× redundant window frames), so long runs are cheap.
- Adam (β 0.9/0.95), lr 3e-4, **batch 64**, **600 epochs**.

```bash
python -m model.train_tokenizer --data data/seed1 --arch fsq_v2 --hidden 128 \
  --loss-stack --lpips-weight 0 --grad-weight 0.5 \
  --frame-cache --cosine --ema 0.999 --epochs 600 --batch-size 64 \
  --out checkpoints
python -m eval.eval_tokenizer --data data/seed1 --ckpt checkpoints/tokenizer.pt   # dataset mean L1
```

Shipped checkpoint: `checkpoints/tokenizer.pt` (the canonical path M2 loads by default).

### Inference
`eval/serve.py` loads any checkpoint and exposes `encode`/`decode`/`roundtrip`, either
as a local self-test or a stdlib HTTP service (see the file header, and the RunPod
network-volume workflow below).

---

## Addendum — the experiments (blog-post material)

Three GPU rounds, each a fair A/B. Full tables + images in
`eval/tokenizer_ab/README.md`; archived weights in
`eval/tokenizer_ab/checkpoints_lowloss/` and on the RunPod volume `sr-models`.

### Run 1 — architecture (baseline `fsq` vs `fsq_v2`)
Same FSQ contract, only the conv net differs. `fsq_v2` (attention + residual +
PixelShuffle, 3.1M) is **visibly sharper** than the baseline (0.53M) — the car keeps
its windshield/shape, road edges and lane dashes survive — at similar pixel-L1 (the
average is dominated by the easy flat background). Codebook usage rose too.
Image: `eval/tokenizer_ab/compare_upscaled.png`.

### Run 2 — loss stack, vocab trim, latency, predictability
| variant | recon L1 | codebook use | decode ms (b1) | AR next-tok acc |
|---|---|---|---|---|
| `fsq` baseline | 0.015 | 40% | 0.43 | — |
| `v2_edge` (arch + edge) | ~0.015 | — | 1.70 | **0.208** |
| `v2_stack` (arch + saliency-L1 + FFL) | **0.0084** | 67% | 1.71 | 0.161 |
| `v2_stack_4k` (stack + levels 7,5,5,5,5) | 0.0086 | **93% of 4375** | 1.72 | — |

- Loss stack ~halved L1 (0.015→0.0084) and lifted usage 40%→67%.
- Trimming vocab to 4375 is ~free (93% used, same recon) — a lighter AR head if wanted.
- Decode latency is a non-issue on GPU (all ≪ 33ms) ⇒ no distillation needed.
- Predictability tension: sharper tokens are slightly *less* AR-predictable (0.161 vs
  0.208) — more info per token costs some predictability. Both ~2000× above chance.
Image: `eval/tokenizer_ab/compare_all.png`.

### Run 3 — driving L1 to ~0.001 (the shipped recipe)
Metric = **dataset-mean L1 over all 2501 frames**. Isolates training vs capacity vs bits.
| variant | dataset-mean L1 | vs 0.0084 | codebook use |
|---|---|---|---|
| `v2_stack` (old, 40 ep, no EMA) | ~0.0084 | — | 67% |
| **L1** — +EMA +cosine +600 ep, hidden 64 | **0.00216** | 3.9× | 83.7% |
| **L2** — + capacity (hidden 128) ← shipped | **0.00093** | **9×** | 80.7% |
| L3 — + bits (6-ch FSQ, vocab 102400) | 0.00104 | 8× | **25.6%** |

- **Undertraining, not the bottleneck, was most of the old loss** — same capacity +
  EMA + cosine + 600 epochs alone hit 0.00216. (The old 0.0084 run was cut to 40
  epochs when a laptop slept while the pod kept billing.)
- **Capacity is the lever** — hidden 64→128 → 0.00093, below the 0.001–0.002 target.
- **More latent bits did not help** — L3 matched L2 but used 25.6% of a 4× vocab; the
  256-token/12800-vocab bottleneck was never the limit, and a bloated vocab only hurts
  the AR head. Rejected.
Image: `eval/tokenizer_ab/compare_lowloss.png`.

Caveat across all runs: L1 is measured on seed1 (the training drive). A held-out seed
would confirm generalization — cheap follow-up. The decisive test remains training M2
on these tokens and eyeballing a dream rollout.

### Archived weights (for the side-by-side)
`eval/tokenizer_ab/checkpoints_lowloss/` (gitignored; also on RunPod volume `sr-models`):
- `fsq_v2_h128_best.pt` — the shipped tokenizer (lean, 50MB) = `checkpoints/tokenizer.pt`.
- `exp_L2_hidden128_winner_full.pt` — same, with optimizer state (resumable).
- `exp_L1_hidden64_ema_cosine_600ep.pt` — the same-capacity training-only point (0.0022).
- L3 (more-bits) weights not archived — its scp truncated; the number (0.00104) stands.
