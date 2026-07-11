# Tokenizer A/B — `fsq` (baseline) vs `fsq_v2` (research stack)

First GPU experiment testing the tokenizer improvements from
`docs/tokenizer_research.md`. Run on a Runpod A5000 (~$0.26 total), pulled back here.

## Setup (identical for both, fair A/B)
- Data: 2500 real sim frames, 64×64 RGB, seed 1 (`data/seed1`, autopilot drive).
- Train: 40 epochs, batch 32, Adam lr 3e-4, loss = L1 + 0.5·edge-gradient.
- Only the architecture differs: `fsq` (baseline conv AE, 0.53M) vs `fsq_v2`
  (bottleneck self-attention + GroupNorm/SiLU residual blocks + PixelShuffle/ICNR
  upsampling, 3.14M). Same FSQ quantizer, same 256-token / 12800-vocab contract.

## Results
| | recon L1 (val) | codebook usage (full eval) |
|---|---|---|
| `fsq` baseline | ~0.015–0.029 | 5136 / 12800 (40%) |
| `fsq_v2` | comparable/slightly lower | higher (richer code use) |

The number that matters is **visual**: see `compare_upscaled.png` (×6, nearest).
`fsq_v2` is clearly sharper — the car keeps its windshield/shape (baseline blurs it
to a blob), road edges and lane dashes survive, trees are crisper. The baseline's
per-frame L1 is already low because the flat background dominates the average; the
edges (car, road, lane markings) are where `fsq_v2` wins, exactly as predicted.

Notable finding: codebook usage is only **40%** — the 12800 vocab is oversized for
these low-entropy stylized frames; trimming the FSQ levels would lighten the AR head
(a cheap follow-up, per the research doc).

## Files
- `compare_upscaled.png` — original | fsq | fsq_v2, ×6 upscaled (the headline).
- `fsq_grid.png`, `fsq_v2_grid.png` — eval_tokenizer recon grids (top orig, bottom recon).
- `checkpoints/` — trained weights (gitignored; regenerate on GPU).

## Regenerate
```bash
node sim/headless/generate_pixels.mjs --seed 1 --steps 2500 --size 64     # data
python -m model.train_tokenizer --data data/seed1 --arch fsq    --epochs 40 --grad-weight 0.5 --out ckpt_fsq
python -m model.train_tokenizer --data data/seed1 --arch fsq_v2 --epochs 40 --grad-weight 0.5 --out ckpt_v2
python -m eval.upscale_compare --data data/seed1 \
  --ckpts ckpt_fsq/tokenizer.pt:fsq,ckpt_v2/tokenizer.pt:fsq_v2 \
  --out eval/tokenizer_ab/compare_upscaled.png
```

## Run 2 — loss stack + vocab trim + latency + predictability (RTX 4090)

Four variants, 40 epochs each (LPIPS disabled — its forward crashed on the pod;
saliency-L1 + FFL + edge are the core sharpeners anyway). See `compare_all.png`.

| variant | recon L1 (frame 400) | codebook usage | decode ms/frame (4090, b1) | AR next-tok acc |
|---|---|---|---|---|
| `fsq` (baseline conv) | 0.015 | 40% (5136/12800) | 0.43 | — |
| `v2_edge` (arch, edge loss) | ~0.015 | — | 1.70 | **0.208** |
| `v2_stack` (arch + saliency-L1 + FFL) | **0.0084** | 67% (8635/12800) | 1.71 | 0.161 |
| `v2_stack_4k` (stack + levels 7,5,5,5,5) | 0.0086 | **93% (4054/4375)** | 1.72 | — |

Findings:
- **Loss stack ~halved recon L1** (0.015 → 0.0084) and lifted codebook usage 40%→67%.
  Visibly the sharpest — car windshield, road edges, lane dashes all preserved.
- **Vocab trim to 4375 is free**: 93% utilization, same reconstruction as 12800 → a
  3× lighter AR head at no visual cost. Adopt.
- **Decode latency is a non-issue on GPU**: all variants 0.4–1.7 ms/frame at batch 1,
  ~20–80× under the 33 ms real-time budget. fsq_v2's 5× heavier decoder (1.92M) is
  still trivially fast — **no distillation needed** for the discrete-GPU target; WebGPU
  keeps large margin.
- **Predictability tension** (the tie-breaker): the sharper/richer loss-stack tokens are
  slightly *less* predictable (AR acc 0.161 vs 0.208 for v2_edge) — more information per
  token costs some predictability. Both are ~2000× above chance, so both are very learnable;
  the world-model quality ceiling (sharper frames) likely outweighs the small predictability
  cost, but it's worth confirming once M2 trains on each.

## Run 3 — driving recon L1 to ~0.001 (RTX 3090)

Goal: get `v2_stack` recon L1 from ~0.0084 down to 0.001–0.002. The old 0.0084 came
from a run cut short at 40 epochs (the pod kept billing after the laptop slept). Two
new levers added to the trainer — **weight EMA** (0.999), **cosine LR + warmup** — plus
an **in-RAM frame cache** (`--frame-cache`; ~12× faster training, so 600 epochs is cheap).
Sweep isolates training vs capacity vs bits. Metric is **dataset-mean L1 over all 2501
frames** (not the 6-frame eyeball number). See `compare_lowloss.png`.

| variant | dataset-mean L1 | vs 0.0084 | codebook usage |
|---|---|---|---|
| `v2_stack` (old, 40 ep, no EMA) | ~0.0084 | — | 67% |
| **L1** — +EMA +cosine +600 ep, hidden 64 | **0.00216** | 3.9× | 83.7% (10713/12800) |
| **L2** — + capacity (hidden 128) | **0.00093** | **9×** | 80.7% (10327/12800) |
| L3 — + bits (6-ch FSQ, vocab 102400) | 0.00104 | 8× | **25.6%** (26166/102400) |

Findings:
- **Undertraining, not the bottleneck, was most of the old loss.** Same capacity + EMA +
  cosine + 600 epochs alone: 0.0084 → **0.00216**, already inside target.
- **Capacity is the lever.** Doubling width (hidden 128) → **0.00093**, below 0.001. Winner.
- **More latent bits do NOT help.** L3 matched L2 but used only 25.6% of its 102400 vocab —
  the 256-token / 12800-vocab bottleneck was never the limit. A bloated vocab is strictly
  worse for the AR head, so reject this route.
- **Recommend: `fsq_v2`, hidden 128, EMA + cosine, ~600 epochs** — ~0.0009 L1 (9× sharper),
  same 256-token / 12800-vocab contract M2 expects. Lean checkpoint:
  `checkpoints_lowloss/fsq_v2_h128_best.pt` (gitignored).
- Caveat: measured on seed1 (the training drive). Confirm on a held-out seed — cheap.

## Regenerate (run 3)
```bash
python -m model.train_tokenizer --data data/seed1 --arch fsq_v2 --hidden 128 \
  --loss-stack --lpips-weight 0 --grad-weight 0.5 --frame-cache --cosine --ema 0.999 \
  --epochs 600 --batch-size 64 --out ck_L2
python -m eval.eval_tokenizer --data data/seed1 --ckpt ck_L2/tokenizer.pt   # dataset mean L1
```
