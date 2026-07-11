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
