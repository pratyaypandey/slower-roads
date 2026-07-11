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
