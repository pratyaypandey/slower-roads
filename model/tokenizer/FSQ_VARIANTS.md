# FSQ Variants Study — Does a non-uniform quantizer beat the uniform grid?

## The intuition being tested

From `jais_notes/notes.md`:

> "I was thinking about how FSQ is a grid, but we should be able to do better
> than that maybe. I was thinking about AEP from information theory, and I was
> wondering if there are certain embeddings we think would be more likely (e.g.
> Gaussian or closer to the origin). Experiment with different FSQ variants and
> see which one would be best for this project."

The hypothesis: a uniform grid spends equal code density everywhere. If the
encoder's latent is Gaussian-ish (mass near the origin), a uniform grid wastes
codes in low-probability corners. By AEP / rate-distortion / companding theory,
a quantizer whose cells are dense where the source is dense should give lower
distortion at the same bitrate.

## Setup

`fsq_variants_study.py` (numpy only) compares three scalar-quantizer variants.
FSQ is channel-independent, so a per-channel scalar analysis is exact and the
multi-channel bitrate is just `sum(log2(levels))`. All variants are measured the
same way: quantize an iid unit-variance source into `L` cells, reconstruct each
cell to its **centroid** (the conditional mean — what a trained decoder
converges to), and report source-space MSE. Rate = `log2(L)` bits/channel.

- **UniformFSQ** (baseline): standard FSQ — `tanh`-bound then uniform levels.
  The uniform grid in `tanh` space maps back to source space via `atanh`.
- **Companded**: equiprobable cells (decision thresholds at source quantiles).
  This is the classic companding solution — warp through the source CDF, then
  quantize uniformly — so every code is equally likely.
- **LloydMax**: MSE-optimal non-uniform scalar quantizer (Lloyd's algorithm on
  the empirical source). This is the achievable lower bound, not a drop-in.

Sources: standard Gaussian, Laplacian, uniform (all unit variance),
n = 400,000 samples, levels L ∈ {5, 8, 16}. `util%` = code entropy / `log2(L)`
(100% = every code used equally often; low = wasted codes).

## Measured results

| source | L | bits | UniformFSQ MSE | util% | Companded MSE | util% | LloydMax MSE | util% |
|---|---|---|---|---|---|---|---|---|
| gaussian | 5 | 2.32 | 1.264e-01 | 99.0 | 1.033e-01 | 100.0 | 8.027e-02 | 94.9 |
| gaussian | 8 | 3.00 | 7.371e-02 | 99.0 | 5.524e-02 | 100.0 | 3.466e-02 | 94.2 |
| gaussian | 16 | 4.00 | 3.221e-02 | 99.2 | 2.235e-02 | 100.0 | 9.539e-03 | 94.4 |
| laplacian | 5 | 2.32 | 2.001e-01 | 99.6 | 2.117e-01 | 100.0 | 1.207e-01 | 83.8 |
| laplacian | 8 | 3.00 | 1.342e-01 | 99.5 | 1.338e-01 | 100.0 | 5.503e-02 | 85.6 |
| laplacian | 16 | 4.00 | 7.757e-02 | 99.5 | 6.750e-02 | 100.0 | 1.564e-02 | 87.7 |
| uniform | 5 | 2.32 | 6.115e-02 | 94.9 | 3.997e-02 | 100.0 | 3.997e-02 | 100.0 |
| uniform | 8 | 3.00 | 2.673e-02 | 95.3 | 1.558e-02 | 100.0 | 1.557e-02 | 100.0 |
| uniform | 16 | 4.00 | 6.723e-03 | 96.4 | 3.907e-03 | 100.0 | 3.906e-03 | 100.0 |

### Distortion reduction vs UniformFSQ (dB, higher = better)

| source | L | Companded | LloydMax |
|---|---|---|---|
| gaussian | 5 | +0.88 | +1.97 |
| gaussian | 8 | +1.25 | +3.28 |
| gaussian | 16 | +1.59 | +5.28 |
| laplacian | 5 | **−0.24** | +2.19 |
| laplacian | 8 | +0.01 | +3.87 |
| laplacian | 16 | +0.60 | +6.95 |
| uniform | 5 | +1.85 | +1.85 |
| uniform | 8 | +2.34 | +2.35 |
| uniform | 16 | +2.36 | +2.36 |

## What the numbers say

1. **Jai's direction is correct, but the magnitude is modest.** For a Gaussian
   source, companding matched to the density lowers MSE by ~0.9–1.6 dB at fixed
   bitrate, growing with L. Lloyd-Max (the ceiling) shows 2–5 dB is theoretically
   on the table for Gaussian and up to ~7 dB for the heavier-tailed Laplacian.

2. **The `tanh` bound in standard FSQ is already doing most of the companding.**
   This is the key finding. Under a Gaussian source, UniformFSQ's code
   utilization is ~99% — the "wasted codes in the corners" the intuition warns
   about barely happens, because `tanh` already warps the uniform grid to
   concentrate cells near the origin. The intuition is much more visible against
   a *uniform* source (util drops to 95–96%, and companding buys a clean
   1.9–2.4 dB) precisely because there `tanh` warps a flat density the wrong way.
   So standard FSQ is not a naive uniform grid — it is a mild fixed compander.

3. **Equiprobable companding is not universally better.** For a Laplacian (heavy
   tails) at low L it is actually *worse* than UniformFSQ (−0.24 dB at L=5).
   Forcing equal code probability over-allocates cells to the tails where large
   errors are cheap in MSE terms. Companding is only near-optimal when the
   source is close to the density the companding law assumes.

4. **Lloyd-Max wins everywhere but is source-specific and non-differentiable.**
   Its levels depend on the exact source distribution and it has no closed-form,
   straight-through-friendly forward map, so it is a benchmark for the ceiling,
   not a practical FSQ bottleneck for a jointly-trained encoder.

## Recommendation

**Keep standard (uniform-grid + `tanh`) FSQ as the default bottleneck.** The
headline reason is that FSQ's core advantage is that the *encoder adapts to a
fixed, simple grid during training* — the codebook is not learned, so there is
nothing to collapse. The measured gain from swapping in a fancier fixed
quantizer is only ~1–1.6 dB under Gaussian latents, and that gap will shrink
further once the encoder is free to shape its own latent distribution to
whatever grid it is given (which a raw open-loop source analysis cannot capture).
`tanh` already supplies most of the density-matching benefit for free, at ~99%
code utilization.

**If, after the tokenizer is training, we measure meaningful under-utilization
(entropy well below `log2(codebook)`) or want to squeeze distortion at a fixed
bitrate, add a fixed compander** — warp each channel through an
`erf`/CDF-style law before the uniform round. It is a cheap, differentiable
(straight-through) change and bought a clean, reliable win here for
Gaussian/uniform-ish sources. But validate it per-source: it can regress on
heavy-tailed latents at low L, and it should be chosen by measuring the actual
latent histogram, not assumed.

**Do not ship Lloyd-Max.** Treat its numbers (2–7 dB) only as the ceiling that
tells us how much room is left — realistically we will capture a fraction of it,
and only if utilization measurements justify the added complexity.

Net: Jai was right that we can beat a pure uniform grid, but standard FSQ is
already a mild compander via `tanh`, so the practical prize is small (~1 dB)
unless the trained latent turns out to be far from the grid `tanh` assumes.

## Reproduce

```
python3 model/tokenizer/fsq_variants_study.py
```

Prints the comparison table and writes `fsq_variants_results.txt`. Seeded
(`np.random.default_rng(0)`), so numbers are deterministic.
